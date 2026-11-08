r"""Citation post-processing for agent answers.

PaperQA formats inline citations as ``(docname pages X-Y)`` where docname is
the PDF filename stem -- so a file named ``6.pdf`` produces the unreadable
"(6 pages 5-6)".  This module rewrites the *printed* answer (session.answer)
before display:

* every inline ``(name pages X-Y)`` citation is replaced by a readable
  ``作者 年份(PDF第X-Y页)`` label;
* numeric/opaque docnames are resolved to real metadata (title/authors/year)
  by reading the first two PDF pages with pypdf (no re-indexing, no PDF
  modification) and, optionally, one small LLM extraction call per paper;
  results are cached in ``index/citation_manifest.json`` so each paper is
  resolved at most once;
* when metadata cannot be resolved, the full filename and path are kept and
  explicitly marked 元数据缺失;
* PDF 顺序页码 (the only page identity PaperQA tracks) is labelled as such;
  a printed page number is added only when confidently detected at the edge
  of the underlying chunk text;
* a rebuilt 参考文献 list -- derived from the citations actually present in
  the body -- replaces PaperQA's own list, so body and list always agree.

Everything except the optional LLM extraction is offline and unit-testable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from paper_ledger import normalize_title

# Matches e.g. "6 pages 5-6", "smith2017 pages 3", "李华2012 pages 11-12".
# Docnames are filename stems, so exclude comma/parens (citation separators).
_INLINE_RE = re.compile(
    r"(?P<name>[^(),;\n]{1,120}?) pages (?P<pages>\d+(?:\s*[-–]\s*\d+)?)"
)

_MANIFEST_VERSION = 1


@dataclass
class MetaEntry:
    docname: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    doi: str | None = None
    source_path: str | None = None
    missing: bool = False  # True when nothing readable could be extracted

    @property
    def label(self) -> str | None:
        """Readable short label like 'Smith 2017' or '李华 2012'."""
        if self.missing:
            return None
        author = self.authors[0] if self.authors else None
        if author and self.year:
            return f"{author} {self.year}"
        if author:
            return author
        if self.title:
            t = self.title if len(self.title) <= 40 else self.title[:40] + "…"
            return f"《{t}》" + (f"({self.year})" if self.year else "")
        return None


def load_manifest(path: Path) -> dict[str, MetaEntry]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupt cache is rebuilt lazily
        return {}
    if data.get("version") != _MANIFEST_VERSION:
        return {}
    return {k: MetaEntry(**v) for k, v in data.get("entries", {}).items()}


def save_manifest(path: Path, entries: dict[str, MetaEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _MANIFEST_VERSION,
        "entries": {k: asdict(v) for k, v in sorted(entries.items())},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def needs_resolution(docname: str) -> bool:
    """Numeric or hash-like docnames carry no readable information."""
    stem = docname.strip()
    return (
        bool(re.fullmatch(r"\d+", stem))
        or bool(re.fullmatch(r"[0-9a-f]{16,}", stem.lower()))
        or len(stem) <= 2
    )


def _find_pdf(docname: str, paper_dir: Path) -> Path | None:
    try:
        return next(paper_dir.rglob(f"{docname}.pdf"))
    except StopIteration:
        return None


def read_first_pages(pdf_path: Path, n_pages: int = 2) -> str:
    """Extract text of the first N pages (read-only; PDF is never modified)."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages[:n_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page is not fatal
            continue
    return "\n".join(parts)


def heuristic_meta(docname: str, first_text: str, pdf_path: Path | None) -> MetaEntry:
    """Best-effort offline metadata from first-page text and PDF properties."""
    entry = MetaEntry(docname=docname, source_path=str(pdf_path) if pdf_path else None)
    if pdf_path is not None:
        try:
            from pypdf import PdfReader

            props = PdfReader(str(pdf_path)).metadata or {}
            raw_title = (props.get("/Title") or "").strip()
            if raw_title and len(raw_title) >= 6 and not raw_title.endswith(".pdf"):
                entry.title = raw_title
            raw_author = (props.get("/Author") or "").strip()
            if raw_author and len(raw_author) <= 80:
                entry.authors = [a.strip() for a in re.split(r"[,;，；]", raw_author) if a.strip()]
        except Exception:  # noqa: BLE001
            pass
    text = unicodedata.normalize("NFKC", first_text or "")
    if entry.year is None:
        m = re.search(r"\b(19|20)(\d{2})\b", text)
        if m:
            entry.year = m.group(0)
    if not entry.title:
        headerish = re.compile(
            r"(journal|issn|vol\.|volume|www\.|https?://|doi[.:]|©|copyright)",
            flags=re.IGNORECASE,
        )
        for line in (ln.strip() for ln in text.splitlines()):
            # First substantial line that is not a journal/header fragment.
            if (
                12 <= len(line) <= 200
                and not re.fullmatch(r"[\d\W_]+", line)
                and not headerish.search(line)
            ):
                entry.title = line
                break
    doi = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    if doi:
        entry.doi = doi.group(0).rstrip(".")
    if not entry.title and not entry.authors:
        entry.missing = True
    return entry


_LLM_EXTRACT_PROMPT = (
    "以下是某文献前两页的文本。提取书目信息,只输出 JSON:"
    '{"title": 完整题名或null, "authors": [前三位作者的姓或全名], '
    '"year": "出版年份或null", "doi": "DOI或null"}。'
    "无法确认的字段用 null,不要编造。\n\n文本:\n"
)


async def default_llm_extract(first_text: str) -> dict[str, Any]:
    """One small DeepSeek call per paper; result is cached in the manifest."""
    import os

    import litellm

    response = await litellm.acompletion(
        model=f"openai/{os.environ['DEEPSEEK_MODEL']}",
        api_base=os.environ["DEEPSEEK_API_BASE"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        messages=[
            {"role": "user", "content": _LLM_EXTRACT_PROMPT + first_text[:4000]}
        ],
    )
    content = response.choices[0].message.content or ""
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def _docname_head(docname: str) -> str:
    """Leading surname token of a filename stem, e.g. 'doe'."""
    return re.split(r"[-_]", docname, maxsplit=1)[0].strip().lower()


def _author_head_mismatch(docname: str, entry: MetaEntry) -> bool:
    """heuristic-extracted authors contradict the filename's surname token.

    Real case: one corpus PDF yielded author "David Christian
    Rose" (a copyright/editor line) while the filename stem starts with the
    real first author.  Cached manifest entries with such a mismatch are
    re-resolved (LLM re-extraction when available); offline, the suspect
    authors are dropped rather than shown -- never a guessed author.
    """
    if entry.missing or not entry.authors:
        return False
    head = _docname_head(docname)
    if len(head) < 4:
        return False
    blob = " ".join(entry.authors).lower()
    return head not in blob


async def resolve_metadata(
    docnames: Iterable[str],
    paper_dir: Path,
    manifest: dict[str, MetaEntry],
    llm_extract: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    resolve_all: bool = False,
) -> dict[str, MetaEntry]:
    """Resolve metadata, mutating+returning the manifest.

    By default only opaque docnames (numeric/hash-like) are resolved;
    ``resolve_all=True`` also resolves readable filename stems, which is
    what lets the answer fill "元数据缺失" markers from the paper's own
    first pages instead of leaving truncated filenames.  Cached entries
    whose authors contradict the filename surname are re-resolved instead
    of being trusted forever.
    """
    for docname in dict.fromkeys(docnames):  # dedupe, keep order
        cached = manifest.get(docname)
        retry = cached is not None and _author_head_mismatch(docname, cached)
        if cached is not None and not retry:
            continue
        if cached is None and not resolve_all and not needs_resolution(docname):
            continue
        pdf = _find_pdf(docname, paper_dir)
        first_text = ""
        if pdf is not None:
            try:
                first_text = await __import__("anyio").to_thread.run_sync(
                    read_first_pages, pdf
                )
            except Exception:  # noqa: BLE001 - unreadable PDF -> missing meta
                first_text = ""
        entry = heuristic_meta(docname, first_text, pdf)
        if llm_extract is not None and first_text:
            try:
                data = await llm_extract(first_text)
            except Exception:  # noqa: BLE001 - offline/test envs: heuristics only
                data = {}
            if data.get("title"):
                entry.title = str(data["title"]).strip()
            if data.get("authors"):
                entry.authors = [str(a).strip() for a in data["authors"] if a]
            if data.get("year"):
                entry.year = str(data["year"]).strip()
            if data.get("doi"):
                entry.doi = str(data["doi"]).strip()
            entry.missing = not (entry.title or entry.authors)
        elif retry:
            # Offline re-resolve: keep the verified title/year but drop the
            # contradicted authors instead of showing a wrong one.
            entry = cached
            entry.authors = []
        manifest[docname] = entry
    return manifest


# --- page handling ------------------------------------------------------------

def detect_printed_page(chunk_text: str, pdf_page: int) -> int | None:
    """Conservatively detect a printed page number at a chunk's text edges."""
    if not chunk_text:
        return None
    edges = chunk_text[:120] + "\n" + chunk_text[-120:]
    candidates = []
    for m in re.finditer(r"(?:^|\n)\s*(\d{1,4})\s*(?=\n|$)", edges):
        n = int(m.group(1))
        # Printed numbers stay within a plausible offset of the PDF sequence.
        if 1 <= n <= pdf_page + 2 and abs(n - pdf_page) <= 30:
            candidates.append(n)
    return candidates[0] if candidates else None


def format_pages(pages: str, printed: int | None) -> str:
    pages = re.sub(r"\s*[-–]\s*", "–", pages.strip())
    out = f"PDF第{pages}页"
    if printed is not None:
        out += f",印刷页{printed}"
    return out


# A trailing LLM-written reference section (its keys get formatted into
# "name pages X" citations like the body's) is stripped before we rebuild
# our own consistent list, so the two never duplicate.
_REF_HEADING_RE = re.compile(
    r"(?m)^\s*(?:#{1,3}\s*)?(?:\*\*|__)?"
    r"(参考文献|参考资料|References|Bibliography)"
    r"(?:\*\*|__)?\s*[:：]?\s*$"
)


def strip_llm_references(text: str) -> str:
    """Drop a trailing LLM-generated reference section (kept if not trailing)."""
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return text
    m = matches[-1]
    tail = text[m.end() :]
    # Only strip when the section looks like a list and nothing but list
    # items follow (guards against cutting a mid-text heading).
    non_empty = [ln for ln in tail.splitlines() if ln.strip()]
    if non_empty and all(
        re.match(r"^\s*(\[?\d+\]?[.、)]|[-*]|\*\*)\s*\S", ln) for ln in non_empty
    ):
        return text[: m.start()].rstrip()
    return text


_PQAC_KEY_RE = re.compile(r"pqac-[0-9a-f]{8}")


def replace_pqac_keys(text: str, session) -> str:
    """Replace raw pqac-xxxxxxxx context keys with their chunk names.

    PaperQA's own replacement only handles ASCII parentheticals; Chinese
    answers routinely cite with full-width parens/semicolons, leaving raw
    keys in the printed answer.  A plain global replace is safe: keys are
    unique tokens, and unknown (hallucinated) keys are left untouched here
    (PaperQA already strips those from the formatted answer).

    The LLM occasionally drops the ``pqac-`` prefix and cites the bare 8-hex
    id (observed in real runs, e.g. "b447b6d2的'多准则'表述"), so known bare
    ids are replaced too, but only as standalone tokens.
    """
    id_to_name = {
        ctx.id: ctx.text.name
        for ctx in session.contexts
        if getattr(ctx, "text", None) is not None
    }

    def repl(m: re.Match) -> str:
        return id_to_name.get(m.group(0), m.group(0))

    out = _PQAC_KEY_RE.sub(repl, text)
    bare = sorted(
        (key[len("pqac-") :] for key in id_to_name if key.startswith("pqac-")),
        key=len,
        reverse=True,
    )
    if bare:
        bare_re = re.compile(
            r"(?<![0-9A-Za-z_-])(?:" + "|".join(map(re.escape, bare)) + r")(?![0-9A-Za-z])"
        )
        out = bare_re.sub(lambda m: id_to_name.get("pqac-" + m.group(0), m.group(0)), out)
    return out


def cited_chunk_names(answer: str, session) -> list[str]:
    """Chunk names ('docname pages X-Y') actually cited in the answer text."""
    known = {
        ctx.text.name for ctx in session.contexts if getattr(ctx, "text", None)
    }
    return sorted((n for n in known if n in answer), key=answer.index)


# --- answer rewriting -----------------------------------------------------------

@dataclass
class RewriteResult:
    text: str
    cited_docnames: list[str]  # unique, in first-appearance order
    verified: int  # cited (docname,pages) pairs matching an actual context chunk
    unresolved: list[str]  # docnames whose metadata stayed missing
    quote_checked: int = 0  # English quoted excerpts checked for verbatim match
    quote_passed: int = 0  # ... of those, verbatim substrings of evidence text
    quote_promoted: int = 0  # 释义 lines promoted to 原文摘录 (verbatim-verified)
    cited_group_count: int = 0  # cited papers after stable-identity dedup
    meta_filled: int = 0  # 元数据缺失 markers filled from the manifest


def _paper_group_key(name: str, manifest: dict[str, MetaEntry]) -> str:
    """Stable paper identity: DOI > normalized title > normalized docname."""
    entry = manifest.get(name)
    if entry is not None and entry.doi:
        return f"doi:{entry.doi.lower().strip()}"
    if entry is not None and entry.title:
        return f"title:{normalize_title(entry.title)}"
    return f"name:{normalize_title(name)}"


# --- verbatim excerpt verification ----------------------------------------------

_WS_RE = re.compile(r"\s+")
# A quoted span lives on a single line: its content must not contain a double
# quote (that would be two spans) or a line break (else a stray quote in a
# truncated title on another line could pair with a real excerpt and mangle
# the text -- observed in real runs).
_QUOTED_RE = re.compile(r"[“\"]([^”\"\n]{10,400})[”\"]")
# An excerpt line whose opening quote never closes (observed when the LLM
# drops the closing mark after an in-quote citation).
_UNCLOSED_EXCERPT_RE = re.compile(
    r"(?m)^(?P<pre>[ \t]*(?:[-*•]|\d+[.、)])?[ \t]*(?:\*\*)?"
    r"(?:原文摘录|摘录)(?:\*\*)?[ \t]*[:：][ \t]*)"
    r"[“\"](?P<content>[^”\"\n]+)$"
)
# Inline-citation markers that may sit inside a quoted span.
_CITE_MARK_RE = re.compile(
    r"pages\s+\d|pqac-[0-9a-f]{8}|PDF第\d+(?:\s*[–-]\s*\d+)?页"
)


def _norm_for_verbatim(s: str) -> str:
    """NFKC-fold, drop soft hyphens, fold typographic quotes/dashes to ASCII,
    collapse whitespace (line breaks in the evidence are allowed)."""
    s = unicodedata.normalize("NFKC", s or "").replace("\u00ad", "")
    s = (
        s.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return _WS_RE.sub(" ", s).strip()


def _cut_trailing_citation(content: str) -> str:
    """Drop a trailing inline citation from a quoted span.

    Citations often sit inside the quotes: '"...conclusion (Doe-2015
    (methods) a pages 208-209)"' (chunk-name style) or '"...prosperity (Name
    (PDF第2–2页))"' (rewritten style).  The excerpt itself is what must
    match the evidence verbatim, so the outermost parenthesis group that
    wraps the last citation marker and reaches the end of the span is
    removed before comparison.
    """
    s = content.rstrip()
    while True:
        last = None
        for m in _CITE_MARK_RE.finditer(s):
            last = m
        if last is None:
            return s
        # Scan backwards for the opening paren of the outermost group that
        # encloses the marker and closes at the end of the span.
        depth = 0
        cut = None
        for i in range(len(s) - 1, -1, -1):
            c = s[i]
            if c in ")）":
                depth += 1
            elif c in "(（":
                depth -= 1
                if depth == 0:
                    cut = i
                    break
        if cut is None or last.start() < cut:
            cut = last.start()
        s = s[:cut].rstrip()
        if not s:
            return s


def verify_quoted_excerpts(
    text: str, evidence_texts: Iterable[str]
) -> tuple[str, int, int]:
    """Force quoted English excerpts to be verbatim substrings of the evidence.

    A quoted span of >= 10 chars that is >= 70% ASCII must, after
    normalization (NFKC, soft-hyphen removal, typographic quote/dash
    folding, whitespace folding) and removal of a trailing inline citation,
    appear verbatim inside some evidence chunk.  Failing spans lose their
    quotes (become paraphrase), and a same-line "原文摘录" label is
    downgraded to "释义".  Excerpt lines whose quote never closes are
    handled line-wise.  Returns (new_text, n_checked, n_passed).
    """
    norms = [_norm_for_verbatim(t) for t in evidence_texts if t]
    checked = passed = 0
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement)

    def consider(content: str, open_start: int, close_start: int | None) -> None:
        nonlocal checked, passed
        if sum(1 for c in content if ord(c) < 128) / max(len(content), 1) < 0.7:
            return  # Chinese/quoted-title spans: not an English excerpt
        checked += 1
        probe = _norm_for_verbatim(_cut_trailing_citation(content))
        if probe and any(probe in t for t in norms):
            passed += 1
            return
        edits.append((open_start, open_start + 1, ""))  # drop opening quote
        if close_start is not None:
            edits.append((close_start, close_start + 1, ""))  # drop closing
        # downgrade a same-line 原文摘录 label (never one on another line)
        line_start = text.rfind("\n", 0, open_start) + 1
        line_end = text.find("\n", open_start)
        if line_end == -1:
            line_end = len(text)
        for lm in re.finditer(r"原文摘录", text[line_start:line_end]):
            s = line_start + lm.start()
            edits.append((s, s + 4, "释义"))

    for m in _QUOTED_RE.finditer(text):
        consider(m.group(1), m.start(), m.end() - 1)
    for m in _UNCLOSED_EXCERPT_RE.finditer(text):
        consider(m.group("content"), m.start("content") - 1, None)

    out = text
    for start, end, rep in sorted(dict.fromkeys(edits), key=lambda e: -e[0]):
        out = out[:start] + rep + out[end:]
    return out, checked, passed


_EXCERPT_HEAD_RE = re.compile(
    r"^(?P<pre>[ \t]*(?:[-*•]|\d+[.、)])?[ \t]*(?:\*\*)?)"
    r"(?P<label>原文摘录|释义|摘录)(?:\*\*)?[ \t]*[:：][ \t]*"
    r"(?P<rest>.*)$"
)
_PARAPHRASE_TAG_RE = re.compile(r"^[（(]\s*释义\s*[)）][ \t]*：?[ \t]*")


def promote_verbatim_excerpts(body: str, chunk_texts) -> tuple[str, int]:
    """Upgrade conservatively-marked 释义 lines that are actually verbatim.

    The answer prompt demands verbatim quotes, which makes the LLM play safe
    and label everything 释义 (observed: 0/23 quotes passed in a real run
    because none were even attempted).  For each 摘录/释义 line, when the
    line's own text (trailing citation labels stripped) is a normalized
    substring of one of the evidence chunks, it IS a verbatim quote: it is
    re-labelled 原文摘录 and quoted, before verify_quoted_excerpts runs, so
    the final statistics count real verbatim quotes.  Non-matching lines are
    left untouched -- this never relaxes the check, it only recovers quotes
    the LLM under-claimed.
    """
    if not chunk_texts:
        return body, 0
    normed = [_norm_for_verbatim(t) for t in chunk_texts]
    promoted = 0
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        m = _EXCERPT_HEAD_RE.match(ln)
        if not m:
            continue
        content = m.group("rest").strip()
        content = _PARAPHRASE_TAG_RE.sub("", content)
        pure = _cut_trailing_citation(content)
        if not content.startswith(pure):
            continue
        tail = content[len(pure):]
        probe = _norm_for_verbatim(pure.strip().strip("\"“”"))
        if len(probe) < 10 or not any(probe in t for t in normed):
            continue
        display = pure.strip()
        if display[:1] in "\"“”":
            if m.group("label") == "原文摘录":
                continue  # already a quoted verbatim excerpt: nothing to do
        else:
            display = f"“{display}”"
        lines[i] = f"{m.group('pre')}原文摘录：{display}{tail}"
        promoted += 1
    if not promoted:
        return body, 0
    return "\n".join(lines), promoted


def _chunk_page_map(session) -> dict[str, str]:
    """Map 'docname pages X-Y' -> chunk text, for printed-page detection."""
    out = {}
    for ctx in session.contexts:
        text = getattr(ctx, "text", None)
        if text is not None and getattr(text, "text", None):
            out.setdefault(text.name, text.text)
    return out


# LLM-written honesty markers (see the FIND answer template): filled from the
# manifest only when the item's citations bind to exactly one known paper.
_META_MARKER_LINE_RE = re.compile(r"年份元数据缺失|题名元数据缺失|作者元数据缺失")
# Numbered entry start ("1. **Author, Year…**") / bullet continuation line.
_NEXT_ITEM_RE = re.compile(r"^\s*\d+[.、)]\s")
_BULLET_RE = re.compile(r"^\s*[-*•]\s")


def _item_start(lines: list[str], i: int) -> int:
    """First line of the entry containing marker line ``i`` (inclusive)."""
    for j in range(i, -1, -1):
        ln = lines[j]
        if _NEXT_ITEM_RE.match(ln):
            return j
        # blank line or flush prose above: the marker belongs to the block
        # that starts right after it, not to the previous paragraph
        if not ln.strip() or (ln[0] not in " \t" and not _BULLET_RE.match(ln)):
            return j + 1
    return 0


def _item_end(lines: list[str], i: int) -> int:
    """End (exclusive) of the entry containing marker line ``i``.

    Stops at the next numbered entry or at flush prose that is neither a
    bullet nor an indented continuation.  The trailing "说明/统计" paragraphs
    name *other* papers' evidence; scanning into them made the last entry's
    binding ambiguous and left it unfilled (observed in a real run where the
    closing note cited two other papers).
    """
    for j in range(i + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            continue
        if _NEXT_ITEM_RE.match(ln):
            return j
        if ln[0] not in " \t" and not _BULLET_RE.match(ln):
            return j
    return len(lines)


def fill_metadata_markers(
    body: str, manifest: dict[str, MetaEntry], docnames: Iterable[str]
) -> tuple[str, int]:
    """Fill "元数据缺失" markers with PDF-verified metadata when unambiguous.

    The LLM writes 年份/题名/作者元数据缺失 when the evidence text did not
    carry the datum.  When the manifest has since verified the datum from the
    paper's own first pages, the marker is filled in -- but only for items
    (entry bounds derived from numbering/indentation) whose chunk citations
    bind to exactly one known paper, so values are never cross-assigned
    between papers.
    """
    known = set(docnames)
    lines = body.split("\n")
    marker_idx = [i for i, ln in enumerate(lines) if _META_MARKER_LINE_RE.search(ln)]
    filled = 0
    for i in marker_idx:
        start, end = _item_start(lines, i), _item_end(lines, i)
        names = {
            d
            for d in known
            if any(f"{d} pages " in ln for ln in lines[start:end])
        }
        if len(names) != 1:
            continue
        entry = manifest.get(next(iter(names)))
        if entry is None or entry.missing:
            continue
        ln = lines[i]
        if entry.year and "年份元数据缺失" in ln:
            ln = ln.replace("年份元数据缺失", entry.year)
            filled += 1
        if entry.title and "题名元数据缺失" in ln:
            # replace the "完整" variant first so no stray prefix survives
            ln = ln.replace("完整题名元数据缺失", f"完整题名《{entry.title}》")
            ln = ln.replace("题名元数据缺失", f"《{entry.title}》")
            filled += 1
        if entry.authors and "作者元数据缺失" in ln:
            ln = ln.replace("作者元数据缺失", ", ".join(entry.authors[:3]))
            filled += 1
        lines[i] = ln
    return "\n".join(lines), filled


def rewrite_answer(
    answer: str,
    session,
    manifest: dict[str, MetaEntry],
) -> RewriteResult:
    """Rewrite inline citations in the answer body and rebuild the ref list.

    Strategy: (1) replace raw pqac keys with chunk names; (2) strip any
    LLM-written trailing reference section; (3) exact-substring replace each
    cited chunk name with a readable label + labelled page range.  Exact
    substring matching (instead of a regex) survives docnames containing
    parentheses, e.g. 'Doe-2015-An introduction to statistics (methods) a'.
    """
    chunk_texts = _chunk_page_map(session)
    known_names = set(chunk_texts) | {
        ctx.text.name for ctx in session.contexts if getattr(ctx, "text", None)
    }

    body = strip_llm_references(replace_pqac_keys(answer, session))
    cited_docnames = {
        ctx.text.doc.docname
        for ctx in session.contexts
        if getattr(ctx, "text", None) is not None
        and getattr(ctx.text, "doc", None) is not None
    }
    body, meta_filled = fill_metadata_markers(body, manifest, cited_docnames)

    def label_for(docname: str) -> str:
        entry = manifest.get(docname)
        if entry and entry.label:
            return entry.label
        if docname in manifest and manifest[docname].missing:
            return f"{docname}(元数据缺失)"
        return docname  # readable filename stem: use as-is

    # First-appearance order is computed before any replacement shifts it.
    cited_names = sorted(
        (n for n in known_names if n in body), key=body.index
    )
    cited: list[str] = []
    verified = 0
    for chunk_name in cited_names:
        docname, _, pages = chunk_name.rpartition(" pages ")
        if not docname:
            continue
        occurrences = body.count(chunk_name)
        verified += occurrences
        if docname not in cited:
            cited.append(docname)
        printed = None
        text = chunk_texts.get(chunk_name)
        if text:
            m = re.match(r"\d+", pages)
            if m:
                printed = detect_printed_page(text, int(m.group(0)))
        body = body.replace(
            chunk_name, f"{label_for(docname)}({format_pages(pages, printed)})"
        )

    # Rebuild the reference list from the citations actually present in body,
    # grouped by stable identity (DOI > title > docname) so one paper under
    # two filename stems is listed (and counted) once.
    unresolved = [n for n in cited if n in manifest and manifest[n].missing]
    groups: dict[str, list[str]] = {}
    for name in cited:
        groups.setdefault(_paper_group_key(name, manifest), []).append(name)
    if cited:
        lines = ["", "", "参考文献(与正文引用一一对应):"]
        for i, (_key, names) in enumerate(groups.items(), start=1):
            entry = next(
                (
                    manifest[n]
                    for n in names
                    if manifest.get(n) is not None and not manifest[n].missing
                ),
                None,
            )
            name = names[0]
            if entry is not None:
                authors = ", ".join(entry.authors) if entry.authors else "作者不详"
                year = entry.year or "年份不详"
                title = entry.title or name
                src = f" | 来源文件:{entry.source_path}" if entry.source_path else ""
                doi = f" | DOI:{entry.doi}" if entry.doi else ""
                lines.append(f"[{i}] {authors}. {year}. 《{title}》.{doi}{src}")
            elif name in manifest:
                src = manifest[name].source_path or "(路径未找到)"
                lines.append(f"[{i}] {name} —— 元数据缺失,来源文件:{src}")
            else:
                lines.append(f"[{i}] {name}")
            if len(names) > 1:
                lines.append(f"    (同一文献的另一文件名:{'、'.join(names[1:])})")
        body = body.rstrip() + "\n" + "\n".join(lines) + "\n"

    # Verbatim-excerpt gate: quoted English must literally appear in evidence.
    # Promotion runs first so quotes the LLM under-claimed as 释义 are
    # recovered (still gated by the same verbatim check below).
    body, quote_promoted = promote_verbatim_excerpts(body, chunk_texts.values())
    body, quote_checked, quote_passed = verify_quoted_excerpts(
        body, chunk_texts.values()
    )

    return RewriteResult(
        text=body,
        cited_docnames=cited,
        verified=verified,
        unresolved=unresolved,
        quote_checked=quote_checked,
        quote_passed=quote_passed,
        quote_promoted=quote_promoted,
        cited_group_count=len(groups),
        meta_filled=meta_filled,
    )
