r"""Per-turn paper ledger: what the previous answer actually found.

Solves the "你上面总共找到了几篇?" class of follow-ups.  After each answered
turn, ``PaperLedger.from_session`` snapshots the papers behind the session's
evidence, and ``restrict_to_cited`` marks which of them the answer text
actually cited.  A later COUNT question is answered by re-counting *this
saved set* -- never by re-running retrieval.

Counting basis: papers actually cited in the answer body (the "found" set).
Papers that entered evidence but were not cited are reported separately, so
the two numbers can never be conflated.

Deduplication rule: DOI when available, otherwise a normalized key built from
docname (our docnames are deterministic filename stems, which already
identify a unique file).

Support tiers (直接符合 / 间接支持 / 仅相关、不计入) are parsed from the
previous answer's raw text -- one structured parse (``parse_support_tiers``)
feeds both the ledger entries and the count report, so the answer body, the
ledger and the count can never disagree.  A tier is only assigned when the
whole numbered item cites keys of exactly ONE paper -- secondary-citation
items (whose keys belong to the quoting paper, not the discussed one) stay
ungraded rather than being mis-assigned.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

TIER_DIRECT = "直接符合"
TIER_INDIRECT = "间接支持"
TIER_BACKGROUND = "仅相关、不计入"
TIER_UNGRADED = "未分级"

# Map tier phrases used in prompts to ledger tiers.
_TIER_ALIASES = {
    "直接表述": TIER_DIRECT,
    "直接符合": TIER_DIRECT,
    "直接支持": TIER_DIRECT,
    "间接支持": TIER_INDIRECT,
    "间接挑战": TIER_INDIRECT,
    "仅背景相关": TIER_BACKGROUND,
    "仅相关": TIER_BACKGROUND,
}

# Numbered FIND items start as "1. **Author…**" or "**1. Author…**" (observed
# in real runs); older runs used "7. **…".  The optional bold markers are what
# the first split regex missed, collapsing the whole answer into one block and
# grading only its first item.
_ITEM_SPLIT_RE = re.compile(r"(?m)(?=^\s*\*{0,2}\s*\d+\s*[.、])")
_SUPPORT_LINE_RE = re.compile(r"支持类型[:：]\s*([一-鿿、]+)")


def parse_support_tiers(raw_answer: str) -> list[tuple[str, str]]:
    """Structured parse of a FIND answer: (item first line, tier) pairs.

    This is the single source of truth for support tiers: the ledger's tier
    assignment and every consistency check read this parse instead of
    re-guessing from the prose.  Each numbered item contributes at most one
    tier, taken from its ``支持类型:`` line; 转引 (secondary-citation) items
    are skipped because their tier describes a paper outside the corpus.
    """
    out: list[tuple[str, str]] = []
    for block in _ITEM_SPLIT_RE.split(raw_answer):
        m = _SUPPORT_LINE_RE.search(block)
        if not m:
            continue
        tier = _TIER_ALIASES.get(m.group(1).strip())
        if not tier:
            continue
        stripped = block.strip()
        first_line = stripped.splitlines()[0].lower() if stripped else ""
        if "转引" in first_line:
            continue
        out.append((first_line, tier))
    return out


def normalize_title(text: str) -> str:
    """Lowercase, NFKC-fold, and strip non-alphanumeric/CJK characters."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^\w一-鿿]+", "", text)


def _entry_head(docname: str) -> str:
    """Leading author/name token of a docname, e.g. 'smith' or '李华'."""
    return re.split(r"[-_]|(?<=\D)(?=\d{4})", docname, maxsplit=1)[0].lower()


@dataclass
class LedgerEntry:
    dockey: str
    docname: str
    citation: str
    n_contexts: int = 0
    max_score: int = 0
    tier: str = TIER_UNGRADED
    doi: str | None = None
    label: str | None = None  # readable "Author Year" label when resolved
    title: str | None = None  # full title when known (manifest / metadata)
    cited: bool = False  # True when the answer body actually cited this paper

    @property
    def dedup_key(self) -> str:
        """Stable identity: DOI > normalized title > normalized docname.

        The cascade keeps one physical paper at one entry even when it shows
        up under different filename stems (e.g. re-downloaded copies of the
        same paper under two docnames).
        """
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        if self.title:
            return f"title:{normalize_title(self.title)}"
        return f"name:{normalize_title(self.docname)}"


@dataclass
class PaperLedger:
    question: str
    intent: str
    entries: list[LedgerEntry] = field(default_factory=list)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_session(cls, session, question: str, intent: str) -> "PaperLedger":
        """Snapshot the distinct papers behind a session's evidence contexts."""
        by_key: dict[str, LedgerEntry] = {}
        for ctx in session.contexts:
            doc = getattr(getattr(ctx, "text", None), "doc", None)
            if doc is None:
                continue
            key = getattr(doc, "dockey", None) or getattr(doc, "docname", "")
            entry = by_key.get(key)
            if entry is None:
                entry = LedgerEntry(
                    dockey=key,
                    docname=getattr(doc, "docname", key),
                    citation=getattr(doc, "citation", "") or "",
                )
                by_key[key] = entry
            entry.n_contexts += 1
            entry.max_score = max(entry.max_score, getattr(ctx, "score", 0) or 0)
        ledger = cls(question=question, intent=intent)
        ledger.entries = sorted(by_key.values(), key=lambda e: e.docname.lower())
        return ledger

    # -- cited/tier handling ---------------------------------------------------
    def restrict_to_cited(
        self, cited_docnames: list[str], raw_answer: str, session
    ) -> None:
        """Mark cited papers and assign support tiers from the raw answer.

        ``cited_docnames`` comes from the citation rewrite pass (papers whose
        chunk names appear in the printed answer).  Tier parsing is
        deliberately conservative: a numbered item's tier is applied only
        when every citation key inside that item belongs to one single paper.
        """
        cited_set = set(cited_docnames)
        for entry in self.entries:
            entry.cited = entry.docname in cited_set

        id_to_dockey = {}
        for ctx in session.contexts:
            doc = getattr(getattr(ctx, "text", None), "doc", None)
            if doc is not None:
                id_to_dockey[ctx.id] = getattr(doc, "dockey", None)
        if not id_to_dockey:
            return

        # One structured parse feeds both the ledger here and any consistency
        # check elsewhere (see parse_support_tiers): tiers are never re-guessed
        # from the prose by a second, divergent parser.
        blocks = [
            block
            for block in _ITEM_SPLIT_RE.split(raw_answer)
            if _SUPPORT_LINE_RE.search(block) and block.strip()
        ]
        by_dockey = {e.dockey: e for e in self.entries}
        cited_entries = [e for e in self.entries if e.cited]
        for block in blocks:
            m = _SUPPORT_LINE_RE.search(block)
            tier = _TIER_ALIASES.get(m.group(1).strip())
            if not tier:
                continue
            first_line = block.strip().splitlines()[0].lower()
            # Secondary-citation guard: an item that is itself a 转引 entry
            # discusses a paper outside the corpus; its tier must not bind to
            # the quoting paper.
            if "转引" in first_line:
                continue
            # Primary signal: the paper named in the item's first line is the
            # one the tier describes.  This stays correct even when the item's
            # citation keys are missing, hallucinated or belong to quoting
            # papers (observed in real runs).
            head_hits = [
                e
                for e in cited_entries
                if len(_entry_head(e.docname)) >= 2
                and _entry_head(e.docname) in first_line
            ]
            if len(head_hits) == 1:
                head_hits[0].tier = tier
                continue
            if len(head_hits) > 1:
                # Ambiguous item naming several cited papers.
                continue
            # Fallback: every citation key in the item belongs to one paper,
            # and that paper's head appears in the first line.
            dockeys = {
                id_to_dockey[key]
                for key in re.findall(r"pqac-[0-9a-f]{8}", block)
                if key in id_to_dockey
            }
            if len(dockeys) != 1:
                continue
            entry = by_dockey.get(next(iter(dockeys)))
            if entry is None:
                continue
            head = _entry_head(entry.docname)
            if len(head) >= 2 and head not in first_line:
                continue
            entry.tier = tier

    # -- counting -------------------------------------------------------------
    def deduped(self, entries: Iterable[LedgerEntry]) -> list[LedgerEntry]:
        seen: dict[str, LedgerEntry] = {}
        for entry in entries:
            seen.setdefault(entry.dedup_key, entry)
        return sorted(seen.values(), key=lambda e: (e.label or e.docname).lower())

    def count_report(self) -> str:
        """Deterministic, item-recountable answer to '你上面找到了几篇?'."""
        if not self.entries:
            return (
                "上一轮没有保存可计数的文献集合(上一轮未产生带引用的回答)。"
                "无法给出数量,请先提出检索类问题。"
            )

        cited = self.deduped([e for e in self.entries if e.cited])
        uncited = self.deduped([e for e in self.entries if not e.cited])
        pool = cited or self.deduped(self.entries)
        basis = "正文实际引用" if cited else "进入证据"

        direct = [e for e in pool if e.tier == TIER_DIRECT]
        indirect = [e for e in pool if e.tier == TIER_INDIRECT]
        background = [e for e in pool if e.tier == TIER_BACKGROUND]
        ungraded = [e for e in pool if e.tier == TIER_UNGRADED]

        lines = [
            f"上一轮问题:{self.question}",
            "",
            f"确定数量:{len(pool)} 篇(口径:{basis}的独立文献,去重后逐篇编号,可逐项复算)",
            "去重规则:按 DOI 去重;无 DOI 时按规范化题名/文档标识去重;每篇只计一次。",
            "",
            "逐篇清单:",
        ]
        for i, e in enumerate(pool, start=1):
            name = e.label or e.docname
            lines.append(f"{i}. {name} [{e.tier}]")

        lines.append("")
        lines.append(
            f"分级统计:直接符合 {len(direct)} 篇;间接支持 {len(indirect)} 篇;"
            f"仅相关、不计入 {len(background)} 篇"
            + (f";未分级 {len(ungraded)} 篇" if ungraded else "")
            + "。"
        )
        if direct or indirect or background:
            lines.append(
                f"按「直接符合+间接支持」口径计 {len(direct) + len(indirect)} 篇;"
                f"按「全部引用」口径计 {len(pool)} 篇。"
            )
        if cited and uncited:
            lines.append("")
            lines.append(
                f"另注:另有 {len(uncited)} 篇进入证据但未在正文引用,不计入上面数量:"
                + "、".join(e.label or e.docname for e in uncited[:10])
                + ("……" if len(uncited) > 10 else "")
            )
        return "\n".join(lines)

    # -- membership follow-ups -------------------------------------------------
    def membership_answer(self, title: str | None) -> str:
        """Answer "《题名》上面有这篇吗?" against the previous cited set.

        Checked ONLY against the papers the previous answer actually cited --
        never against retrieval candidates, and never by scanning other
        papers' reference lists.  A "no" is always qualified: not being
        listed does not mean the paper is irrelevant or absent from the
        corpus.
        """
        if not self.entries:
            return (
                "上一轮没有保存可查询的文献集合(上一轮未产生带引用的回答)。"
                "请先提出检索类问题。"
            )
        cited = self.deduped([e for e in self.entries if e.cited])
        if not cited:
            return (
                "上一轮答案没有实际引用任何文献,所以上面没有列出文献。"
                "这不代表该文献不相关或语料库中没有。"
            )
        listed = "\n".join(f"- {_display(e)}" for e in cited)
        if not title:
            return (
                "没有提取到具体题名。上一轮答案实际列出的引用文献是:\n"
                f"{listed}\n请指明其中某一篇的题名或作者再问。"
            )
        matches = [e for e in cited if title_matches_entry(title, e)]
        if matches:
            e = matches[0]
            lines = [
                "有。上一轮答案实际引用的文献中有这一篇:",
                f"- {_display(e)}",
            ]
            if e.title and normalize_title(e.title) != normalize_title(e.docname):
                lines.append(f"  完整题名:《{e.title}》")
            if e.doi:
                lines.append(f"  DOI:{e.doi}")
            lines.append("说明:此处仅对照上一轮答案实际列出的文献,未做新的检索。")
            return "\n".join(lines)
        return (
            "没有。上一轮答案实际列出的引用文献是:\n"
            f"{listed}\n"
            "说明:上一轮未列出,不代表该文献不相关或语料库中没有。"
            "此处仅对照上一轮答案实际列出的文献,未做新的检索。"
        )


def _display(entry: LedgerEntry) -> str:
    if entry.label:
        out = entry.label
        if entry.title:
            out += f"《{entry.title}》"
        return out
    return entry.title or entry.docname


def title_matches_entry(title: str, entry: LedgerEntry) -> bool:
    """Does ``title`` (from a membership question) name this entry's paper?

    Signals, any one suffices (all on NFKC-folded, alphanumeric-only text):
    1. one normalized string contains the other (>= 8 chars);
    2. a >= 15-char tail of the stored name (filename stems are truncated,
       e.g. 'an introduction to statistical lea') prefixes the asked title;
    3. >= 60% of the title's >= 4-letter words appear in the entry's fields
       (prefix-tolerant, so 'lea' matches 'learning' only in the direction
       entry-word -> title-word prefix of >= 3 chars).
    """
    target = normalize_title(title)
    if len(target) < 4:
        return False
    sources = [entry.docname, entry.title or "", entry.citation or ""]
    for src in sources:
        s = normalize_title(src)
        if not s:
            continue
        if len(target) >= 8 and target in s:
            return True
        if len(s) >= 8 and s in target:
            return True
        if len(s) >= 15 and any(
            target.startswith(s[i:]) for i in range(0, len(s) - 14)
        ):
            return True
    words = [w for w in re.findall(r"[a-z0-9]{4,}", (title or "").lower())]
    if words:
        pool: set[str] = set()
        for src in sources + [entry.label or ""]:
            pool |= set(re.findall(r"[a-z0-9]{3,}", (src or "").lower()))
        hits = sum(
            1
            for w in words
            if w in pool or any(len(p) >= 3 and w.startswith(p) for p in pool)
        )
        if hits / len(words) >= 0.6:
            return True
    return False


def entries_from_pairs(pairs: Iterable[tuple[str, str]]) -> list[LedgerEntry]:
    """Test helper: build entries from (dockey, docname) pairs."""
    return [LedgerEntry(dockey=k, docname=n, citation="") for k, n in pairs]
