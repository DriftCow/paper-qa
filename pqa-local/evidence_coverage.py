r"""Evidence coverage check + supplementation for FIND answers.

Root cause this module addresses (observed in a real run): 84 candidates,
27 papers entered evidence, only 6 were cited -- papers whose evidence
directly supports the question (one with 2 chunks, another with 8 chunks)
were silently dropped by the answer generator, and the audit
could only record "原因未说明".

Mechanism (runs between the raw answer and the citation rewrite):

1. ``check_evidence_coverage`` diffs evidence papers (session contexts)
   against cited papers.
2. For each uncited evidence paper, one small LLM call judges its evidence
   against the question: direct / indirect / none, plus a reason grounded in
   the evidence text and a quote candidate.
3. ``direct`` papers get a supplemented FIND item appended to the answer.
   Everything structural in the item is deterministic (manifest authors/
   year/title, a chunk-name citation the normal rewrite pass turns into
   ``label(PDF第X–Y页)``, support tier ``直接表述``); only the 表述 sentence
   is LLM-written, and the quote is printed only when it passes the
   verbatim-substring check against the cited chunk.
4. ``indirect``/``none`` papers keep the judge's evidence-grounded reason,
   which replaces "未说明" in the retrieval audit.

Never done here: inventing reasons, hard-coding any paper, relaxing the
verbatim check, or adding papers whose evidence does not support the
question (the judge's "none"/"indirect" keeps them out).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from citation_fix import MetaEntry, _norm_for_verbatim

# Per-chunk text cap fed to the judge (keeps each call small).
_CHUNK_CAP = 1200
_MAX_CHUNKS_PER_PAPER = 8
_MAX_CONCURRENT = 8

_JUDGE_PROMPT = (
    "研究问题:{question}\n\n"
    "某文献的证据片段(来自其PDF的真实摘录):\n{evidence}\n\n"
    "判断该文献的证据对研究问题的支持程度,只输出一行 JSON(不要多余文字):\n"
    '{{"support":"direct或indirect或none",'
    '"statement":"面向用户的相关表述,1-2句,概括该文献证据如何论述问题所问的主张",'
    '"reason":"一句话判定理由,必须引用证据中的具体内容(内部审计用)",'
    '"quote":"支撑判定的英文原文短句,不超过40词,必须是上面证据文本的逐字子串;'
    '找不到逐字短句就给空串"}}\n'
    "判定标准:\n"
    "- direct: 证据文本明确论述问题所问的主张本身;\n"
    "- indirect: 证据与主题相关但未直接论述该主张;\n"
    "- none: 证据与主张无实质关联(如仅出现在参考文献表、纯方法学描述)。"
)


@dataclass
class PaperEvidence:
    dockey: str
    chunks: list[tuple[str, str]] = field(default_factory=list)  # (name, text)

    def capped_texts(self) -> list[str]:
        return [t[:_CHUNK_CAP] for _, t in self.chunks[:_MAX_CHUNKS_PER_PAPER]]


@dataclass
class CoverageVerdict:
    docname: str
    dockey: str
    support: str = "none"  # "direct" | "indirect" | "none"
    reason: str = ""
    statement: str = ""
    quote: str = ""
    quote_chunk: str | None = None  # chunk name the quote verbatim-matches
    n_chunks: int = 0  # evidence chunks this paper had in the session
    added: bool = False  # True when a supplemented item was inserted

    def audit_reason(self) -> str:
        base = f"进入证据({self.n_chunks}个片段)但未被答案正文引用"
        if self.reason:
            return f"{base};覆盖检查判定为{self.support}:{self.reason}"
        return f"{base};覆盖检查判定为{self.support}(判定理由未能取得)"


@dataclass
class CoverageReport:
    verdicts: dict[str, CoverageVerdict] = field(default_factory=dict)  # by dockey
    supplement_text: str = ""  # the inserted item block, for tier parsing

    @property
    def added_docnames(self) -> list[str]:
        return [v.docname for v in self.verdicts.values() if v.added]

    def summary(self) -> str:
        if not self.verdicts:
            return "覆盖检查:答案引用了全部进入证据的文献,无遗漏"
        n = {"direct": 0, "indirect": 0, "none": 0}
        for v in self.verdicts.values():
            n[v.support] = n.get(v.support, 0) + 1
        return (
            f"覆盖检查:对 {len(self.verdicts)} 篇进入证据但未引用的文献逐篇判定:"
            f"直接支持 {n['direct']} 篇(补充 {len(self.added_docnames)} 篇入答案)、"
            f"间接 {n['indirect']} 篇、不相关 {n['none']} 篇;"
            "逐篇判定理由见检索审计"
        )


def _collect_evidence(session) -> dict[str, PaperEvidence]:
    """docname -> PaperEvidence (chunk names paired with their texts)."""
    out: dict[str, PaperEvidence] = {}
    for ctx in getattr(session, "contexts", []) or []:
        text = getattr(ctx, "text", None)
        doc = getattr(text, "doc", None)
        if doc is None or not getattr(text, "text", None):
            continue
        dockey = getattr(doc, "dockey", None) or doc.docname
        ev = out.setdefault(doc.docname, PaperEvidence(dockey=dockey))
        ev.chunks.append((text.name, text.text))
    return out


async def _default_judge(question: str, docname: str, evidence: list[str]) -> dict[str, Any]:
    """One small DeepSeek call per uncited evidence paper."""
    import litellm

    body = "\n\n".join(
        f"[片段{i+1}]\n{t}" for i, t in enumerate(evidence)
    )
    response = await litellm.acompletion(
        model=f"openai/{os.environ['DEEPSEEK_MODEL']}",
        api_base=os.environ["DEEPSEEK_API_BASE"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(question=question, evidence=body),
            }
        ],
    )
    content = response.choices[0].message.content or ""
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def _match_quote_chunk(quote: str, evidence: PaperEvidence) -> str | None:
    """Chunk whose normalized text contains the normalized quote, if any."""
    q = _norm_for_verbatim(quote)
    if not q:
        return None
    for name, text in evidence.chunks:
        if q in _norm_for_verbatim(text):
            return name
    return None


async def check_evidence_coverage(
    question: str,
    session,
    cited_docnames: set[str],
    judge: Callable[[str, str, list[str]], Awaitable[dict[str, Any]]] | None = None,
) -> tuple[CoverageReport, dict[str, PaperEvidence]]:
    """Judge every evidence paper the answer did not cite.

    Returns (report, evidence map); the map is reused when building the
    supplement items so texts are read from the session once.
    """
    judge = judge or _default_judge
    evidence_map = _collect_evidence(session)
    report = CoverageReport()
    uncited = [d for d in evidence_map if d not in cited_docnames]
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def run_one_paper(docname: str) -> None:
        evidence = evidence_map[docname]
        verdict = CoverageVerdict(
            docname=docname, dockey=evidence.dockey, n_chunks=len(evidence.chunks)
        )
        try:
            async with sem:
                data = await judge(question, docname, evidence.capped_texts())
        except Exception as exc:  # noqa: BLE001 - offline/broken API: honest skip
            verdict.support = "none"
            verdict.reason = f"覆盖检查调用失败({type(exc).__name__}),未能判定"
            report.verdicts[evidence.dockey] = verdict
            return
        support = str(data.get("support", "none")).strip().lower()
        verdict.support = support if support in {"direct", "indirect", "none"} else "none"
        verdict.reason = str(data.get("reason", "") or "").strip()
        verdict.statement = str(data.get("statement", "") or "").strip()
        quote = str(data.get("quote", "") or "").strip().strip('"“”')
        if quote:
            chunk = _match_quote_chunk(quote, evidence)
            if chunk is not None:  # quote kept only when verbatim-verifiable
                verdict.quote = quote
                verdict.quote_chunk = chunk
        report.verdicts[evidence.dockey] = verdict

    if uncited:
        await asyncio.gather(*(run_one_paper(d) for d in uncited))
    return report, evidence_map


def _authors_line(entry: MetaEntry | None, docname: str) -> str:
    """Deterministic '作者（年份）。《题名》' head from the manifest (no guessing)."""
    if entry is None or entry.missing:
        return f"{docname}(元数据缺失)"
    authors = ", ".join(entry.authors[:3]) if entry.authors else "作者不详"
    year = entry.year or "年份不详"
    title = entry.title or docname
    return f"{authors}（{year}）。《{title}》"


def _next_index(answer_text: str) -> int:
    nums = [
        int(n)
        for n in re.findall(r"(?m)^\s*\*{0,2}\s*(\d+)\s*[.、]", answer_text)
    ]
    return (max(nums) + 1) if nums else 1


_CLOSING_RE = re.compile(r"(?m)^[^\n]*以上为本次检索命中的文献")
_REFERENCES_RE = re.compile(r"(?m)^[ \t]*[#＃*]*[ \t]*参考文献")


def build_supplement_item(
    verdict: CoverageVerdict, entry: MetaEntry | None, evidence: PaperEvidence, index: int
) -> str:
    """One FIND-style item; only the 表述 sentence is LLM-written.

    The citation is the chunk-name form, so the existing rewrite pass turns
    it into ``label(PDF第X–Y页)`` and lists the paper in the rebuilt
    references.  The quote is printed only when it passed the verbatim check.
    """
    docname = verdict.docname
    cite_chunk = verdict.quote_chunk or (evidence.chunks[0][0] if evidence.chunks else "")
    lines = [
        f"{index}. **{_authors_line(entry, docname)}**",
        f"- 相关表述："
        f"{verdict.statement or verdict.reason or '该文献证据直接论述问题所问的主张。'}",
    ]
    if verdict.quote and cite_chunk:
        lines.append(f'- 原文摘录："{verdict.quote}" ({cite_chunk})')
    elif cite_chunk:
        lines.append(f"- 释义：见其证据片段 ({cite_chunk})")
    lines.append("- 支持类型：直接表述")
    lines.append(f"- 来源:覆盖检查补充(该文献证据直接支持问题,但初答未列出)")
    return "\n".join(lines)


def insert_items(answer_text: str, items: list[str]) -> str:
    """Insert supplement items before the closing note (or references)."""
    if not items:
        return answer_text
    block = "\n\n".join(items)
    for pattern in (_CLOSING_RE, _REFERENCES_RE):
        m = pattern.search(answer_text)
        if m:
            head = answer_text[: m.start()].rstrip() + "\n\n"
            return head + block + "\n\n" + answer_text[m.start():]
    return answer_text.rstrip() + "\n\n" + block + "\n"


async def check_and_supplement(
    question: str,
    answer_text: str,
    session,
    cited_docnames: set[str],
    manifest: dict[str, MetaEntry],
    judge: Callable[[str, str, list[str]], Awaitable[dict[str, Any]]] | None = None,
) -> tuple[str, CoverageReport]:
    """Full pass: judge uncited evidence papers, append items for direct ones.

    Returns the (possibly extended) answer text and the report for the audit.
    """
    report, evidence_map = await check_evidence_coverage(
        question, session, cited_docnames, judge=judge
    )
    items: list[str] = []
    index = _next_index(answer_text)
    for verdict in sorted(report.verdicts.values(), key=lambda v: v.docname.lower()):
        if verdict.support != "direct":
            continue
        entry = manifest.get(verdict.docname)
        evidence = evidence_map.get(verdict.docname)
        if evidence is None:
            continue
        items.append(build_supplement_item(verdict, entry, evidence, index))
        verdict.added = True
        index += 1
    report.supplement_text = "\n\n".join(items)
    return insert_items(answer_text, items), report
