"""Evidence construction and two-axis paper review (centrality, then stance)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Awaitable, Callable, Literal, Sequence

from citation_fix import _norm_for_verbatim
from hybrid_retrieval import Candidate, QuerySpec


Centrality = Literal["core", "background", "irrelevant", "uncertain"]
Stance = Literal[
    "support", "oppose", "mixed", "neutral", "uncertain", "not_applicable"
]
ReviewStatus = Literal["completed", "pending", "review_failed"]
_CORE_ROLES = {
    "research_question",
    "study_object",
    "main_analysis",
    "main_conclusion",
}


@dataclass(frozen=True)
class FindRequest:
    raw_question: str
    proposition: str | None = None

    def __post_init__(self) -> None:
        if not self.raw_question.strip():
            raise ValueError("A non-empty research question is required")


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    chunk_name: str
    pages: str
    section: str
    own_work_role: str
    text: str


@dataclass
class EvidencePack:
    dockey: str
    docname: str
    spans: list[EvidenceSpan] = field(default_factory=list)
    expanded: bool = False


@dataclass
class PaperVerdict:
    dockey: str
    docname: str
    centrality: Centrality = "uncertain"
    stance: Stance = "not_applicable"
    review_status: ReviewStatus = "pending"
    confidence: float = 0.0
    reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    evidence_role: str = ""
    quote: str = ""
    quote_span_id: str = ""
    quote_status: str = "not_provided"
    exclusion_code: str = ""
    error: str = ""


@dataclass
class FindResult:
    request: FindRequest
    queries: list[QuerySpec]
    candidates: list[Candidate]
    verdicts: dict[str, PaperVerdict]
    coverage: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def validate_accounting(self) -> None:
        dockeys = [candidate.dockey for candidate in self.candidates]
        if len(dockeys) != len(set(dockeys)):
            raise ValueError("Duplicate dockeys in candidate list")
        if set(dockeys) != set(self.verdicts):
            missing = set(dockeys) - set(self.verdicts)
            extra = set(self.verdicts) - set(dockeys)
            raise ValueError(f"Candidate/verdict mismatch: missing={missing}, extra={extra}")

    @property
    def counts(self) -> dict[str, int]:
        self.validate_accounting()
        counts = {
            key: sum(v.centrality == key for v in self.verdicts.values())
            for key in ("core", "background", "irrelevant", "uncertain")
        }
        counts["pending"] = sum(
            v.review_status == "pending" for v in self.verdicts.values()
        )
        counts["review_failed"] = sum(
            v.review_status == "review_failed" for v in self.verdicts.values()
        )
        counts["candidates"] = len(self.candidates)
        return counts

    @property
    def is_complete(self) -> bool:
        return not self.errors and all(
            verdict.review_status == "completed" for verdict in self.verdicts.values()
        )


def pages_from_chunk_name(name: str) -> str:
    head, separator, pages = name.rpartition(" pages ")
    return pages.strip() if separator and head else "未知"


_SECTION_PATTERNS = (
    ("references", r"^\s*(references|bibliography|参考文献)\s*$"),
    ("related_work", r"^\s*(related work|literature review|文献综述)\b"),
    ("abstract", r"^\s*(abstract|摘要)\b"),
    ("introduction", r"^\s*(introduction|引言|前言)\b"),
    ("methods", r"^\s*(methods?|methodology|materials and methods|方法|研究方法)\b"),
    ("results", r"^\s*(results?|结果|研究结果)\b"),
    ("discussion", r"^\s*(discussion|讨论)\b"),
    ("conclusion", r"^\s*(conclusions?|concluding remarks|结论|研究结论)\b"),
)
_OWN_QUESTION = re.compile(
    r"\b(this (?:study|paper)|we (?:investigate|analy[sz]e|examine|develop))\b|"
    r"本文(?:研究|分析|考察)|本研究(?:旨在|研究|分析)",
    re.IGNORECASE,
)
_OWN_CONCLUSION = re.compile(
    r"\b(we (?:find|found|show|demonstrate|conclude)|our (?:results|findings))\b|"
    r"本文(?:发现|表明|认为)|研究结果(?:表明|显示)",
    re.IGNORECASE,
)


def detect_section(text: str, relative_position: float = 0.0) -> str:
    """Best-effort section label; unknown is preferable to a confident invention."""
    for section, pattern in _SECTION_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match and (section != "references" or relative_position >= 0.4):
            return section
    return "unknown"


def infer_evidence_role(section: str, text: str) -> str:
    if section == "references":
        return "secondary_citation"
    if _OWN_CONCLUSION.search(text):
        return "main_conclusion"
    if _OWN_QUESTION.search(text):
        return "research_question"
    if section in {"results", "methods"}:
        return "main_analysis"
    if section in {"discussion", "conclusion"}:
        return "main_conclusion"
    if section in {"introduction", "related_work"}:
        return "author_background"
    return "unknown"


def _window(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Keep both the opening context and the end, where conclusions often occur.
    head = int(limit * 0.7)
    tail = limit - head
    return text[:head] + "\n[...中间文本省略...]\n" + text[-tail:]


class EvidenceBuilder:
    def __init__(
        self,
        texts_by_dockey: dict[str, Sequence[Any]],
        chunk_by_id: dict[str, Any],
        max_spans: int = 6,
        span_chars: int = 2000,
        screen_spans: int = 3,
        screen_chars: int = 1200,
    ):
        self.texts_by_dockey = texts_by_dockey
        self.chunk_by_id = chunk_by_id
        self.max_spans = max_spans
        self.span_chars = span_chars
        self.screen_spans = screen_spans
        self.screen_chars = screen_chars

    def build(
        self, candidate: Candidate, expanded: bool = False, screening: bool = False
    ) -> EvidencePack:
        from hybrid_retrieval import stable_chunk_id

        texts = list(self.texts_by_dockey.get(candidate.dockey, ()))
        docname = str(texts[0].doc.docname) if texts else candidate.dockey
        if not texts:
            return EvidencePack(candidate.dockey, docname, expanded=expanded)
        indices: dict[int, None] = {}

        def add(index: int) -> None:
            if 0 <= index < len(texts):
                indices.setdefault(index)

        add(0)
        id_to_index = {stable_chunk_id(text): i for i, text in enumerate(texts)}
        if screening:
            # 初筛只喂首块（标题/摘要）、最佳命中块与结论块，控制单篇 token。
            for chunk_id in candidate.best_chunk_ids[:2]:
                index = id_to_index.get(chunk_id)
                if index is not None:
                    add(index)
            conclusion = next(
                (
                    i
                    for i, text in enumerate(texts)
                    if detect_section(text.text, i / max(len(texts) - 1, 1)) == "conclusion"
                ),
                len(texts) - 1,
            )
            add(conclusion)
            limit = self.screen_spans
            chars = self.screen_chars
        else:
            for chunk_id in candidate.best_chunk_ids:
                index = id_to_index.get(chunk_id)
                if index is not None:
                    add(index)
                    add(index - 1)
                    add(index + 1)
            for i, text in enumerate(texts):
                section = detect_section(text.text, i / max(len(texts) - 1, 1))
                if section in {"abstract", "methods", "results", "discussion", "conclusion"}:
                    add(i)
            add(len(texts) - 1)
            limit = self.max_spans * 2 if expanded else self.max_spans
            chars = self.span_chars * 2 if expanded else self.span_chars
        spans: list[EvidenceSpan] = []
        for index in sorted(list(indices)[:limit]):
            text = texts[index]
            section = detect_section(text.text, index / max(len(texts) - 1, 1))
            chunk_id = stable_chunk_id(text)
            spans.append(
                EvidenceSpan(
                    span_id=chunk_id[:16],
                    chunk_name=text.name,
                    pages=pages_from_chunk_name(text.name),
                    section=section,
                    own_work_role=infer_evidence_role(section, text.text),
                    text=_window(text.text, chars),
                )
            )
        return EvidencePack(candidate.dockey, docname, spans, expanded)


def render_evidence(pack: EvidencePack) -> str:
    return "\n\n".join(
        f"[{span.span_id}] section={span.section}; role={span.own_work_role}; "
        f"chunk={span.chunk_name}\n{span.text}"
        for span in pack.spans
    )


_CENTRALITY_PROMPT = """你是系统综述的文献筛选员。请判断论文自身研究是否以用户主题为核心。
用户问题：{question}

证据片段（section/role 只是启发式提示，需根据原文判断）：
{evidence}

核心度定义：
- core：论文自身的研究问题、研究对象、主要分析或主要结论围绕主题。支持、反对或中性都可以是 core。
- background：正文有实质背景讨论，但主题仅在引言、文献综述、附带比较或非主要讨论中出现。
- irrelevant：无实质研究关联，或仅参考文献表命中。
- uncertain：证据不足，不能可靠判断。

背景中的支持性句子不能证明论文自身是 core；找不到证据时不能猜成 irrelevant。
只输出 JSON：{{"centrality":"core|background|irrelevant|uncertain",
"confidence":0到1,"reason":"中文具体理由","evidence_ids":["片段ID"],
"evidence_role":"research_question|study_object|main_analysis|main_conclusion|author_background|secondary_citation|methods_only|unknown",
"quote":"可选，必须是某个所引片段中的逐字短句","quote_span_id":"对应片段ID或空串",
"exclusion_code":"无排除原因则空串；仅参考文献命中填reference_only"}}
"""

_STANCE_PROMPT = """论文已被认定为核心研究用户主题。现在只判断论文自身主要结论对命题的立场。
用户问题：{question}
待判断命题：{proposition}
证据：
{evidence}

support=主要结论支持；oppose=主要结论反驳或实质相反；mixed=不同条件下结论明显混合；
neutral=核心研究该主题但无支持/反对结论；uncertain=证据不足。不要把背景引文当作者结论。
只输出 JSON：{{"stance":"support|oppose|mixed|neutral|uncertain",
"confidence":0到1,"reason":"中文具体理由","evidence_ids":["片段ID"],
"quote":"可选，必须是某个所引片段中的逐字短句","quote_span_id":"对应片段ID或空串"}}
"""

Judge = Callable[[str], Awaitable[dict[str, Any]]]


async def default_llm_judge(prompt: str) -> dict[str, Any]:
    import litellm

    response = await litellm.acompletion(
        model=f"openai/{os.environ['KIMI_MODEL']}",
        api_base=os.environ["KIMI_API_BASE"],
        api_key=os.environ["KIMI_API_KEY"],
        # coding 端点仅允许 temperature=1，故不传该参数（与其他调用点一致）
        timeout=120,
        num_retries=2,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge returned no JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object")
    return data


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _valid_evidence_ids(data: dict[str, Any], pack: EvidencePack) -> list[str]:
    allowed = {span.span_id for span in pack.spans}
    ids = data.get("evidence_ids", [])
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if str(item) in allowed]


def _own_work_evidence_ids(evidence_ids: Sequence[str], pack: EvidencePack) -> list[str]:
    selected = set(evidence_ids)
    return [
        span.span_id
        for span in pack.spans
        if span.span_id in selected
        and span.section not in {"references", "related_work"}
        and span.own_work_role not in {"secondary_citation", "author_background"}
    ]


def _verify_quote(
    verdict: PaperVerdict, data: dict[str, Any], pack: EvidencePack
) -> None:
    quote = str(data.get("quote", "") or "").strip()
    span_id = str(data.get("quote_span_id", "") or "").strip()
    if not quote:
        return
    span = next((item for item in pack.spans if item.span_id == span_id), None)
    if span and _norm_for_verbatim(quote) in _norm_for_verbatim(span.text):
        verdict.quote = quote
        verdict.quote_span_id = span_id
        verdict.quote_status = "verified"
    else:
        verdict.quote = ""
        verdict.quote_span_id = ""
        verdict.quote_status = "invalid"


class PaperReviewer:
    def __init__(self, judge: Judge | None = None, second_pass_threshold: float = 0.6):
        self.judge = judge or default_llm_judge
        self.second_pass_threshold = second_pass_threshold

    async def _call(self, prompt: str) -> dict[str, Any]:
        try:
            return await self.judge(prompt)
        except (ValueError, TypeError, json.JSONDecodeError):
            # One bounded retry for malformed structured output, not an API retry storm.
            return await self.judge(prompt)

    async def review(
        self, request: FindRequest, pack: EvidencePack, expanded_pack: EvidencePack | None = None
    ) -> PaperVerdict:
        if not pack.spans:
            return PaperVerdict(
                pack.dockey,
                pack.docname,
                review_status="completed",
                reason="没有可审阅文本，无法判断核心度",
                exclusion_code="no_text",
            )
        verdict = await self._centrality(request, pack)
        needs_second = verdict.centrality == "uncertain" or (
            verdict.confidence < self.second_pass_threshold and expanded_pack is not None
        )
        if needs_second and expanded_pack is not None and expanded_pack.spans:
            verdict = await self._centrality(request, expanded_pack)
        if verdict.centrality == "core" and request.proposition:
            stance_pack = expanded_pack if expanded_pack is not None else pack
            stance = await self._call(
                _STANCE_PROMPT.format(
                    question=request.raw_question,
                    proposition=request.proposition,
                    evidence=render_evidence(stance_pack),
                )
            )
            value = str(stance.get("stance", "uncertain")).lower()
            if value not in {"support", "oppose", "mixed", "neutral", "uncertain"}:
                raise ValueError(f"Invalid stance: {value}")
            verdict.stance = value  # type: ignore[assignment]
            stance_ids = _valid_evidence_ids(stance, stance_pack)
            own_work_stance_ids = _own_work_evidence_ids(stance_ids, stance_pack)
            if not own_work_stance_ids:
                verdict.stance = "uncertain"
                verdict.reason += "；立场缺少论文自身研究的有效证据ID"
            else:
                verdict.evidence_ids = list(dict.fromkeys(verdict.evidence_ids + stance_ids))
            verdict.confidence = min(verdict.confidence, _confidence(stance.get("confidence")))
            verdict.reason += "；立场：" + str(stance.get("reason", "") or "")
            if stance.get("quote"):
                _verify_quote(verdict, stance, stance_pack)
        return verdict

    async def _centrality(self, request: FindRequest, pack: EvidencePack) -> PaperVerdict:
        data = await self._call(
            _CENTRALITY_PROMPT.format(
                question=request.raw_question, evidence=render_evidence(pack)
            )
        )
        centrality = str(data.get("centrality", "uncertain")).lower()
        if centrality not in {"core", "background", "irrelevant", "uncertain"}:
            raise ValueError(f"Invalid centrality: {centrality}")
        verdict = PaperVerdict(
            dockey=pack.dockey,
            docname=pack.docname,
            centrality=centrality,  # type: ignore[arg-type]
            review_status="completed",
            confidence=_confidence(data.get("confidence")),
            reason=str(data.get("reason", "") or "").strip(),
            evidence_ids=_valid_evidence_ids(data, pack),
            evidence_role=str(data.get("evidence_role", "unknown") or "unknown"),
            exclusion_code=str(data.get("exclusion_code", "") or "").strip(),
        )
        own_work_ids = _own_work_evidence_ids(verdict.evidence_ids, pack)
        if centrality == "core" and (
            not own_work_ids or verdict.evidence_role not in _CORE_ROLES
        ):
            verdict.centrality = "uncertain"
            verdict.stance = "not_applicable"
            verdict.reason += "；核心判定缺少论文自身研究的有效证据，已降为待复核"
        if centrality != "core":
            verdict.stance = "not_applicable"
        _verify_quote(verdict, data, pack)
        return verdict

    async def review_many(
        self,
        request: FindRequest,
        candidates: Sequence[Candidate],
        packs: dict[str, EvidencePack],
        expanded_packs: dict[str, EvidencePack] | None = None,
        concurrency: int = 4,
        max_review: int | None = None,
        on_verdict: Callable[[PaperVerdict], None] | None = None,
        on_progress: Callable[[int, int, PaperVerdict], None] | None = None,
    ) -> dict[str, PaperVerdict]:
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if max_review is not None and max_review < 0:
            raise ValueError("max_review must be non-negative")
        semaphore = asyncio.Semaphore(concurrency)
        selected = set(range(len(candidates) if max_review is None else max_review))
        verdicts: dict[str, PaperVerdict] = {}
        done = 0

        def store(verdict: PaperVerdict) -> None:
            nonlocal done
            verdicts[verdict.dockey] = verdict
            done += 1
            if on_verdict is not None:
                on_verdict(verdict)
            if on_progress is not None:
                on_progress(done, len(candidates), verdict)

        async def run_one(index: int, candidate: Candidate) -> None:
            pack = packs[candidate.dockey]
            if index not in selected:
                store(
                    PaperVerdict(
                        candidate.dockey, pack.docname, reason="达到用户设置的复核预算"
                    )
                )
                return
            try:
                async with semaphore:
                    verdict = await self.review(
                        request,
                        pack,
                        (expanded_packs or {}).get(candidate.dockey),
                    )
                store(verdict)
            except Exception as exc:
                store(
                    PaperVerdict(
                        candidate.dockey,
                        pack.docname,
                        review_status="review_failed",
                        reason="模型未能完成复核",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        await asyncio.gather(*(run_one(i, c) for i, c in enumerate(candidates)))
        return verdicts
