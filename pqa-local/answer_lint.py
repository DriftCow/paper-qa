r"""Deterministic consistency checks on generated answers.

Prompt-level evidence discipline is the primary defence; this module is the
safety net.  Each rule is small, deterministic, and unit-testable:

1. Recession-count consistency -- an answer must not call the same statistic
   "五次衰退中的四次" in one place and "四次中的四次" (or "四分之四") in
   another.  When a stated total exists elsewhere in the answer, conflicting
   "N次中的N次" / "N分之N" phrasings are rewritten to that total and the fix
   is reported.

2. Negative-conclusion guard -- for CRITIQUE intent or truncated runs,
   absolute phrasings like "文献中不存在"/"学界没有" must be qualified as a
   this-round result ("本轮检索未命中" etc.).  Unqualified occurrences are
   not rewritten (too risky); instead an explicit limitation note is
   appended so the answer cannot read as a non-existence claim.

3. Page-number consistency -- an answer whose citations already print PDF
   page numbers ("PDF第2-3页") must not also claim that page numbers were
   not given ("页码未给出"/"证据片段自带PDF顺序页码缺失"/"证据片段未标注
   PDF顺序页码（元数据缺失）"); the contradictory claims (or whole bullet
   lines) are removed and the fix is reported.  Honest claims about
   *printed* page numbers ("印刷页码缺失") are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_CN_INV = {v: k for k, v in _CN.items()}


def _num(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _CN.get(token)


def _cn(n: int) -> str:
    return _CN_INV.get(n, str(n))


@dataclass
class LintResult:
    text: str
    fixes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def fix_count_consistency(text: str) -> LintResult:
    """Normalize 'N次中的N次' / 'N分之N' against the stated recession total."""
    result = LintResult(text=text)

    totals = [
        n
        for tok in re.findall(r"([一二三四五六七八九十\d]+)\s*次(?:经济)?(?:大)?衰退", text)
        if (n := _num(tok)) is not None
    ]
    if not totals:
        return result
    total = max(set(totals), key=totals.count)

    def fix_self_over_self(m: re.Match) -> str:
        a, b = _num(m.group(1)), _num(m.group(2))
        if a is None or b is None or a != b or a == total:
            return m.group(0)
        result.fixes.append(
            f"计数口径统一: “{m.group(1)}次中的{m.group(2)}次” → "
            f"“{_cn(total)}次中的{_cn(b)}次”(与正文“{_cn(total)}次衰退”一致)"
        )
        return f"{_cn(total)}次中的{_cn(b)}次"

    text = re.sub(
        r"([一二三四五六七八九十\d]+)次中的([一二三四五六七八九十\d]+)次",
        fix_self_over_self,
        text,
    )

    def fix_fraction(m: re.Match) -> str:
        a, b = _num(m.group(1)), _num(m.group(2))
        # only rewrite "X分之X" (all-of-X) phrasings conflicting with total
        if a is None or b is None or a != b or a == total:
            return m.group(0)
        result.fixes.append(
            f"计数口径统一: “{m.group(1)}分之{m.group(2)}” → "
            f"“{_cn(total)}分之{_cn(b)}”(与正文“{_cn(total)}次衰退”一致)"
        )
        return f"{_cn(total)}分之{_cn(b)}"

    text = re.sub(
        r"([一二三四五六七八九十\d]+)分之([一二三四五六七八九十\d]+)",
        fix_fraction,
        text,
    )
    result.text = text
    return result


_ABS_NEG_RE = re.compile(r"(文献中不存在|文献里不存在|学界没有|没有文献|不存在.{0,6}文献|没有任何文献)")
_QUALIFIER_RE = re.compile(r"(本轮|本次|此次|目前检索|阶段性|已检索的|检索范围内|在我们检索)")


def guard_negative_conclusions(text: str, truncated: bool) -> LintResult:
    """Absolute non-existence claims must be qualified as this-round results."""
    result = LintResult(text=text)
    hits = []
    for m in _ABS_NEG_RE.finditer(text):
        # sentence-ish window around the claim
        start = max(text.rfind("。", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        sentence = text[start : m.end() + 30]
        if not _QUALIFIER_RE.search(sentence):
            hits.append(m.group(0))
    if hits:
        result.notes.append(
            "⚠ 口径提示:以上“" + "”、“".join(sorted(set(hits))) + "”类表述"
            "仅指本轮检索的阶段性结果,不能解读为该文献在学科中不存在"
            + (";本次运行被截断/不完整,结论尤其应视为阶段性。" if truncated else "。")
        )
    return result


# --- page-number consistency -------------------------------------------------
# Citations are rewritten to carry explicit PDF page numbers ("PDF第2-3页").
# An answer that already prints real page numbers must not simultaneously
# claim that page numbers were not given.  Real-run wording variants:
# "PDF顺序页码未给出", "证据片段自带PDF顺序页码缺失", "证据片段未标注PDF顺序
# 页码（元数据缺失）" (whole bullet lines).
_PAGE_GIVEN_RE = re.compile(r"PDF\s*第\s*\d+(?:\s*[–-]\s*\d+)?\s*页")
# A whole "页码: …" bullet whose content is only a missing-page claim.  Real
# runs use two word orders -- "…PDF顺序页码未给出" and "证据片段未标注PDF顺序
# 页码" -- and may append "（元数据缺失）", which is swallowed with the line.
_PAGE_MISSING_LINE_RE = re.compile(
    r"(?m)^[ \t]*(?:[-*•]|\d+[.、)])?[ \t]*页码[ \t]*[:：][ \t]*"
    r"(?:证据片段(?:自带)?|片段自带)?[ \t]*"
    r"(?:"
    r"(?:PDF)?(?:顺序)?页码(?:未给出|缺失|不可用|未标注)"
    r"|(?:未给出|未标注|缺失)[ \t]*(?:PDF)?(?:顺序)?页码"
    r")"
    r"[^\n]*\n?"
)
_PAGE_MISSING_RES = [
    # Longest wording variants first: a shorter pattern would leave debris
    # behind (e.g. "页码未给出" alone would leave "具体数值").
    re.compile(r"(?:证据片段(?:自带)?|片段自带)?\s*PDF\s*顺序页码(?:未给出|缺失|不可用)(?:具体数值)?"),
    re.compile(r"(?:证据片段(?:自带)?|片段自带)?\s*(?:未给出|未标注)\s*(?:PDF)?(?:顺序)?页码"),
    re.compile(r"页码未给出具体数值"),
    re.compile(r"页码未给出"),
    re.compile(r"未给出(?:PDF)?(?:顺序)?页码(?:具体数值)?"),
    re.compile(r"未给出具体数值"),
]
# "pages 1-2（label(PDF第1–2页)）" -- the bare pages range duplicates the
# page numbers already printed inside the following citation.  The rewritten
# citation is "label(PDF第…页)", so one inner "(" sits between the label and
# "PDF第" and must be allowed by the lookahead.
_DUP_PAGES_RANGE_RE = re.compile(
    r"\bpages\s+\d+(?:\s*[–-]\s*\d+)?\s*(?=[（(][^()（）]{0,80}?[(（]?PDF第)"
)
# "((label(PDF第2–3页)))" -- the citation key wrapped twice; the inner part
# is itself "label(PDF第…页)", so the inner parenthesis pair is allowed.
_DUP_CITE_WRAP_RE = re.compile(
    r"[（(]\s*([（(][^()（）]{0,60}?\(PDF第[\d\s–-]+页\)[)）])\s*[)）]"
)


def fix_page_number_consistency(text: str) -> LintResult:
    """Remove "页码未给出/缺失" claims when the text already prints PDF pages."""
    result = LintResult(text=text)
    if not _PAGE_GIVEN_RE.search(text):
        return result
    out, removed = _PAGE_MISSING_LINE_RE.subn("", text)
    for pattern in _PAGE_MISSING_RES:
        out, n = pattern.subn("", out)
        removed += n
    # "pages X-Y（label(PDF第X–Y页)）" states the same pages twice -> keep
    # only the citation that carries the PDF page numbers.
    out, n_dup = _DUP_PAGES_RANGE_RE.subn("", out)
    # "((label(PDF第X–Y页)))" wraps the citation key twice -> unwrap once.
    # The capture is already the complete inner citation (with its own outer
    # paren), so the redundant outer pair is simply dropped.
    out, n_wrap = _DUP_CITE_WRAP_RE.subn(r"\1", out)
    if removed:
        # tidy dangling separators left behind by the removals
        out = re.sub(r"[,，;；]\s*[。.]", "。", out)
        out = re.sub(r"。\s*[,，;；]\s*", "。", out)
        out = re.sub(r"\(\s*\)|\(\s*[,，;；]\s*\)", "", out)
        out = re.sub(r"。{2,}", "。", out)
        # drop bullets left empty by the removals
        out = re.sub(r"(?m)^[ \t]*(?:[-*•]|\d+[.、)])[ \t]*$", "", out)
        result.fixes.append(
            f"页码一致性:正文已给出PDF页码,移除了{removed}处“页码未给出/缺失”类表述"
        )
    if n_dup or n_wrap:
        result.fixes.append(
            f"页码去重:删除了{n_dup}处与引用内PDF页码重复的裸pages区间、"
            f"修正了{n_wrap}处重复包裹的引用括号"
        )
    result.text = out
    return result


def lint_answer(text: str, truncated: bool = False) -> LintResult:
    """Run all checks; later checks see earlier checks' output."""
    merged = LintResult(text=text)
    for step in (fix_count_consistency, fix_page_number_consistency):
        r = step(merged.text)
        merged.text = r.text
        merged.fixes.extend(r.fixes)
        merged.notes.extend(r.notes)
    r = guard_negative_conclusions(merged.text, truncated=truncated)
    merged.notes.extend(r.notes)
    return merged
