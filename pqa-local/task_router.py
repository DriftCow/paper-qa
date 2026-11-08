r"""Rule-based task router for agent_qa.py.

Not every user question is a survey request.  This module classifies each
incoming question into one of six intents *before* any retrieval happens, so
that the answer format follows the task instead of a fixed template:

* ``COUNT``            -- meta-question about the previous turn's paper set
                         ("你上面总共找到了几篇?").  Answered locally from the
                         saved ledger; never triggers a new retrieval.
* ``PAPER_MEMBERSHIP`` -- "《题名》上面有这篇吗?"  Answered locally by checking
                         the previous turn's *actually cited* paper set; never
                         triggers retrieval, LLM rewriting, or reference-list
                         scanning of other papers.
* ``SURVEY``           -- user explicitly asks for a 综述/研究现状/综合分析.
* ``CRITIQUE``         -- looking for papers that question/oppose a claim
                         ("有没有文献质疑某个观点?").
* ``FIND``             -- looking for papers that mention/support a claim
                         ("找出所有提到某个观点的文献").
* ``QA``               -- ordinary question answering (fallback).

Classification is deterministic (ordered regex rules) so it is testable
offline and does not cost LLM tokens.  Retrieval breadth still comes from the
standard/review preset; the intent only selects prompts and output shape.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(Enum):
    COUNT = "count"
    PAPER_MEMBERSHIP = "paper_membership"
    SURVEY = "survey"
    CRITIQUE = "critique"
    FIND = "find"
    QA = "qa"


INTENT_LABELS = {
    Intent.COUNT: "上一轮文献计数",
    Intent.PAPER_MEMBERSHIP: "上一轮文献成员查询",
    Intent.SURVEY: "综述",
    Intent.CRITIQUE: "反对意见检索",
    Intent.FIND: "文献查找",
    Intent.QA: "问答",
}


# --- COUNT -----------------------------------------------------------------
# Meta-questions about the previous answer's paper set.  Must reference the
# previous turn explicitly (上面/上一轮/刚才/...) or be a bare count question.
_COUNT_PATTERNS = [
    r"(上面|上一轮|上轮|上次|刚才|之前|前面|上述).{0,20}(几篇|多少篇|数量|几 篇|计数|统计)",
    # The trailing noun must be a countable plural ("找到的文献"), not the
    # demonstrative "这篇/该篇" of a membership question.
    r"(上面|上一轮|上轮|上次|刚才|之前|前面|上述).{0,20}(找到|列出|检索到|找到过).{0,10}(?<![这该此每哪])(篇|文献|论文|文章)",
    r"^(总共|一共|共)?(找到|找到过|列出|检索到)?了?(几篇|多少篇)",
    r"(how many|count).{0,40}(papers|articles|references|sources)",
]

# --- PAPER_MEMBERSHIP ---------------------------------------------------------
# "《题名》上面有这篇吗?" / "刚才列出这篇了吗?" -- membership of a specific
# paper in the previous answer's cited set.  These MUST reference a previous
# turn explicitly, so that "这篇是否相关"/"这篇在哪里" (no previous-turn
# reference) keep routing to QA/FIND and locate the paper's own evidence.
_PREV_TURN_RE = r"(上面|上一轮|上轮|上次|刚才|之前|前面|上述)"

_MEMBERSHIP_PATTERNS = [
    # 上面 ... 有 ... 这篇
    _PREV_TURN_RE + r".{0,80}(有|包含|列出|提到|出现|收录)[^。?？\n]{0,30}(这篇|该篇|此篇|该文献|这篇文章|这篇文献|这篇论文)",
    # 这篇 ... 上面 ... (有|在|列出)
    r"(这篇|该篇|此篇|这篇文章|这篇文献|这篇论文).{0,60}" + _PREV_TURN_RE + r".{0,30}(有|在|列出|出现|收录)",
    # 《题名》 ... 上面 ... 有
    r"(《[^》]{3,300}》|“[^”]{3,300}”|\"[^\"]{3,300}\").{0,80}" + _PREV_TURN_RE + r"[^。?？\n]{0,40}(有|在|列出|出现|收录|包含)",
    # 上面 有没有 《题名》/这篇
    _PREV_TURN_RE + r"[^。?？\n]{0,40}(有没有|是否有|有没有提到|有没有列出|在不在)[^。?？\n]{0,60}(《[^》]{3,300}》|\"[^\"]{3,300}\"|“[^”]{3,300}”|这篇|该篇|此篇)",
    r"(is|was)\s+(it|this\s+(paper|article|one)|that\s+(paper|article|one))\s+(listed|included|cited|mentioned)\s+(above|earlier|in\s+the\s+(previous|last|above))",
]


def extract_membership_title(question: str) -> str | None:
    """Best-effort extraction of the paper title a membership question names.

    Priority: explicit 《...》/quoted span > text before the previous-turn
    marker or "这篇" pronoun.  Returns None when nothing title-like remains.
    """
    q = question.strip()
    m = re.search(r"《([^》]{3,300})》", q)
    if m:
        return m.group(1).strip()
    m = re.search(r"[“\"]([^”\"]{3,300})[”\"]", q)
    if m:
        return m.group(1).strip()
    # Cut at the first previous-turn marker or demonstrative, take what's left.
    cut = re.search(
        _PREV_TURN_RE + r"|这篇|该篇|此篇|这篇文章|这篇文献|这篇论文|有没有|是否有",
        q,
    )
    if not cut:
        return None
    title = q[: cut.start()].strip()
    title = title.rstrip("，,。.；;:：?？!！~的 ")
    return title or None


# --- SURVEY ----------------------------------------------------------------
# Only explicit survey/analysis requests; a topic question alone is QA.
_SURVEY_PATTERNS = [
    r"(写|生成|做|整理|撰写).{0,8}(综述|文献综述|调研报告)",
    r"(综述|文献综述|研究现状|研究进展|研究动态|发展脉络)",
    r"(综合分析|系统梳理|全面梳理|系统回顾)",
    r"(systematic|literature|narrative)\s+review|survey of",
]

# --- CRITIQUE ---------------------------------------------------------------
_CRITIQUE_PATTERNS = [
    r"(质疑|反对|批评|批判|反驳|驳斥|挑战|否定|怀疑|商榷|争议)",
    r"(有没有|是否有|是否存在).{0,20}(文献|论文|研究|学者).{0,12}(质疑|反对|批评|挑战|不认同|否定|怀疑)",
    r"(批评|质疑|挑战|反对).{0,12}(文献|论文|研究)",
    r"(critic|critique|criticism|challenge|disput|argues? against|question(s|ed)? whether|pushback)",
]

# --- FIND -------------------------------------------------------------------
_FIND_PATTERNS = [
    # find-verb + literature-word anywhere in the question
    r"(找出|帮我找|查找|检索|列出|搜罗|搜集)[\s\S]{0,40}?(文献|论文|文章|说法|论述|表述|观点)",
    r"(哪些|那一些).{0,4}(文献|论文|文章|研究|作者)",
    r"(所有|全部).{0,8}(文献|论文|文章).{0,12}(提到|提及|认为|指出|主张|讨论|支持)",
    r"(提到|提及|认为|指出|主张).{0,20}(的)?(文献|论文|文章|研究).{0,6}(有|都|哪些|找|列)",
    r"类似(的)?(说法|表述|论述|观点|提法)",
    r"(find|list|identify|which)\s+(all\s+)?(the\s+)?(papers?|articles?|studies|references)",
    r"papers?\s+(that|which)\s+(mention|argue|claim|state|support)",
]


# --- FIND multi-query retrieval plans -----------------------------------------
# Root cause of the recall failure (verified against a live index): with
# 1000+ papers, BM25 *keyword* queries under OR semantics rarely surface a
# short target paper into the top-N, while *phrase* queries quoting the
# target's exact wording hit rank 1.  A retrieval plan therefore mixes a
# keyword angle (recall on synonyms) with a phrase angle (precision on
# exact formulations).  Plans are executed before the agent starts, so
# their candidates enter the corpus pool ahead of gather_evidence.
# A corpus-specific theme can get a curated multi-angle plan: register a
# ``re.compile`` topic detector plus a ``list[(mode, query)]`` here and
# return it from ``find_retrieval_plans`` when it matches.

_CLAIM_RE = re.compile(
    r"(?:提到|谈到|认为|指出|主张|说过?|支持)([^,，。;；!！?？]{4,80}?)"
    r"(?:这样)?的?(?:类似|相关|相同)?(?:说法|表述|论述|观点|提法|文献|论文|文章)"
)


def find_retrieval_plans(question: str) -> list[tuple[str, str]]:
    """Retrieval plan for a FIND question: ``[(mode, query), ...]``.

    The claim text found in the question becomes both a keyword query and a
    quoted phrase query.  Returns [] when nothing plausible can be planned
    -- the agent's own paper_search calls then remain the only retrieval
    path.
    """
    q = question.strip()
    if not q:
        return []
    m = _CLAIM_RE.search(q)
    if m:
        claim = m.group(1).strip()
        if claim:
            return [("keyword", claim), ("phrase", f'"{claim}"')]
    return []


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_intent(question: str, has_ledger: bool = False) -> Intent:
    """Classify a user question into an Intent.

    ``has_ledger`` tells whether a previous turn saved a countable paper set;
    COUNT patterns only fire when a ledger exists OR the question explicitly
    references a previous turn (so single-shot count questions still route to
    COUNT and get an honest "no saved set" reply instead of a new retrieval).
    """
    q = question.strip()
    if not q:
        return Intent.QA

    count_hit = _matches(_COUNT_PATTERNS, q)
    references_previous = bool(re.search(r"(上面|上一轮|上轮|上次|刚才|之前|前面|上述)", q))
    if count_hit and (has_ledger or references_previous):
        return Intent.COUNT

    # Membership follow-ups are answered from the previous turn's actually
    # cited set; checked before SURVEY/FIND so they never trigger retrieval,
    # LLM rewriting, or reference-list scanning of other papers.
    if _matches(_MEMBERSHIP_PATTERNS, q):
        return Intent.PAPER_MEMBERSHIP

    if _matches(_SURVEY_PATTERNS, q):
        return Intent.SURVEY
    if _matches(_CRITIQUE_PATTERNS, q):
        return Intent.CRITIQUE
    if _matches(_FIND_PATTERNS, q):
        return Intent.FIND
    return Intent.QA


# ---------------------------------------------------------------------------
# Shared prompt blocks
# ---------------------------------------------------------------------------

# Evidence-discipline rules appended to every generation prompt.  These are
# *constraints on claims*, not style polish: they exist to stop the answer
# from upgrading weak evidence into strong conclusions.
EVIDENCE_DISCIPLINE = (
    "证据纪律(必须遵守):\n"
    "- 区分命题强度:支持较弱命题(如相关关系、必要条件)的证据,"
    "不得当作支持更强命题(如首要或决定性因素)的证据;\n"
    "- 相关分析不得写成因果决定;思想实验不得写成实证测定;模型情景不得写成已发生事实;"
    "作者猜测不得写成精确阈值;二手转述不得写成已核对原始文献(转述须标注\"转引自\");\n"
    "- 不同国家、不同系统边界、不同对象得到的阈值不得当成同一参数直接比较;\n"
    "- 保留每篇文献的方法与限定条件(如理论框架、相关分析、计量估计、模型情景各自的方法学边界),"
    "不得抹去其方法学限定;\n"
    "- 每个引用只支撑紧邻它的那一句论断,不得用一篇文献笼统支撑整段多个强结论;\n"
    "- \"本轮证据未覆盖\"不得写成整个学科的研究空白。\n"
)

FIND_QA_PROMPT = (
    "请基于以下证据,找出并列出与问题相关的文献,输出结构化文献证据清单。\n\n"
    "证据:\n\n{context}\n\n---\n\n"
    "问题:{question}\n\n"
    "输出要求:\n"
    "1. 逐条编号,每篇文献只出现一次(同一文献的多条证据合并到该文献条目下);\n"
    "2. 每条包含:作者、年份、完整题名(以该文献自身证据中的信息为准,无法确认时写\"元数据缺失\";"
    "禁止根据其他论文的参考文献表或猜测补写未核实的元数据;"
    "作者名单只取自该文献自身的标题页/作者行,不得从其他论文的引文中拼凑);\n"
    "3. 与问题相关的直接表述或准确释义;\n"
    "4. 原文摘录:优先直接复制证据文本中支撑论断的完整原句并加英文双引号——"
    "逐字复制即可通过核验,不要因为怕错而全部改写成释义;"
    "摘录后标注引用key,形如 {example_citation};"
    "只使用上方证据中出现过的key,原样使用,不得拼接或改写;"
    "带引号的摘录必须是所引证据文本的逐字子串(允许换行与空白差异,不允许增删改写词句);"
    "确实无法逐字引用时才改为不带引号的释义并标注\"(释义)\";\n"
    "5. 支持类型(三选一):直接表述 / 间接支持 / 仅背景相关,格式为\"支持类型:直接表述\",\n"
    "   每条编号条目必须有一行\"支持类型:…\",便于结构化核对;\n"
    "6. 页码:证据片段自带PDF顺序页码,直接采用并注明;不要写\"页码未给出\";\n"
    "7. 已进入上方证据、且其内容直接表述或直接支持问题所问说法的文献,"
    "默认必须列入清单;不列入时必须说明理由,不得无说明地略去;\n"
    "8. 不要写\"引言、共识、分歧、研究空白\"等综述章节,不要泛泛概括主题;\n"
    "9. 不承诺穷尽性:结尾写明\"以上为本次检索命中的文献,不保证语料库中无遗漏\";\n"
    "10. 结尾附「参考文献」清单,每行一条,使用与正文相同的引用key,且只列正文实际引用过的key;\n"
    "11. 页码与引用key各标注一次即可,不要在同一处重复堆砌。\n\n"
    + EVIDENCE_DISCIPLINE
    + "\n{prior_answer_prompt}"
    "文献证据清单({answer_length}):"
)

FIND_AGENT_PROMPT = (
    "用户要找出所有与问题相关的文献(不是写综述)。问题:{question}\n\n"
    "要求:\n"
    "0. 系统已针对该问题把多路检索(关键词+原文短语)命中的候选文献预置进语料池;"
    "这些候选仍需你用 gather_evidence 逐篇取证核验,你仍须继续多角度检索;\n"
    "1. 用 paper_search 检索至少 3-5 次,每次换不同角度的 query"
    "(同义词、近义表述、中英文关键词组合),以提高查全率;"
    "并至少包含一次带双引号的原文短语式 query"
    "(如 \"观点原句的逐字短语\"),精确表述的命中率远高于宽泛关键词;\n"
    "2. 对最重要的 query 可以再用相同 query 调用一次 paper_search 翻页;\n"
    "3. 然后对每个候选方向调用 gather_evidence,收集带原文摘录的证据;\n"
    "4. 证据足够后调用 gen_answer 生成文献证据清单;\n"
    "5. 完成后调用 {complete_tool_name} 结束。\n\n"
    "当前状态:{status}"
)

CRITIQUE_QA_PROMPT = (
    "请基于以下证据,检索并整理对问题中观点的质疑、批评或替代性论述。\n\n"
    "证据:\n\n{context}\n\n---\n\n"
    "问题:{question}\n\n"
    "输出要求:\n"
    "1. 分四类分别报告(没有的类别写\"本轮检索未命中\",四类标题固定):\n"
    "   一、直接批评;二、方法或边界批评;三、间接挑战;四、替代理论框架;\n"
    "2. 每条给出:作者、年份、题名(以证据引用信息为准)、批评或挑战的具体内容、"
    "原文摘录(带引号),并标注引用key,形如 {example_citation};"
    "只使用证据中出现过的key,原样使用;\n"
    "3. \"本轮未检索到\"只能写成本轮检索的阶段性结果,"
    "严禁写成\"文献中不存在\"\"学界没有\"这类全称否定;\n"
    "4. 对已命中的论文,只依据已读到的证据下结论;"
    "不得仅凭前一两页的证据断言该论文全文没有其他批评内容;\n"
    "5. 结尾附「参考文献」清单,每行一条,使用与正文相同的引用key,且只列正文实际引用过的key。\n\n"
    + EVIDENCE_DISCIPLINE
    + "\n{prior_answer_prompt}"
    "批评与替代观点梳理({answer_length}):"
)

CRITIQUE_AGENT_PROMPT = (
    "用户在检索质疑/反对某观点的文献。问题:{question}\n\n"
    "要求:\n"
    "1. 用 paper_search 检索至少 3-5 次,query 要覆盖批评性词汇"
    "(critique/criticism/limitation/problem/challenge/质疑/批评/局限)"
    "与被批评概念的组合,也要检索替代理论框架;\n"
    "2. 对命中的关键论文,用 gather_evidence 针对\"该文后半部分/讨论部分"
    "是否包含批评或局限\"单独取证,不要只看开头;\n"
    "3. 证据足够后调用 gen_answer;\n"
    "4. 完成后调用 {complete_tool_name} 结束。\n\n"
    "当前状态:{status}"
)


def prompts_for(intent: Intent, preset: dict, defaults: dict) -> tuple[str, str]:
    """Return (qa_prompt, agent_prompt) for an intent.

    Intent takes precedence over the preset's fixed template: FIND/CRITIQUE
    get their own prompts in both modes; SURVEY uses the review templates;
    QA keeps whatever the preset says (paperqa defaults in standard mode).
    """
    if intent is Intent.FIND:
        return FIND_QA_PROMPT, FIND_AGENT_PROMPT
    if intent is Intent.CRITIQUE:
        return CRITIQUE_QA_PROMPT, CRITIQUE_AGENT_PROMPT
    if intent is Intent.SURVEY:
        return (
            preset["qa_prompt"] or defaults["qa"],
            preset["agent_prompt"] or defaults["agent"],
        )
    # QA: preset templates only when they exist (review mode), else defaults.
    return preset["qa_prompt"] or defaults["qa"], (
        preset["agent_prompt"] or defaults["agent"]
    )
