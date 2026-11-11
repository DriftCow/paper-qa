r"""Multi-turn PaperQA agent chat over the persistent index, with a review mode.

Production entry point for conversational literature Q&A:

* Loads the persistent Tantivy index built by ``build_agent_index.py``
  (``agent.rebuild_index=False``) and installs the CJK bigram query patch from
  ``cjk_search.py``, so Chinese queries actually hit Chinese papers.
* Two presets: ``standard`` for everyday Q&A, ``review`` for survey-style
  questions that need broad coverage (more papers per search, more evidence
  summarized, more sources cited, longer sectioned Chinese answers).
* Multi-turn context within the session: each follow-up is first rewritten by
  the LLM into a standalone retrieval question using recent dialogue history,
  then answered by the agent. History lives in memory only.

Prerequisites (first time, and after adding PDFs):

    .\.venv\Scripts\python.exe build_index.py --allow-low-entropy-text
    .\.venv\Scripts\python.exe build_agent_index.py

Usage
-----
Interactive mode (slash commands: /review /standard /new /history /mode /exit):

    .\.venv\Scripts\python.exe agent_qa.py
    .\.venv\Scripts\python.exe agent_qa.py --review

Single-shot mode (ask one question and exit, like the old demo):

    .\.venv\Scripts\python.exe agent_qa.py "这份语料库主要涵盖哪些研究主题?"

Cost (Kimi kimi-k2.8-preview list prices: input ¥6.5/M tokens, output ¥27/M)
------------------------------------------------------------------------
standard: ~60-80k prompt tokens per question   -> ~¥0.3-0.6
review:   ~250-350k prompt tokens per question -> ~¥1.8-2.6
(cache-read input is only ¥1.7/M, so cached turns cost far less)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import litellm

from answer_lint import lint_answer
from citation_fix import (
    cited_chunk_names,
    load_manifest,
    replace_pqac_keys,
    resolve_metadata,
    rewrite_answer,
    save_manifest,
)
from cjk_search import install_cjk_query_patch
from index_cache import cache_metadata, load_docs
from paper_ledger import PaperLedger
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.main import agent_query
from paperqa.agents.models import AgentStatus
from paperqa.docs import Docs
from qa import build_settings, format_cost
from retrieval_audit import RetrievalAudit
from task_router import (
    EVIDENCE_DISCIPLINE,
    INTENT_LABELS,
    Intent,
    classify_intent,
    extract_membership_title,
    find_retrieval_plans,
    prompts_for,
)

DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"
DEFAULT_CORPUS = BASE_DIR.parent.parent / "PDF"
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_MANIFEST = BASE_DIR / "index" / "citation_manifest.json"
DEFAULT_AUDIT_DIR = BASE_DIR / "audit"


class PreserveDocsEnvironment(PaperQAEnvironment):
    """Environment that keeps externally pre-seeded docs alive.

    The base class clears the docs on reset (it assumes it owns an empty
    ``Docs``); FIND pre-seeding needs its candidates to survive into the
    agent run, so this subclass makes the reset a no-op.
    """

    async def _reset_docs(self) -> None:
        return None

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Every value maps to a public Settings field; see paperqa/settings.py.
# Defaults for reference: search_count=8, evidence_k=10, answer_max_sources=5,
# agent_evidence_n=1, max_timesteps=None, answer_length="about 200 words",
# evidence_summary_length="about 100 words", max_concurrent_requests=4.

REVIEW_QA_PROMPT = (
    "请基于以下证据撰写一篇中文小型文献综述,回答给定问题。\n\n"
    "证据:\n\n{context}\n\n---\n\n"
    "问题:{question}\n\n"
    "写作要求:\n"
    "1. 按主题或维度分成若干小节(加简短小标题),系统梳理证据,不要逐篇罗列流水账;\n"
    "2. 每个论断末尾标注支持它的引用 key,形如 {example_citation};"
    "只引用上方证据中出现过的 key,引用 key 必须原样使用,不要拼接或改写;\n"
    "3. 对不同来源的共识、分歧与研究空白加以比较和综合;\n"
    '4. 证据不足或未涉及的方面,明确写出"现有证据未涵盖";\n'
    "5. 结尾附「参考文献」清单,每行一条,使用与正文相同的引用 key;\n"
    "6. 不要输出与回答无关的内容。\n\n"
    + EVIDENCE_DISCIPLINE
    + "\n{prior_answer_prompt}"
    "综述({answer_length}):"
)

REVIEW_AGENT_PROMPT = (
    "你要为一篇中文文献综述收集证据。问题:{question}\n\n"
    "要求:\n"
    "1. 用 paper_search 检索至少 4-6 次,每次使用不同角度的 query"
    "(覆盖不同子主题、同义词、中英文关键词组合);\n"
    "2. 对最重要的 query,可以用相同的 query 再调用一次 paper_search 翻页,"
    "获取更多相关论文;\n"
    "3. 检索之后,针对各个子主题分别调用 gather_evidence 收集证据;\n"
    "4. 证据覆盖充分后调用 gen_answer 生成综述;"
    "如果答案已经足够全面,调用 {complete_tool_name} 结束。\n\n"
    "当前状态:{status}"
)

PRESETS: dict[str, dict] = {
    "standard": {
        "search_count": 12,
        "evidence_k": 15,
        "answer_max_sources": 15,
        "agent_evidence_n": 5,
        "max_timesteps": 14,
        "answer_length": "约600字",
        "evidence_summary_length": "about 150 words",
        "max_concurrent_requests": 8,
        "qa_prompt": None,  # keep paperqa's default prompts
        "agent_prompt": None,
        "cost_hint": "约 ￥0.3-0.6/次",
    },
    "review": {
        "search_count": 40,
        "evidence_k": 40,
        "answer_max_sources": 40,
        "agent_evidence_n": 8,
        "max_timesteps": 24,
        "answer_length": "1500-2500字,分主题小节",
        "evidence_summary_length": "about 200 words",
        "max_concurrent_requests": 8,
        "qa_prompt": REVIEW_QA_PROMPT,
        "agent_prompt": REVIEW_AGENT_PROMPT,
        "cost_hint": "约 ￥1.8-2.6/次",
    },
}

REWRITE_SYSTEM_PROMPT = (
    "根据对话历史,把用户的追问改写成一个可以独立用于文献检索的完整问题:"
    "补全指代(它/该/这些/其)和被省略的主语,保留关键术语。"
    "只输出改写后的问题本身,不要解释,不要回答。"
)

HELP_TEXT = """\
斜杠命令:
  /review    切换到综述档(广检索、长答案,约 ￥1.8-2.6/次)
  /standard  切换到标准档(日常问答,约 ￥0.3-0.6/次)
  /mode      查看当前档位及参数
  /new       清空对话历史和上一轮文献记录
  /history   查看本次会话的对话历史(含改写后的问题)
  /exit      退出(也可用 /quit 或 Ctrl+C)
  /help      显示本帮助
普通输入即提问。系统会先识别任务类型并按类型输出:
  - 文献查找(“找出所有提到X的文献”)-> 结构化文献证据清单,非综述;
  - 反对意见检索(“有没有文献质疑X”) -> 按直接批评/方法批评/间接挑战/
    替代框架四类报告,检索不完整时只给阶段性结论;
  - 综述(明确要求“综述/研究现状/综合分析”) -> 分节综述;
  - 计数追问(“你上面找到了几篇?”) -> 直接对上一轮文献集合去重计数,
    不重新检索;
  - 成员追问(“《某题名》上面有这篇吗?”) -> 只对照上一轮答案实际
    列出的引用文献回答“有/没有”,不重新检索,不检查其他论文的参考文献表;
  - 其他问题 -> 常规问答。
非首问(计数类和成员类除外)会先用 LLM 把追问改写成独立问题再检索。
"""


def build_chat_settings(args: argparse.Namespace) -> "object":
    """Settings wired to the persistent index (same path as agent_ask.py)."""
    settings = build_settings()
    settings.verbosity = 0
    settings.agent.index.name = args.name
    settings.agent.index.index_directory = str(args.index_dir)
    settings.agent.index.paper_directory = str(args.corpus.resolve())
    settings.agent.index.use_absolute_paper_directory = True
    # Load the existing index instead of re-scanning/re-indexing.
    settings.agent.rebuild_index = False
    settings.agent.timeout = args.timeout
    install_cjk_query_patch()
    return settings


def apply_preset(settings, mode: str, defaults: dict) -> None:
    """Apply a PRESETS entry; prompt overrides fall back to captured defaults."""
    p = PRESETS[mode]
    settings.agent.search_count = p["search_count"]
    settings.answer.evidence_k = p["evidence_k"]
    settings.answer.answer_max_sources = p["answer_max_sources"]
    settings.agent.agent_evidence_n = p["agent_evidence_n"]
    settings.agent.max_timesteps = p["max_timesteps"]
    settings.answer.answer_length = p["answer_length"]
    settings.answer.evidence_summary_length = p["evidence_summary_length"]
    settings.answer.max_concurrent_requests = p["max_concurrent_requests"]
    settings.prompts.qa = p["qa_prompt"] or defaults["qa"]
    settings.agent.agent_prompt = p["agent_prompt"] or defaults["agent"]


async def rewrite_question(history: list[dict], question: str) -> str:
    """Rewrite a follow-up into a standalone retrieval question using history."""
    if not history:
        return question
    convo = "\n\n".join(
        f"问:{h['question']}\n答:{h['answer'][:500]}" for h in history[-4:]
    )
    response = await litellm.acompletion(
        model=f"openai/{os.environ['KIMI_MODEL']}",
        api_base=os.environ["KIMI_API_BASE"],
        api_key=os.environ["KIMI_API_KEY"],
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"对话历史:\n{convo}\n\n追问:{question}"},
        ],
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten or question


async def run_one(
    settings,
    question: str,
    *,
    intent: Intent = Intent.QA,
    paper_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    audit_dir: Path | None = DEFAULT_AUDIT_DIR,
) -> tuple[str, PaperLedger, dict | None]:
    """Run one agent query, post-process citations, print stats, save audit.

    Every ``paper_search`` call is audited (query, top_n, ranked candidates).
    FIND questions additionally pre-seed vector, keyword, and exact-phrase
    candidates into the corpus pool before the agent starts. These are candidate
    unions, so the vector path does not replace or narrow keyword retrieval.

    Returns (displayed answer text, paper ledger, audit payload).
    """
    t0 = time.perf_counter()
    audit = RetrievalAudit()
    audit.install_hook()

    docs: Docs | None = None
    try:
        if intent is Intent.FIND:
            docs = Docs()
            corpus_docs, _ = load_docs(
                DEFAULT_CACHE, expected_meta=cache_metadata(settings)
            )
            n_vector = await audit.run_vector_query(
                docs, corpus_docs, question, settings
            )
            print(f"(FIND向量检索预置: 新增 {n_vector} 篇候选进入语料池)")
            plans = find_retrieval_plans(question)
            if plans:
                n_pre = await audit.run_retrieval_plans(docs, plans, settings)
                print(
                    f"(FIND多路检索预置: {len(plans)} 条查询"
                    f"(关键词+短语), 新增 {n_pre} 篇候选进入语料池)"
                )
        if docs is not None:
            response = await agent_query(
                question, settings, docs=docs, env_class=PreserveDocsEnvironment
            )
        else:
            response = await agent_query(question, settings)
    finally:
        audit.uninstall_hook()
    elapsed = time.perf_counter() - t0
    session = response.session
    status = getattr(response, "status", None)
    truncated = status == AgentStatus.TRUNCATED

    answer = session.answer

    # -- citation post-processing -------------------------------------------
    manifest = load_manifest(manifest_path)
    keyed = replace_pqac_keys(answer, session)
    cited_docnames = {
        name.rpartition(" pages ")[0]
        for name in cited_chunk_names(keyed, session)
    }
    # Evidence coverage check (FIND): papers whose evidence directly supports
    # the question but were dropped by the answer generator get a supplement
    # item appended here -- before the rewrite, so their citations go through
    # the same label/reference/verbatim pipeline as the LLM's own items.
    coverage_report = None
    if intent is Intent.FIND:
        from evidence_coverage import check_and_supplement

        try:
            keyed, coverage_report = await check_and_supplement(
                question, keyed, session, cited_docnames, manifest
            )
        except Exception as exc:  # noqa: BLE001 - coverage check must not kill the answer
            print(f"(覆盖检查失败,按原答案继续: {type(exc).__name__}: {exc})")
        if coverage_report is not None and coverage_report.added_docnames:
            # supplements introduced new chunk-name citations
            cited_docnames |= {
                name.rpartition(" pages ")[0]
                for name in cited_chunk_names(keyed, session)
            }
    # Resolve metadata for *all* cited papers (not just opaque docnames) so
    # "元数据缺失" markers and bare filename labels can be filled from each
    # paper's own first pages; results are cached in the manifest.  Cached
    # entries whose authors contradict the filename surname are re-resolved.
    await resolve_metadata(
        sorted(cited_docnames),
        paper_dir,
        manifest,
        llm_extract=_llm_extract_or_none(),
        resolve_all=True,
    )
    save_manifest(manifest_path, manifest)
    # ``keyed`` carries the pqac-key replacement (and, for FIND, any coverage
    # supplement items); non-FIND intents keep the same rewrite input as before.
    rewrite = rewrite_answer(keyed, session, manifest)
    lint = lint_answer(rewrite.text, truncated=truncated)

    print("\n=== 回答 ===")
    print(lint.text)
    for fix in lint.fixes:
        print(f"\n[一致性修正] {fix}")
    for note in lint.notes:
        print(f"\n{note}")
    if truncated:
        print(
            "\n⚠ 本次运行达到步数/时间上限被截断,以上为阶段性结论,"
            "不代表检索已穷尽或文献中不存在相反证据。"
        )

    # -- statistics -----------------------------------------------------------
    tool_calls = [name for step in session.tool_history for name in step]
    n_search = tool_calls.count("paper_search")
    n_gather = tool_calls.count("gather_evidence")
    dockeys = set()
    for context in session.contexts:
        doc = getattr(getattr(context, "text", None), "doc", None)
        if doc is not None:
            dockeys.add(doc.dockey)

    coverage_reasons = None
    if coverage_report is not None:
        coverage_reasons = {
            v.dockey: v.audit_reason()
            for v in coverage_report.verdicts.values()
            if not v.added
        }
    audit_payload = audit.to_dict(
        question, intent.value, session, rewrite.cited_docnames,
        coverage_reasons=coverage_reasons,
    )
    print("\n=== 统计(口径分列) ===")
    print(f"检索动作: paper_search×{n_search}, gather_evidence×{n_gather}")
    counts = audit_payload["counts"]
    print(
        f"检索命中候选(去重): {counts['n_candidate_papers']} 篇 | "
        f"进入证据的文献(去重): {len(dockeys)} 篇 | "
        f"读取的证据片段: {len(session.contexts)} 条"
    )
    if coverage_report is not None:
        print(coverage_report.summary())
    print(
        f"正文实际引用的独立文献(稳定ID去重): {rewrite.cited_group_count} 篇"
        + (
            f"(其中元数据缺失 {len(rewrite.unresolved)} 篇)"
            if rewrite.unresolved
            else ""
        )
    )
    print(f"引用能回读到证据片段: {rewrite.verified} 处 | "
          f"引号内容通过逐字核验: {rewrite.quote_passed}/{rewrite.quote_checked} 处"
          + (
              f"(其中 {rewrite.quote_promoted} 处由'释义'核验为逐字后提升为原文摘录)"
              if rewrite.quote_promoted
              else ""
          ))
    if counts["n_evidence_not_cited"]:
        print(
            f"进入证据但未进入最终答案: {counts['n_evidence_not_cited']} 篇"
            "(排除原因见检索审计)"
        )
    total_prompt = sum(v[0] for v in session.token_counts.values())
    total_completion = sum(v[1] for v in session.token_counts.values())
    print(f"耗时: {elapsed:.1f}s | tokens: {total_prompt:,} in / {total_completion:,} out")
    if session.cost:
        print(f"成本: {format_cost(session.cost)} | 状态: {status}")
    else:
        print(f"状态: {status}")
    if audit_dir is not None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        audit_path = audit_dir / f"audit_{stamp}_{intent.value}.json"
        audit.save(audit_payload, audit_path)
        print(f"检索审计: {audit_path}")

    ledger = PaperLedger.from_session(session, question, intent.value)
    # Tier parsing runs on the raw answer (its pqac keys keep the key-based
    # binding alive) plus the coverage supplement block when one was inserted,
    # so supplemented papers get their 直接表述 tier into the same structured
    # parse as the LLM's own items.
    tier_source = session.raw_answer
    if coverage_report is not None and coverage_report.supplement_text:
        tier_source = (
            f"{session.raw_answer.rstrip()}\n\n{coverage_report.supplement_text}"
        )
    ledger.restrict_to_cited(rewrite.cited_docnames, tier_source, session)
    # Attach readable labels + DOI + title from the manifest for nicer reports.
    for entry in ledger.entries:
        meta = manifest.get(entry.docname)
        if meta is not None:
            entry.label = meta.label
            entry.doi = entry.doi or meta.doi
            entry.title = entry.title or meta.title
    return lint.text, ledger, audit_payload


def _llm_extract_or_none():
    """LLM metadata extractor; disabled when API env is missing (offline)."""
    import os

    if os.environ.get("KIMI_API_KEY"):
        from citation_fix import default_llm_extract

        return default_llm_extract
    return None


def resolve_paper_dir(cli_corpus: Path) -> Path:
    """Directory actually containing the indexed PDFs.

    The Tantivy index is queried with ``rebuild_index=False``, so the CLI
    ``--corpus`` default may be stale; the build report records the corpus
    that was really indexed.  Used only for read-only first-page metadata
    extraction of cited papers.
    """
    import json

    report = BASE_DIR / "index" / "build_report.json"
    try:
        corpora = json.loads(report.read_text(encoding="utf-8")).get("corpus") or []
    except Exception:  # noqa: BLE001
        corpora = []
    for c in corpora:
        if Path(c).is_dir():
            return Path(c)
    return cli_corpus


def print_mode(mode: str) -> None:
    p = PRESETS[mode]
    print(
        f"当前档位: {mode} | search_count={p['search_count']} "
        f"evidence_k={p['evidence_k']} max_sources={p['answer_max_sources']} "
        f"max_timesteps={p['max_timesteps']} | 成本 {p['cost_hint']}"
    )


async def chat_loop(settings, mode: str, defaults: dict, paper_dir: Path) -> None:
    history: list[dict] = []
    ledger: PaperLedger | None = None  # previous turn's countable paper set
    print("\n进入多轮对话模式。输入 /help 查看命令,普通输入即提问。")
    print_mode(mode)
    while True:
        try:
            raw = input(f"\n[{mode}] 提问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not raw:
            continue
        if raw.startswith("/"):
            cmd = raw.split()[0].lower()
            if cmd in ("/exit", "/quit"):
                print("再见。")
                return
            if cmd == "/new":
                history.clear()
                ledger = None
                print("已清空对话历史和上一轮文献记录。")
            elif cmd == "/history":
                if not history:
                    print("(暂无历史)")
                for i, h in enumerate(history, start=1):
                    print(f"[{i}] 问: {h['question']}")
                    if h["rewritten"] != h["question"]:
                        print(f"    改写后: {h['rewritten']}")
            elif cmd in ("/review", "/standard"):
                mode = cmd[1:]
                apply_preset(settings, mode, defaults)
                print_mode(mode)
            elif cmd == "/mode":
                print_mode(mode)
            elif cmd == "/help":
                print(HELP_TEXT)
            else:
                print("未知命令,输入 /help 查看。")
            continue

        question = raw
        intent = classify_intent(question, has_ledger=ledger is not None)

        # 计数类追问:直接复用上一轮保存的文献集合,绝不重新检索。
        if intent is Intent.COUNT:
            if ledger is None:
                print(
                    "\n上一轮没有保存可计数的文献集合(可能上一轮未产生"
                    "带引用的回答)。请先提出检索类问题,再追问数量。"
                )
            else:
                print("\n=== 回答(基于上一轮文献集合,未重新检索) ===")
                print(ledger.count_report())
            history.append(
                {"question": question, "rewritten": question, "answer": "(计数回答)"}
            )
            continue

        # 成员类追问("《题名》上面有这篇吗?"):只对照上一轮答案实际列出的
        # 引用文献集合回答,在 LLM 改写和重新检索之前处理,绝不扫描其他
        # 论文的参考文献表。ledger 不更新:成员查询针对最近一次检索答案。
        if intent is Intent.PAPER_MEMBERSHIP:
            title = extract_membership_title(question)
            if ledger is None:
                print(
                    "\n上一轮没有保存可查询的文献集合(可能上一轮未产生"
                    "带引用的回答)。请先提出检索类问题。"
                )
            else:
                print(
                    "\n=== 回答(仅对照上一轮答案实际列出的文献,未重新检索) ==="
                )
                print(ledger.membership_answer(title))
            history.append(
                {"question": question, "rewritten": question, "answer": "(成员查询回答)"}
            )
            continue

        # 任务意图优先于固定输出模板:按意图选择生成/检索提示词。
        settings.prompts.qa, settings.agent.agent_prompt = prompts_for(
            intent, PRESETS[mode], defaults
        )
        if intent is not Intent.QA:
            print(f"(识别为{INTENT_LABELS[intent]},使用对应输出模板)")
        try:
            rewritten = await rewrite_question(history, question)
        except Exception as exc:  # rewrite failure must not lose the question
            print(f"(追问改写失败,按原问题检索: {exc})")
            rewritten = question
        if rewritten != question:
            print(f"改写后: {rewritten}")
        ledger_answer, ledger, _audit = await run_one(
            settings, rewritten, intent=intent, paper_dir=paper_dir
        )
        history.append(
            {"question": question, "rewritten": rewritten, "answer": ledger_answer}
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn PaperQA agent chat (standard / review presets)."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. If given, runs once and exits (no chat loop).",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Start in review mode (broad retrieval, survey-style answer).",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default="lit")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
    logging.getLogger("paperqa").setLevel(logging.WARNING)

    settings = build_chat_settings(args)
    defaults = {"qa": settings.prompts.qa, "agent": settings.agent.agent_prompt}
    mode = "review" if args.review else "standard"
    apply_preset(settings, mode, defaults)
    paper_dir = resolve_paper_dir(args.corpus)

    question = " ".join(args.question).strip()
    if question:
        print(f"问题: {question}", flush=True)
        print_mode(mode)
        intent = classify_intent(question, has_ledger=False)
        if intent is Intent.COUNT:
            print(
                "单次模式没有上一轮文献集合可供计数;"
                "请用交互模式(不带问题参数启动)先检索再追问。"
            )
            return
        if intent is Intent.PAPER_MEMBERSHIP:
            print(
                "单次模式没有上一轮答案可供对照;"
                "请用交互模式(不带问题参数启动)先检索再追问某篇是否列出。"
            )
            return
        settings.prompts.qa, settings.agent.agent_prompt = prompts_for(
            intent, PRESETS[mode], defaults
        )
        if intent is not Intent.QA:
            print(f"(识别为{INTENT_LABELS[intent]},使用对应输出模板)")
        await run_one(settings, question, intent=intent, paper_dir=paper_dir)
        return
    await chat_loop(settings, mode, defaults, paper_dir)


if __name__ == "__main__":
    asyncio.run(main())
