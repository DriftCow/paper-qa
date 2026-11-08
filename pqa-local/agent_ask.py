r"""Ask a question with PaperQA's iterative agent, using the persistent index.

This is the "B" path: ``ToolSelector`` can call ``paper_search`` several times,
``gather_evidence`` with sub-questions, and finally ``gen_answer``.  The
Tantivy index and the per-paper ``Docs`` (including chunk embeddings) are
loaded from disk; ``agent.rebuild_index=False`` prevents a directory rescan.

Usage
-----
    cd pqa-local
    .\.venv\Scripts\python.exe agent_ask.py "your question"
    .\.venv\Scripts\python.exe agent_ask.py --max-timesteps 15 "..."

The first run must be preceded by ``build_agent_index.py``.  If new PDFs are
added, re-run ``build_index.py`` (Docs cache) and ``build_agent_index.py``
(Tantivy index); both are incremental and do not re-embed unchanged papers.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from cjk_search import install_cjk_query_patch
from paperqa.agents.main import agent_query
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"
DEFAULT_CORPUS = BASE_DIR.parent.parent / "PDF"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative PaperQA agent over a persistent index."
    )
    parser.add_argument("question", nargs="+", help="Question to ask.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default="lit")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--search-count",
        type=int,
        default=8,
        help="Papers loaded by each paper_search call (PaperQA default: 8).",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=12,
        help="Cap on agent steps (prevents endless searching).",
    )
    parser.add_argument(
        "--evidence-k",
        type=int,
        default=8,
        help="Evidence chunks summarized per gather_evidence call.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print every evidence context used for the answer.",
    )
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
    logging.getLogger("paperqa").setLevel(logging.WARNING)

    question = " ".join(args.question).strip()
    settings = build_settings()
    settings.verbosity = 0

    settings.agent.index.name = args.name
    settings.agent.index.index_directory = str(args.index_dir)
    settings.agent.index.paper_directory = str(args.corpus.resolve())
    settings.agent.index.use_absolute_paper_directory = True
    # The whole point: load the existing index instead of re-scanning/re-indexing.
    settings.agent.rebuild_index = False

    settings.agent.search_count = args.search_count
    settings.agent.max_timesteps = args.max_timesteps
    settings.agent.timeout = args.timeout
    settings.answer.evidence_k = args.evidence_k

    install_cjk_query_patch()

    print(f"Question: {question}", flush=True)
    print(
        f"Agent: ToolSelector | search_count={args.search_count} | "
        f"max_timesteps={args.max_timesteps} | evidence_k={args.evidence_k}",
        flush=True,
    )
    t0 = time.perf_counter()
    response = await agent_query(question, settings)
    elapsed = time.perf_counter() - t0

    session = response.session
    print("\n=== Answer ===")
    print(session.answer)

    tool_calls = [name for step in session.tool_history for name in step]
    print("\n=== Agent trace ===")
    print("Tools called:", ", ".join(tool_calls) if tool_calls else "(none)")
    print(f"Contexts used: {len(session.contexts)}")
    if args.show_contexts:
        for i, context in enumerate(session.contexts, start=1):
            print(f"\n[{i}] {context.citation} (score={context.score:.2f})")
            print(context.text[:700].replace("\n", " "))

    total_prompt = sum(v[0] for v in session.token_counts.values())
    total_completion = sum(v[1] for v in session.token_counts.values())
    print("\n=== Usage ===")
    print(f"Wall time:     {elapsed:.1f}s")
    print(f"Prompt tokens: {total_prompt:,}")
    print(f"Output tokens: {total_completion:,}")
    if session.cost:
        print(f"Cost (USD):    ${session.cost:.6f}")
    else:
        print("Cost (USD):    n/a (model not in LiteLLM's price map)")
    print(f"Status:        {getattr(response, 'status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
