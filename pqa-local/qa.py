"""End-to-end paper-qa demo: index PDFs and ask a question.

Put one or more PDFs into the papers/ folder next to this script; all of them
are indexed on each run, and the question is answered across all papers.

LLM: DeepSeek (deepseek-chat) via OpenAI-compatible endpoint.
Embedding: Aliyun DashScope text-embedding-v4 via OpenAI-compatible
endpoint (override with DASHSCOPE_EMBEDDING_MODEL in .env).
Keys are read from .env next to this script.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import litellm
from paperqa import Docs, Settings
from paperqa.settings import AgentSettings, MultimodalOptions, ParsingSettings
from paperqa_pypdf import parse_pdf_to_pages as parse_pdf_pypdf


def _register_llm_pricing(model: str) -> None:
    """LiteLLM's price map has no entry for the endpoint's model name, which
    makes every call log "Failed to calculate cost". Register DeepSeek's list
    prices (per million tokens, overridable via env) so per-call and session
    costs are actually computed."""
    # Unmapped models make litellm print a "Provider List" banner on every
    # cost lookup; we register pricing ourselves, so silence it.
    litellm.suppress_debug_info = True
    prices = {
        "mode": "chat",
        "litellm_provider": "openai",
        "max_tokens": 8192,
        "max_input_tokens": 65536,
        "max_output_tokens": 8192,
        "input_cost_per_token": float(
            os.environ.get("DEEPSEEK_INPUT_COST_PER_MTOK", "0.27")
        )
        / 1_000_000,
        "output_cost_per_token": float(
            os.environ.get("DEEPSEEK_OUTPUT_COST_PER_MTOK", "1.10")
        )
        / 1_000_000,
        "cache_read_input_token_cost": float(
            os.environ.get("DEEPSEEK_CACHE_READ_COST_PER_MTOK", "0.07")
        )
        / 1_000_000,
        "supports_prompt_caching": True,
    }
    # The gateway answers with model="deepseek-flash" even when the request
    # asked for another name, so lmi's cost lookup (keyed on the response's
    # model) needs an entry for that name too.
    for name in {model, "deepseek-flash"}:
        litellm.register_model({name: prices})


def deepseek_llm_config() -> tuple[str, dict]:
    """Return (model_name, LiteLLM Router model_list config) for DeepSeek."""
    model = os.environ["DEEPSEEK_MODEL"]
    _register_llm_pricing(model)
    config = {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"openai/{model}",
                    "api_base": os.environ["DEEPSEEK_API_BASE"],
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            }
        ]
    }
    return model, config


def build_settings() -> Settings:
    model, llm_config = deepseek_llm_config()
    embedding_model = os.environ.get(
        "SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"
    )
    return Settings(
        llm=model,
        llm_config=llm_config,
        summary_llm=model,
        summary_llm_config=llm_config,
        embedding=f"openai/{embedding_model}",
        # lmi's embedding_model_factory passes this dict straight through as
        # LiteLLMEmbeddingModel.config; its "kwargs" go to litellm.aembedding.
        embedding_config={
            # Free-tier bge-m3 enforces a server-side TPM cap; throttle
            # client-side (moving-window token bucket weighted by estimated
            # tokens per batch) to stay under it. Lower the number if 429s
            # still appear.
            "rate_limit": "300000/minute",
            "kwargs": {
                "api_base": "https://api.siliconflow.cn/v1",
                "api_key": os.environ["SILICONFLOW_API_KEY"],
                # DashScope rejects litellm's default encoding_format
                "encoding_format": "float",
                # Retry 429s with backoff so per-minute TPM windows are
                # waited out instead of failing the whole file.
                "num_retries": 8,
            },
        },
        agent=AgentSettings(agent_llm=model, agent_llm_config=llm_config),
        parsing=ParsingSettings(
            # Keep cache identity stable when the optional PyMuPDF extra is installed.
            parse_pdf=parse_pdf_pypdf,
            # text-only LLM, so skip image parsing/enrichment
            multimodal=MultimodalOptions.OFF,
            # skip Crossref/Semantic Scholar metadata lookups (rate-limited here)
            use_doc_details=False,
            enrichment_llm=model,
            enrichment_llm_config=llm_config,
        ),
        verbosity=1,
    )


async def main() -> None:
    settings = build_settings()
    pdf_dir = BASE_DIR / "papers"
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found. Put your papers into {pdf_dir} and re-run.")
        sys.exit(1)

    docs = Docs()
    t0 = time.perf_counter()
    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[1/2] Indexing ({i}/{len(pdf_paths)}) {pdf_path.name} ...", flush=True)
        docname = await docs.aadd(pdf_path, settings=settings)
        print(f"      Added doc '{docname}'", flush=True)
    t_add = time.perf_counter() - t0
    print(
        f"      Indexed {len(docs.docs)} doc(s),"
        f" {len(docs.texts)} chunks total in {t_add:.1f}s",
        flush=True,
    )

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this paper about?"
    print(f"[2/2] Querying: {question}", flush=True)
    t0 = time.perf_counter()
    session = await docs.aquery(question, settings=settings)
    t_query = time.perf_counter() - t0
    print(f"      Answered in {t_query:.1f}s", flush=True)
    print("\n=== Answer ===")
    print(session.answer)
    print(f"\n=== Contexts used: {len(session.contexts)} ===")
    print(f"=== Cost (USD): {session.cost:.4f} ===")


if __name__ == "__main__":
    asyncio.run(main())
