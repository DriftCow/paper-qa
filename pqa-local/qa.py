"""End-to-end paper-qa demo: index PDFs and ask a question.

Put one or more PDFs into the papers/ folder next to this script; all of them
are indexed on each run, and the question is answered across all papers.

LLM: Kimi (kimi-k2.8-preview) via OpenAI-compatible endpoint.
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


def _usd_cny_rate() -> float:
    """USD→CNY exchange rate (KIMI_USD_CNY_RATE in .env) used both to
    convert Kimi's CNY list prices into litellm's USD cost map and to
    convert session costs back to CNY for display."""
    return float(os.environ.get("KIMI_USD_CNY_RATE", "7.10"))


def format_cost(usd: float) -> str:
    """Render a litellm USD session cost in CNY (prices are CNY list prices).
    Uses the fullwidth yen sign (U+FFE5) because the halfwidth one (U+00A5)
    cannot be printed on Windows GBK consoles."""
    return f"￥{usd * _usd_cny_rate():.4f}"


def _register_llm_pricing(model: str) -> None:
    """LiteLLM's price map has no entry for the endpoint's model name, which
    makes every call log "Failed to calculate cost". Kimi's list prices are
    quoted in CNY per million tokens (overridable via env), but litellm's cost
    map is in USD, so register the CNY prices converted at KIMI_USD_CNY_RATE;
    use format_cost() to show session costs back in CNY."""
    # Unmapped models make litellm print a "Provider List" banner on every
    # cost lookup; we register pricing ourselves, so silence it.
    litellm.suppress_debug_info = True
    rate = _usd_cny_rate()
    prices = {
        "mode": "chat",
        "litellm_provider": "openai",
        "max_tokens": 32768,
        "max_input_tokens": 1048576,
        "max_output_tokens": 32768,
        "input_cost_per_token": float(
            os.environ.get("KIMI_INPUT_COST_PER_MTOK", "6.50")
        )
        / rate
        / 1_000_000,
        "output_cost_per_token": float(
            os.environ.get("KIMI_OUTPUT_COST_PER_MTOK", "27.00")
        )
        / rate
        / 1_000_000,
        "cache_read_input_token_cost": float(
            os.environ.get("KIMI_CACHE_READ_COST_PER_MTOK", "1.70")
        )
        / rate
        / 1_000_000,
        "supports_prompt_caching": True,
    }
    litellm.register_model({model: prices})


def kimi_llm_config() -> tuple[str, dict]:
    """Return (model_name, LiteLLM Router model_list config) for Kimi."""
    model = os.environ["KIMI_MODEL"]
    _register_llm_pricing(model)
    config = {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"openai/{model}",
                    "api_base": os.environ["KIMI_API_BASE"],
                    "api_key": os.environ["KIMI_API_KEY"],
                },
            }
        ]
    }
    return model, config


def build_settings() -> Settings:
    model, llm_config = kimi_llm_config()
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
    print(f"=== Cost: {format_cost(session.cost)} ===")


if __name__ == "__main__":
    asyncio.run(main())
