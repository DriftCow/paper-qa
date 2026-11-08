"""Measure document-level retrieval recall without generating an answer.

Example:
    python retrieval_check.py "有哪些文献主张数据质量决定模型效果？" \
        --expect "Doe-2014-An introduction to statistical lea" \
        --phrase '"data quality determines performance"' --phrase '"exact claim wording"'

Each --expect value is a case-insensitive docname substring and represents one
relevant paper. Vector retrieval costs one query embedding; no LLM is called.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from cjk_search import install_cjk_query_patch
from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from paperqa.agents.search import SearchDocumentStorage, SearchIndex
from qa import build_settings
from retrieval_audit import phrase_query

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"


def deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], constant: int = 60
) -> list[str]:
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, key in enumerate(deduplicate(ranking), start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores, key=lambda key: (-scores[key], first_seen[key]))


def expected_ranks(
    ranking: Sequence[str], expected: dict[str, set[str]]
) -> dict[str, int | None]:
    positions = {key: rank for rank, key in enumerate(ranking, start=1)}
    return {
        label: min((positions[key] for key in keys if key in positions), default=None)
        for label, keys in expected.items()
    }


def recall_at_k(
    ranking: Sequence[str], expected: dict[str, set[str]], k: int
) -> float:
    ranks = expected_ranks(ranking[:k], expected)
    return sum(rank is not None for rank in ranks.values()) / max(len(expected), 1)


def resolve_expected(
    patterns: Sequence[str], docs: object
) -> dict[str, set[str]]:
    return {
        pattern: {
            key
            for key, doc in getattr(docs, "docs").items()
            if pattern.casefold() in doc.docname.casefold() or pattern == key
        }
        for pattern in patterns
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare document-level keyword/vector/hybrid retrieval recall."
    )
    parser.add_argument("question", help="Standalone natural-language retrieval query.")
    parser.add_argument(
        "--expect",
        action="append",
        required=True,
        help="Known relevant docname substring or dockey; repeat for multiple papers.",
    )
    parser.add_argument(
        "--phrase",
        action="append",
        default=[],
        help="Additional exact phrase query, including double quotes; repeatable.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default="lit")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--show", type=int, default=5)
    parser.add_argument(
        "--vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include vector retrieval (one query embedding API call).",
    )
    args = parser.parse_args()
    if args.candidate_k <= 0:
        parser.error("--candidate-k must be positive")

    configure_utf8_stdio()
    logging.getLogger("paperqa").setLevel(logging.WARNING)
    settings = build_settings()
    settings.verbosity = 0
    docs, payload = load_docs(args.cache, expected_meta=cache_metadata(settings))
    expected = resolve_expected(args.expect, docs)

    print(f"缓存: {len(docs.docs)} 篇 / {len(docs.texts):,} chunks")
    print(f"缓存更新时间: {payload.get('updated_at', '?')}")
    for label, keys in expected.items():
        if not keys:
            print(f"未入库: {label}")
        else:
            names = sorted(docs.docs[key].docname for key in keys)
            print(f"标注论文: {label} -> {' | '.join(names)}")

    install_cjk_query_patch()
    index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=args.name,
        index_directory=args.index_dir,
        storage=SearchDocumentStorage.PICKLE_COMPRESSED,
    )
    fields = [field for field in index.fields if field != "year"]
    keyword_results = await index.query(
        args.question, top_n=args.candidate_k, field_subset=fields
    )
    keyword = deduplicate(
        next(iter(result.docs.values())).dockey
        for result in keyword_results
        if result.docs
    )
    rankings: dict[str, list[str]] = {"keyword": keyword}
    fusion_inputs = [keyword]

    if args.vector:
        texts = await docs.retrieve_texts(
            args.question,
            k=min(len(docs.texts), args.candidate_k * 4),
            settings=settings,
        )
        vector = deduplicate(text.doc.dockey for text in texts)
        rankings["vector"] = vector
        fusion_inputs.append(vector)

    for phrase in args.phrase:
        phrase_results = await phrase_query(
            index, phrase, top_n=args.candidate_k, field_subset=fields
        )
        ranking = deduplicate(
            next(iter(result.docs.values())).dockey
            for result in phrase_results
            if result.docs
        )
        rankings[f"phrase {phrase}"] = ranking
        fusion_inputs.append(ranking)

    rankings["hybrid"] = reciprocal_rank_fusion(fusion_inputs)
    cutoffs = [k for k in (1, 5, 10, 20, 50, 100) if k <= args.candidate_k]
    print("\n=== 按论文去重的召回率（仅针对 --expect 标注） ===")
    for method, ranking in rankings.items():
        metrics = " ".join(
            f"R@{k}={recall_at_k(ranking, expected, k):.0%}" for k in cutoffs
        )
        print(f"{method}: 候选 {len(ranking)} 篇 | {metrics}")
        ranks = expected_ranks(ranking, expected)
        for label, rank in ranks.items():
            print(f"  {label}: rank={rank if rank is not None else '未召回'}")
        if args.show:
            for rank, key in enumerate(ranking[: args.show], start=1):
                print(f"    {rank}. {docs.docs[key].docname}")

    print("\n说明: 未标注的相关论文不会计入召回率；该结果不等于全库穷尽性。")


if __name__ == "__main__":
    asyncio.run(main())
