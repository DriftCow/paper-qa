"""High-recall, per-paper literature screening without the PaperQA answer agent."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from citation_fix import MetaEntry, load_manifest
from hybrid_retrieval import (
    Candidate,
    DensePaperRetriever,
    HybridRetriever,
    QuerySpec,
    TantivyDocRetriever,
)
from index_cache import cache_metadata, configure_utf8_stdio, load_docs
from paper_review import (
    EvidenceBuilder,
    EvidencePack,
    FindRequest,
    FindResult,
    PaperReviewer,
    PaperVerdict,
    default_llm_judge,
)
from qa import build_settings


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_INDEX_DIR = BASE_DIR / "index_agent"
DEFAULT_BUILD_REPORT = BASE_DIR / "index" / "build_report.json"
DEFAULT_MANIFEST = BASE_DIR / "index" / "citation_manifest.json"
PROMPT_SCHEMA_VERSION = "find-papers-v1"


def _json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _deduplicate_queries(queries: Sequence[QuerySpec]) -> list[QuerySpec]:
    out: list[QuerySpec] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        key = query.channel, query.text.strip()
        if query.text.strip() and key not in seen:
            seen.add(key)
            out.append(query)
    return out


def fallback_query_plan(
    request: FindRequest,
    extra_queries: Sequence[str] = (),
    phrases: Sequence[str] = (),
) -> list[QuerySpec]:
    """Always retain the raw question and explicit user-provided variants."""
    queries = [
        QuerySpec("q000", request.raw_question, "both", "auto", "original")
    ]
    for index, text in enumerate(extra_queries, start=1):
        queries.append(QuerySpec(f"user{index:03d}", text, "both", "auto", "user"))
    extracted = list(phrases)
    extracted += re.findall(r'"([^"\n]+)"|“([^”\n]+)”', request.raw_question)
    flattened = [item if isinstance(item, str) else next(x for x in item if x) for item in extracted]
    for index, text in enumerate(flattened, start=1):
        phrase = text.strip().strip('"“”')
        queries.append(
            QuerySpec(f"phrase{index:03d}", f'"{phrase}"', "phrase", "auto", "explicit")
        )
    return _deduplicate_queries(queries)


async def build_query_plan(
    request: FindRequest,
    extra_queries: Sequence[str] = (),
    phrases: Sequence[str] = (),
    expand: bool = True,
    max_variants: int = 10,
    planner: Any = default_llm_judge,
) -> tuple[list[QuerySpec], list[str]]:
    base = fallback_query_plan(request, extra_queries, phrases)
    if not expand or max_variants <= 0:
        return base, []
    prompt = f"""为文献高召回检索生成学术查询变体。不得虚构论文、作者或已知原文句子。
原问题：{request.raw_question}
生成最多{max_variants}个查询，覆盖中文、英文、同义表达、概念拆解、上下位概念。
保留原始术语，避免过窄的长句。只输出JSON：
{{"queries":[{{"text":"查询文本","language":"zh|en|mixed","family":"synonym|facet|broader|narrower"}}]}}
"""
    try:
        data = await planner(prompt)
        raw_queries = data.get("queries", [])
        if not isinstance(raw_queries, list):
            raise ValueError("Query planner did not return a queries list")
        generated = []
        for index, item in enumerate(raw_queries[:max_variants], start=1):
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            generated.append(
                QuerySpec(
                    f"plan{index:03d}",
                    str(item["text"]).strip(),
                    "both",
                    str(item.get("language", "auto")),
                    str(item.get("family", "expanded")),
                )
            )
        return _deduplicate_queries([*base, *generated]), []
    except Exception as exc:
        return base, [f"degraded_plan: {type(exc).__name__}: {exc}"]


def load_query_plan(path: Path) -> list[QuerySpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("queries", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("Query plan must be a list or contain a queries list")
    queries = [QuerySpec(**item) for item in raw]
    if not queries:
        raise ValueError("Query plan is empty")
    return queries


class Corpus:
    def __init__(
        self,
        docs: Any,
        payload: dict[str, Any],
        index: Any,
        dense: DensePaperRetriever,
        coverage: dict[str, Any],
        manifest: dict[str, MetaEntry],
    ):
        self.docs = docs
        self.payload = payload
        self.index = index
        self.dense = dense
        self.coverage = coverage
        self.manifest = manifest
        self.texts_by_dockey: dict[str, list[Any]] = {}
        for text in docs.texts:
            self.texts_by_dockey.setdefault(text.doc.dockey, []).append(text)


async def load_corpus(
    settings: Any,
    cache_path: Path = DEFAULT_CACHE,
    index_dir: Path = DEFAULT_INDEX_DIR,
    index_name: str = "lit",
    build_report_path: Path = DEFAULT_BUILD_REPORT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> Corpus:
    from paperqa.agents.search import SearchDocumentStorage, SearchIndex

    index_root = index_dir / index_name
    if not (index_root / "index" / "meta.json").exists() or not (
        index_root / "files.zip"
    ).exists():
        raise FileNotFoundError(
            f"Tantivy index is incomplete at {index_root}; run build_agent_index.py first"
        )
    docs, payload = load_docs(cache_path, expected_meta=cache_metadata(settings))
    dense = DensePaperRetriever(docs.texts)
    index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=index_name,
        index_directory=index_dir,
        storage=SearchDocumentStorage.PICKLE_COMPRESSED,
    )
    index_dockeys = set(await index.index_files)
    cache_dockeys = set(docs.docs)
    if index_dockeys != cache_dockeys:
        raise ValueError(
            "Docs cache and Tantivy index differ: "
            f"cache_only={len(cache_dockeys - index_dockeys)}, "
            f"index_only={len(index_dockeys - cache_dockeys)}. "
            "Run build_agent_index.py before screening."
        )
    report: dict[str, Any] = {}
    if build_report_path.exists():
        report = json.loads(build_report_path.read_text(encoding="utf-8"))
    coverage = {
        "pdfs_discovered": report.get("pdfs_in_corpus"),
        "indexed_papers": len(docs.docs),
        "chunks": len(docs.texts),
        "parse_failures": len(report.get("errors", [])),
        "duplicate_files": len(report.get("duplicates", [])),
        "cache_updated_at": payload.get("updated_at"),
    }
    return Corpus(docs, payload, index, dense, coverage, load_manifest(manifest_path))


def _review_key(request: FindRequest, dockey: str, pack: EvidencePack) -> str:
    raw = json.dumps(
        {
            "question": request.raw_question,
            "proposition": request.proposition,
            "dockey": dockey,
            "evidence": [span.span_id for span in pack.spans],
            "schema": PROMPT_SCHEMA_VERSION,
            "model": os.environ.get("KIMI_MODEL", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[str, tuple[str, PaperVerdict]]:
    out: dict[str, tuple[str, PaperVerdict]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            verdict = PaperVerdict(**row["verdict"])
            if verdict.review_status == "completed":
                out[verdict.dockey] = row["cache_key"], verdict
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return out


def _meta(docname: str, doc: Any, manifest: dict[str, MetaEntry]) -> dict[str, str]:
    entry = manifest.get(docname)
    return {
        "title": (entry.title if entry and entry.title else getattr(doc, "title", "") or docname),
        "authors": ", ".join(entry.authors) if entry else "",
        "year": entry.year or "" if entry else "",
        "doi": entry.doi or "" if entry else "",
        "source_path": entry.source_path or "" if entry else "",
    }


def _span_lookup(packs: dict[str, EvidencePack]) -> dict[str, Any]:
    return {span.span_id: span for pack in packs.values() for span in pack.spans}


def _hit_summary(candidate: Candidate) -> str:
    return "; ".join(
        f"{hit.query_id}:{hit.channel}#{hit.rank}"
        for hit in sorted(candidate.hits, key=lambda h: (h.rank, h.query_id, h.channel))
    )


def _paper_block(
    verdict: PaperVerdict,
    candidate: Candidate,
    meta: dict[str, str],
    spans: dict[str, Any],
) -> str:
    evidence = []
    for span_id in verdict.evidence_ids:
        span = spans.get(span_id)
        if span:
            evidence.append(f"{span.chunk_name}（PDF第{span.pages}页）")
    lines = [
        f"### {meta['title']}",
        f"- 文件：`{verdict.docname}`；dockey：`{verdict.dockey}`",
        f"- 核心度：**{verdict.centrality}**；立场：**{verdict.stance}**；置信度：{verdict.confidence:.2f}",
        f"- 判定理由：{verdict.reason or '未提供'}",
        f"- 命中来源：{_hit_summary(candidate) or 'strict全库补入'}",
        f"- 证据位置：{'；'.join(dict.fromkeys(evidence)) or '待补充'}",
    ]
    if verdict.quote and verdict.quote_status == "verified":
        lines.append(f"- 原文摘录：\"{verdict.quote}\"")
    if verdict.error:
        lines.append(f"- 错误：`{verdict.error}`")
    return "\n".join(lines)


def write_outputs(
    result: FindResult,
    output_dir: Path,
    docs: Any,
    manifest: dict[str, MetaEntry],
    packs: dict[str, EvidencePack],
    review_keys: dict[str, str] | None = None,
) -> None:
    import csv

    result.validate_accounting()
    output_dir.mkdir(parents=True, exist_ok=True)
    spans = _span_lookup(packs)
    groups: dict[str, list[str]] = {"support": [], "oppose": [], "other": []}
    background: list[str] = []
    pending: list[str] = []
    rows: list[dict[str, Any]] = []
    for candidate in result.candidates:
        verdict = result.verdicts[candidate.dockey]
        doc = docs.docs[candidate.dockey]
        meta = _meta(verdict.docname, doc, manifest)
        block = _paper_block(verdict, candidate, meta, spans)
        if verdict.review_status == "completed" and verdict.centrality == "core":
            group = verdict.stance if verdict.stance in {"support", "oppose"} else "other"
            groups[group].append(block)
        elif verdict.review_status == "completed" and verdict.centrality == "background":
            background.append(block)
        elif verdict.review_status != "completed" or verdict.centrality == "uncertain":
            pending.append(block)
        quote_span = spans.get(verdict.quote_span_id)
        rows.append(
            {
                "title": meta["title"],
                "docname": verdict.docname,
                "dockey": verdict.dockey,
                "authors": meta["authors"],
                "year": meta["year"],
                "doi": meta["doi"],
                "source_path": meta["source_path"],
                "centrality": verdict.centrality,
                "stance": verdict.stance,
                "review_status": verdict.review_status,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
                "evidence_ids": " ".join(verdict.evidence_ids),
                "pages": quote_span.pages if quote_span else "",
                "quote": verdict.quote,
                "quote_status": verdict.quote_status,
                "hit_by": _hit_summary(candidate),
                "exclusion_code": verdict.exclusion_code,
                "error": verdict.error,
            }
        )
    main = ["# 核心相关论文", "", f"问题：{result.request.raw_question}"]
    for key, title in (
        ("support", "支持"),
        ("oppose", "反对"),
        ("other", "中性、混合或立场不确定"),
    ):
        main.extend(["", f"## {title}（{len(groups[key])}篇）", ""])
        main.append("\n\n".join(groups[key]) if groups[key] else "无")
    (output_dir / "main.md").write_text("\n".join(main) + "\n", encoding="utf-8")
    (output_dir / "background.md").write_text(
        f"# 背景相关附录（{len(background)}篇）\n\n" + ("\n\n".join(background) or "无") + "\n",
        encoding="utf-8",
    )
    (output_dir / "pending.md").write_text(
        f"# 待复核与失败项（{len(pending)}篇）\n\n" + ("\n\n".join(pending) or "无") + "\n",
        encoding="utf-8",
    )
    with (output_dir / "papers.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dockey"])
        writer.writeheader()
        writer.writerows(rows)
    _json_dump(
        output_dir / "evidence.json",
        {key: asdict(pack) for key, pack in packs.items()},
    )
    _json_dump(output_dir / "corpus_dockeys.json", sorted(docs.docs))
    payload = asdict(result)
    payload["complete"] = result.is_complete
    payload["counts"] = result.counts
    _json_dump(output_dir / "result.json", payload)
    _json_dump(
        output_dir / "run_manifest.json",
        {
            "schema": PROMPT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": result.request.raw_question,
            "complete": result.is_complete,
            "counts": result.counts,
            "coverage": result.coverage,
            "errors": result.errors,
        },
    )
    with (output_dir / "verdicts.jsonl").open("w", encoding="utf-8") as handle:
        for dockey, verdict in result.verdicts.items():
            key = (review_keys or {}).get(dockey, "")
            handle.write(
                json.dumps(
                    {"cache_key": key, "verdict": asdict(verdict)}, ensure_ascii=False
                )
                + "\n"
            )


async def run(args: argparse.Namespace) -> FindResult | None:
    settings = build_settings()
    settings.verbosity = 0
    corpus = await load_corpus(
        settings,
        args.cache,
        args.index_dir,
        args.name,
        args.build_report,
        args.manifest,
    )
    print(
        f"语料预检通过：{len(corpus.docs.docs)} 篇 / {len(corpus.docs.texts):,} chunks；"
        f"解析失败 {corpus.coverage['parse_failures']} 个文件"
    )
    if args.preflight_only:
        return None
    proposition = None if args.topic_only else (args.proposition or args.question)
    request = FindRequest(args.question, proposition)
    errors: list[str] = []
    if args.resume and not args.query_plan:
        previous = json.loads((args.resume / "result.json").read_text(encoding="utf-8"))
        if previous["request"]["raw_question"] != args.question:
            raise ValueError("Resume question does not match the previous run")
        queries = [QuerySpec(**item) for item in previous["queries"]]
    elif args.query_plan:
        queries = load_query_plan(args.query_plan)
    else:
        queries, errors = await build_query_plan(
            request,
            args.query,
            args.phrase,
            expand=not args.no_query_expansion,
            max_variants=args.max_variants,
        )
    queries = _deduplicate_queries(
        [QuerySpec("q000", request.raw_question, "both", "auto", "original"), *queries]
    )
    if args.save_query_plan:
        _json_dump(args.save_query_plan, {"queries": [asdict(q) for q in queries]})
    hybrid = HybridRetriever(corpus.dense, TantivyDocRetriever(corpus.index))
    retrieval = await hybrid.search(
        queries, settings.get_embedding_model(), depth=args.depth
    )
    errors.extend(retrieval.errors)
    candidates = retrieval.candidates
    if args.strict_corpus:
        present = {candidate.dockey for candidate in candidates}
        candidates.extend(
            Candidate(dockey, best_rank=0)
            for dockey in sorted(set(corpus.docs.docs) - present)
        )
    builder = EvidenceBuilder(corpus.texts_by_dockey, corpus.dense.chunk_by_id)
    packs = {candidate.dockey: builder.build(candidate) for candidate in candidates}
    expanded = {
        candidate.dockey: builder.build(candidate, expanded=True)
        for candidate in candidates
    }
    all_packs = dict(packs)
    for dockey, pack in expanded.items():
        all_packs[f"{dockey}:expanded"] = pack
    output_dir = args.resume or (args.output_dir / (args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")))
    checkpoint_path = output_dir / "verdicts.jsonl"
    cached = load_checkpoint(checkpoint_path) if args.resume else {}
    keys = {
        candidate.dockey: _review_key(request, candidate.dockey, packs[candidate.dockey])
        for candidate in candidates
    }
    verdicts = {
        candidate.dockey: cached[candidate.dockey][1]
        for candidate in candidates
        if candidate.dockey in cached and cached[candidate.dockey][0] == keys[candidate.dockey]
    }
    remaining = [candidate for candidate in candidates if candidate.dockey not in verdicts]
    if args.retrieve_only:
        for candidate in remaining:
            verdicts[candidate.dockey] = PaperVerdict(
                candidate.dockey,
                packs[candidate.dockey].docname,
                reason="retrieve-only：尚未进行论文复核",
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        def checkpoint(verdict: PaperVerdict) -> None:
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"cache_key": keys[verdict.dockey], "verdict": asdict(verdict)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        verdicts.update(
            await PaperReviewer().review_many(
                request,
                remaining,
                packs,
                expanded,
                concurrency=args.concurrency,
                max_review=args.max_review,
                on_verdict=checkpoint,
            )
        )
    coverage = {
        **corpus.coverage,
        "strict_corpus": args.strict_corpus,
        "retrieval_depth": args.depth,
        "retrieve_only": args.retrieve_only,
        "llm_model": os.environ.get("KIMI_MODEL", ""),
        "embedding_model": corpus.payload.get("embedding", ""),
        "prompt_schema": PROMPT_SCHEMA_VERSION,
    }
    result = FindResult(
        request,
        queries,
        candidates,
        {candidate.dockey: verdicts[candidate.dockey] for candidate in candidates},
        coverage,
        errors,
    )
    write_outputs(result, output_dir, corpus.docs, corpus.manifest, all_packs, keys)
    print(f"结果目录：{output_dir}")
    print(f"统计：{result.counts}；完整运行：{result.is_complete}")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--proposition", help="待判断命题；默认使用原问题")
    parser.add_argument("--topic-only", action="store_true", help="纯主题检索，不判断支持/反对")
    parser.add_argument("--query", action="append", default=[], help="额外中英文查询，可重复")
    parser.add_argument("--phrase", action="append", default=[], help="精确短语，可重复")
    parser.add_argument("--query-plan", type=Path, help="冻结的查询计划JSON")
    parser.add_argument("--save-query-plan", type=Path)
    parser.add_argument("--no-query-expansion", action="store_true")
    parser.add_argument("--max-variants", type=int, default=10)
    parser.add_argument("--depth", type=int, default=80, help="每路每查询的论文深度")
    parser.add_argument("--max-review", type=int, help="显式复核预算；其余论文进入pending")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--strict-corpus", action="store_true", help="全部已索引论文至少复核一次")
    parser.add_argument("--retrieve-only", action="store_true", help="只生成候选，不调用复核LLM")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", type=Path, help="已有运行目录，复用相同证据的完成项")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "reviews")
    parser.add_argument("--run-id")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--name", default="lit")
    parser.add_argument("--build-report", type=Path, default=DEFAULT_BUILD_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main() -> None:
    configure_utf8_stdio()
    args = make_parser().parse_args()
    if args.depth <= 0 or args.concurrency <= 0:
        raise SystemExit("--depth and --concurrency must be positive")
    if args.resume and not (args.resume / "result.json").exists():
        raise SystemExit("--resume directory must contain result.json")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
