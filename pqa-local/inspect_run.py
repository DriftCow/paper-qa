r"""Bounded viewer for find_papers run outputs (reviews/<run-id>/).

Agents must use this tool instead of reading result.json/evidence.json
directly or writing ad-hoc Python to parse them: those files contain
megabytes of raw PDF text and will blow up the agent context. Every mode
here hard-caps its output so a single turn can never flood the context.

Usage
-----
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --meta
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --list 60
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --paper <dockey>
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --paper <dockey1> <dockey2> <dockey3>
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --span <span_id> --chars 800
    .venv\\Scripts\\python.exe inspect_run.py reviews\\<run-id> --grep "pattern"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from index_cache import configure_utf8_stdio

LIST_DEFAULT = 60
LIST_MAX = 300
TITLE_WIDTH = 60
PAPER_CHARS_DEFAULT = 400
PAPER_CHARS_MAX = 1200
PAPER_BATCH_CHARS_DEFAULT = 240
PAPER_BATCH_CHARS_MAX = 800
PAPER_BATCH_SPANS_PER_PAPER = 4
PAPER_BATCH_SPANS_TOTAL = 24
SPAN_CHARS_DEFAULT = 800
SPAN_CHARS_MAX = 2400
GREP_WINDOW = 80
GREP_SNIPPET_MAX = 280
GREP_MAX_MATCHES = 20


def _clip(text: str, limit: int) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[:limit] + f"…[截断，原文共{len(flattened)}字符，用 --span <span_id> 取更多]"


def _short(text: str, limit: int) -> str:
    """Metadata-only clip: plain ellipsis, no span hint."""
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1] + "…"


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict], dict[str, dict]]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    evidence = json.loads((run_dir / "evidence.json").read_text(encoding="utf-8"))
    meta: dict[str, dict] = {}
    with (run_dir / "papers.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            meta[row["dockey"]] = row
    spans: dict[str, dict] = {}
    for dockey, pack in evidence.items():
        for span in pack.get("spans", []):
            spans[span["span_id"]] = {**span, "dockey": dockey.removesuffix(":expanded")}
    return result, evidence, meta, spans


def show_meta(result: dict[str, Any]) -> None:
    request = result["request"]
    counts = result.get("counts", {})
    print(f"问题：{request.get('raw_question', '')}")
    if request.get("proposition"):
        print(f"命题：{request['proposition']}")
    print(f"查询数：{len(result.get('queries', []))}；候选数：{len(result.get('candidates', []))}")
    print(f"完整运行：{result.get('complete')}")
    if counts:
        print(f"counts（--retrieve-only 下为占位值）：{counts}")
    coverage = result.get("coverage", {})
    if coverage:
        print(f"coverage：indexed={coverage.get('indexed_papers')} chunks={coverage.get('chunks')} depth={coverage.get('retrieval_depth')} strict={coverage.get('strict_corpus')}")
    errors = result.get("errors", [])
    if errors:
        print(f"errors（{len(errors)} 条）:")
        for item in errors[:10]:
            print(f"  - {item}")
    else:
        print("errors：无")


def show_list(
    result: dict[str, Any],
    evidence: dict[str, Any],
    meta: dict[str, dict],
    count: int,
) -> None:
    count = max(1, min(count, LIST_MAX))
    candidates = sorted(
        result.get("candidates", []),
        key=lambda c: (-float(c.get("rrf_score", 0.0)), str(c.get("dockey"))),
    )
    total = len(candidates)
    shown = candidates[:count]
    print(f"候选总数 {total}；按 rrf_score 降序显示前 {len(shown)} 篇（提高 --list 上限最多 {LIST_MAX}）")
    print("rank | rrf_score | best_rank | dockey | 年份 | spans | 页码 | 题名")
    for index, cand in enumerate(shown, start=1):
        dockey = cand["dockey"]
        row = meta.get(dockey, {})
        pack = evidence.get(dockey, {})
        pages = ";".join(
            dict.fromkeys(str(s.get("pages", "")) for s in pack.get("spans", []))
        )
        title = row.get("title") or pack.get("docname", "") or dockey
        print(
            f"{index:4d} | {float(cand.get('rrf_score', 0.0)):7.3f} | "
            f"{int(cand.get('best_rank', 0)):3d} | {dockey} | {row.get('year', '') or '?':4s} | "
            f"{len(pack.get('spans', []))} | {_short(pages, 20):20s} | {_short(title, TITLE_WIDTH)}"
        )


def show_paper(
    dockeys: list[str],
    evidence: dict[str, Any],
    meta: dict[str, dict],
    chars: int | None,
) -> None:
    batch = len(dockeys) > 1
    if batch:
        chars = max(80, min(chars if chars is not None else PAPER_BATCH_CHARS_DEFAULT, PAPER_BATCH_CHARS_MAX))
        print(
            f"批量模式：{len(dockeys)} 篇｜每片段截断 {chars} 字符"
            f"｜每篇最多 {PAPER_BATCH_SPANS_PER_PAPER} 片段、全程最多 {PAPER_BATCH_SPANS_TOTAL} 片段"
        )
    else:
        chars = max(80, min(chars if chars is not None else PAPER_CHARS_DEFAULT, PAPER_CHARS_MAX))
    shown_spans = 0
    for index, dockey in enumerate(dockeys, start=1):
        pack = evidence.get(dockey)
        if pack is None:
            message = f"No evidence pack for dockey {dockey!r} (use --list first)"
            if batch:
                print(f"\n### [{index}/{len(dockeys)}] {dockey}：{message}")
                continue
            raise SystemExit(message)
        row = meta.get(dockey, {})
        title = row.get("title") or pack.get("docname", "") or dockey
        if batch:
            print(f"\n### [{index}/{len(dockeys)}] {dockey}")
            print(f"题名：{_short(title, TITLE_WIDTH * 2)}")
            print(f"作者：{_short(row.get('authors', '') or '?', 80)}；年份：{row.get('year', '') or '?'}")
        else:
            print(f"题名：{title}")
            print(f"作者：{row.get('authors', '') or '?'}；年份：{row.get('year', '') or '?'}；DOI：{row.get('doi', '') or '?'}")
            print(f"路径：{row.get('source_path', '') or '?'}")
        expanded = evidence.get(f"{dockey}:expanded", {})
        spans = pack.get("spans", [])
        print(
            f"证据片段 {len(spans)} 个"
            f"（另有扩展片段 {len(expanded.get('spans', []))} 个，需逐个 --span 查看）"
        )
        if batch:
            spans = spans[:PAPER_BATCH_SPANS_PER_PAPER]
        for span in spans:
            if batch:
                if shown_spans >= PAPER_BATCH_SPANS_TOTAL:
                    print("…批量片段总数已达上限，剩余论文请拆成更小批次")
                    return
                shown_spans += 1
            print(
                f"\n[{span['span_id']}] {span['chunk_name']}（PDF第{span['pages']}页）"
                f"｜{span.get('section', '')}｜{span.get('own_work_role', '')}"
            )
            print(_clip(span.get("text", ""), chars))


def show_span(span_id: str, spans: dict[str, dict], chars: int) -> None:
    span = spans.get(span_id)
    if span is None:
        raise SystemExit(f"No span with id {span_id!r}")
    chars = max(80, min(chars, SPAN_CHARS_MAX))
    print(f"{span.get('dockey', '?')}｜[{span_id}] {span['chunk_name']}（PDF第{span['pages']}页）｜{span.get('section', '')}")
    print(_clip(span.get("text", ""), chars))


def show_grep(pattern: str, spans: dict[str, dict]) -> None:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise SystemExit(f"Invalid regex: {exc}")
    matches = 0
    for span_id, span in spans.items():
        text = " ".join(span.get("text", "").split())
        found = regex.search(text)
        if not found:
            continue
        matches += 1
        if matches > GREP_MAX_MATCHES:
            print(f"…命中片段超过 {GREP_MAX_MATCHES} 个，已截断；请改用更窄的模式")
            return
        lo = max(0, found.start() - GREP_WINDOW)
        hi = min(len(text), found.end() + GREP_WINDOW)
        snippet = text[lo:hi]
        if len(snippet) > GREP_SNIPPET_MAX:
            snippet = snippet[:GREP_SNIPPET_MAX] + "…"
        print(f"{span.get('dockey', '?')}｜[{span_id}] {span['chunk_name']}（PDF第{span['pages']}页）")
        print(f"  …{snippet}…")
    print(f"\n命中片段共 {matches} 个")


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="find_papers 输出目录，如 reviews\\<run-id>")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--meta", action="store_true", help="运行概览：问题、候选数、counts、errors")
    group.add_argument("--list", type=int, nargs="?", const=LIST_DEFAULT, default=None,
                       metavar="N", help=f"按 rrf_score 降序列出前 N 篇候选（默认 {LIST_DEFAULT}，上限 {LIST_MAX}）")
    group.add_argument("--paper", nargs="+", metavar="DOCKEY",
                       help="论文摘要：1 篇 = 完整视图；多篇 = 批量模式（每篇最多 4 片段、每片段默认 240 字符）")
    group.add_argument("--span", metavar="SPAN_ID", help="单个证据片段的较长文本")
    group.add_argument("--grep", nargs="+", metavar="PATTERN",
                       help="正则检索证据片段（多词直接跟在后面即可），仅返回带窗口的小片段")
    parser.add_argument("--chars", type=int, default=None,
                        help="文本截断长度：--paper 单篇默认 400/上限 1200、批量默认 240/上限 800；--span 默认 800/上限 2400")
    args = parser.parse_args()

    if not (args.run_dir / "result.json").exists():
        raise SystemExit(f"{args.run_dir} does not look like a find_papers output directory")

    result, evidence, meta, spans = load_run(args.run_dir)
    if args.meta:
        show_meta(result)
    elif args.list is not None:
        show_list(result, evidence, meta, args.list)
    elif args.paper:
        show_paper(args.paper, evidence, meta, args.chars)
    elif args.span:
        show_span(args.span, spans, args.chars or SPAN_CHARS_DEFAULT)
    elif args.grep:
        show_grep(" ".join(args.grep), spans)


if __name__ == "__main__":
    main()
