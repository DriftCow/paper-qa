r"""One-time (resumable) index builder for a folder of PDFs.

Usage
-----
    cd pqa-local
    .\.venv\Scripts\python.exe build_index.py

The first run parses every PDF, chunks it, and embeds every chunk exactly once.
The result is a single cache file (default ``index/docs.pkl``) that
``ask_index.py`` loads in a few seconds.  Re-running this script only processes
PDFs whose content changed; unchanged files are skipped without touching the
embedding API.

Notes
-----
* Only PDFs are indexed by default.  ``.docx/.xlsx/.pptx`` need the optional
  ``unstructured`` dependency, and ``.doc`` is not supported by PaperQA.
* Citation generation via the LLM is deliberately skipped by passing a
  deterministic citation.  Indexing therefore costs embeddings only.
* Files are added sequentially so that automatic docname de-duplication stays
  deterministic.  Progress is saved every few files and can be resumed.
* Document-specific pypdf failures retry with the optional PyMuPDF reader.
  Fallbacks extracting fewer than 100 characters are rejected for OCR/repair.
* ``--ocr-dir`` points at a mirror of OCR-repaired PDFs (same relative paths as
  the corpus, produced by ocr_repair.py); repaired copies are indexed instead
  of the read-only originals, which are never modified.
* Consecutive-error aborts count service/configuration failures, not bad PDFs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path

from index_cache import (
    cache_metadata,
    configure_utf8_stdio,
    load_docs,
    save_docs,
)
from paperqa import Docs
from qa import build_settings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE_DIR.parent.parent / "PDF"
DEFAULT_CACHE = BASE_DIR / "index" / "docs.pkl"
DEFAULT_REPORT = BASE_DIR / "index" / "build_report.json"


def file_md5(path: Path) -> str:
    h = md5()  # noqa: S324 - content identity, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_pdfs(corpus: Path) -> list[Path]:
    return sorted(
        (p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: str(p).lower(),
    )


def human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def is_document_error(exc: BaseException) -> bool:
    """Whether an exception is specific to one PDF rather than the service."""
    if isinstance(exc, UnicodeDecodeError):
        return True
    if type(exc).__name__ in {
        "DependencyError",
        "ImpossibleParsingError",
        "LimitReachedError",
    }:
        return True
    return isinstance(exc, ValueError) and "This does not look like a text document" in str(
        exc
    )


async def aadd_with_pdf_fallback(docs: Docs, path: Path, settings, **kwargs):
    """Retry a document-specific pypdf failure with PyMuPDF when available."""
    try:
        return await docs.aadd(path, settings=settings, **kwargs)
    except Exception as exc:  # noqa: BLE001 - classified before trying the fallback
        if not is_document_error(exc):
            raise
        try:
            from paperqa_pymupdf import parse_pdf_to_pages
        except ImportError:
            raise exc
        parsed = parse_pdf_to_pages(path, parse_media=False)
        n_chars = sum(
            len(value if isinstance(value, str) else value[0])
            for value in parsed.content.values()
        )
        if n_chars < 100:
            raise ValueError(
                f"PyMuPDF fallback extracted only {n_chars} characters from {path}; "
                "the PDF needs OCR or repair."
            ) from exc
        print(
            f"      primary parser {type(exc).__name__}; "
            f"retrying {n_chars:,} extracted chars with PyMuPDF",
            flush=True,
        )
        fallback_settings = settings.model_copy(deep=True)
        fallback_settings.parsing.parse_pdf = parse_pdf_to_pages
        return await docs.aadd(path, settings=fallback_settings, **kwargs)


def load_report(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt report is not fatal
            pass
    return {"per_file": [], "errors": [], "duplicates": []}


def normalize_report(report: dict) -> dict:
    """Keep one entry per path and drop stale errors for now-indexed files."""
    per_file = {entry["path"]: entry for entry in report.get("per_file", [])}
    added = {path for path, entry in per_file.items() if entry.get("status") == "added"}
    errors = {
        entry["path"]: entry
        for entry in report.get("errors", [])
        if entry["path"] not in added
    }
    report["per_file"] = [per_file[p] for p in sorted(per_file)]
    report["errors"] = [errors[p] for p in sorted(errors)]
    return report


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_report(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_or_create_docs(
    cache: Path, settings, rebuild: bool
) -> tuple[Docs, dict, str | None]:
    if rebuild or not cache.exists():
        return (
            Docs(),
            cache_metadata(settings),
            datetime.now(timezone.utc).isoformat(),
        )
    docs, payload = load_docs(cache, expected_meta=cache_metadata(settings))
    print(
        f"Loaded cache: {len(docs.docs)} docs, {len(docs.texts):,} chunks, "
        f"built {payload.get('updated_at', '?')}",
        flush=True,
    )
    return docs, cache_metadata(settings), payload.get("created_at")


def display_path(path: Path, corpora: list[Path]) -> Path:
    for root in corpora:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a persistent PaperQA Docs cache (parse+embed once)."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[DEFAULT_CORPUS],
        help="One or more directories containing PDFs (default: ../../PDF).",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=None,
        help="Mirror dir of OCR-repaired PDFs (same relative paths as the "
        "corpus, see ocr_repair.py); repaired copies replace originals at "
        "index time without touching the read-only corpus.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--rebuild", action="store_true", help="Ignore any cache.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only add the first N new PDFs."
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save the cache after this many newly indexed PDFs (0 = only at end).",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop cached docs whose PDF is no longer in the corpus (default: on).",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=10,
        help="Abort after this many failures in a row (likely a config problem).",
    )
    parser.add_argument(
        "--max-rate-retries",
        type=int,
        default=4,
        help="Retry a file this many times when the embedding API rate-limits "
        "(HTTP 429); waits ~65s between tries to clear the per-minute window.",
    )
    parser.add_argument(
        "--allow-low-entropy-text",
        action="store_true",
        help=(
            "Disable PaperQA's text-entropy sanity check. Useful for dense CJK "
            "papers whose character entropy exceeds the heuristic's 8.0 limit."
        ),
    )
    args = parser.parse_args()

    configure_utf8_stdio()
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("paperqa").setLevel(logging.WARNING)

    settings = build_settings()
    settings.verbosity = 0
    if args.allow_low_entropy_text:
        settings.parsing.disable_doc_valid_check = True

    corpora = [c.resolve() for c in args.corpus]
    missing = [c for c in corpora if not c.is_dir()]
    if missing:
        sys.exit(f"Corpus directory not found: {', '.join(map(str, missing))}")
    files = sorted(
        {p for c in corpora for p in discover_pdfs(c)}, key=lambda p: str(p).lower()
    )
    override_map: dict[Path, Path] = {}
    if args.ocr_dir:
        swapped = []
        for p in files:
            rel = display_path(p, corpora)
            ov = args.ocr_dir / rel
            if not rel.is_absolute() and ov.is_file():
                override_map[ov] = p
                swapped.append(ov)
            else:
                swapped.append(p)
        files = swapped
        print(f"OCR overrides: {len(override_map)} repaired PDF(s) from {args.ocr_dir}")

    def shown(p: Path) -> Path:
        """Report paths always refer to the original corpus location."""
        return display_path(override_map.get(p, p), corpora)

    if not files:
        sys.exit(f"No PDFs found under {', '.join(map(str, corpora))}")

    docs, meta, created_at = load_or_create_docs(args.cache, settings, args.rebuild)

    t_start = time.perf_counter()
    print("Corpus: " + " | ".join(str(c) for c in corpora))
    print(f"PDFs:   {len(files)}")
    print(f"Cache:  {args.cache}")

    # Content hashes let us skip unchanged files before parsing them.
    md5_by_path = {path: file_md5(path) for path in files}
    path_by_md5: dict[str, Path] = {}
    duplicates: list[dict] = []
    for path, digest in md5_by_path.items():
        if digest in path_by_md5:
            duplicates.append(
                {
                    "kept": str(shown(path_by_md5[digest])),
                    "duplicate": str(shown(path)),
                }
            )
        else:
            path_by_md5[digest] = path

    report = load_report(args.report)
    # Drop stale errors for files no longer in the corpus (e.g. removed bad PDFs).
    current_rel = {str(shown(p)) for p in files}
    report["errors"] = [
        e for e in report.get("errors", []) if e["path"] in current_rel
    ]
    report.update(
        {
            "corpus": [str(c) for c in corpora],
            "cache": str(args.cache),
            "paperqa_version": meta["paperqa_version"],
            "embedding": meta["embedding"],
            "pdfs_in_corpus": len(files),
            "duplicates": duplicates,
            "started_at": report.get("started_at")
            or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_docs": len(docs.docs),
            "n_texts": len(docs.texts),
        }
    )

    if args.prune:
        valid = set(path_by_md5)
        stale = [dockey for dockey in docs.docs if dockey not in valid]
        for dockey in stale:
            docs.delete(dockey=dockey)
        if stale:
            docs.deleted_dockeys.clear()
            print(f"Pruned {len(stale)} stale doc(s) no longer in the corpus.")

    known = set(docs.docs)
    known_files = sum(1 for p in files if md5_by_path[p] in known)
    pending = [p for p in files if md5_by_path[p] not in known]
    new_files = pending if args.limit is None else pending[: args.limit]

    print(f"Already indexed: {known_files}")
    print(
        f"PDFs to index now: {len(new_files)}"
        + (f" (of {len(pending)} pending)" if args.limit is not None else "")
    )
    if not new_files:
        print("Nothing to do; cache is up to date.")
    else:
        consecutive_errors = 0
        indexed_now = 0
        for i, path in enumerate(new_files, start=1):
            rel = shown(path)
            print(f"[{i}/{len(new_files)}] {rel} ...", flush=True)
            before = len(docs.texts)
            t0 = time.perf_counter()
            try:
                docname = None
                for attempt in range(1, args.max_rate_retries + 1):
                    try:
                        docname = await aadd_with_pdf_fallback(
                            docs,
                            path,
                            settings,
                            citation=f"{path.stem} [{path.parent.name}]",
                            docname=path.stem,
                        )
                        break
                    except Exception as e:  # noqa: BLE001
                        rate_limited = (
                            "RateLimit" in type(e).__name__ or "429" in str(e)
                        )
                        if not rate_limited or attempt == args.max_rate_retries:
                            raise
                        print(
                            f"      rate limit (429), waiting 65s "
                            f"(retry {attempt}/{args.max_rate_retries - 1})",
                            flush=True,
                        )
                        await asyncio.sleep(65)
                elapsed = time.perf_counter() - t0
                added = len(docs.texts) - before
                if docname is None:
                    status = "duplicate"
                    print(f"      skipped (already present) {elapsed:.1f}s", flush=True)
                else:
                    status = "added"
                    indexed_now += 1
                    print(
                        f"      +{added} chunks (total {len(docs.texts):,}) "
                        f"in {elapsed:.1f}s",
                        flush=True,
                    )
                report["per_file"].append(
                    {
                        "path": str(rel),
                        "md5": md5_by_path[path],
                        "docname": docname,
                        "chunks_added": added,
                        "seconds": round(elapsed, 2),
                        "status": status,
                    }
                )
                consecutive_errors = 0
            except Exception as e:  # noqa: BLE001 - keep going, report per file
                elapsed = time.perf_counter() - t0
                document_error = is_document_error(e)
                if not document_error:
                    consecutive_errors += 1
                err = {
                    "path": str(rel),
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback_tail": traceback.format_exc()[-1200:],
                    "seconds": round(elapsed, 2),
                }
                report["errors"].append(err)
                print(f"      ERROR {type(e).__name__}: {e}", flush=True)
                if (
                    not document_error
                    and consecutive_errors >= args.max_consecutive_errors
                ):
                    print(
                        f"Aborting after {consecutive_errors} consecutive "
                        "non-document errors. Fix the configuration and re-run; "
                        "progress is saved.",
                        flush=True,
                    )
                    save_docs(docs, args.cache, meta, created_at=created_at)
                    report.update(
                        {
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "n_docs": len(docs.docs),
                            "n_texts": len(docs.texts),
                        }
                    )
                    write_report(args.report, report)
                    raise SystemExit(1) from e

            report.update(
                {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "n_docs": len(docs.docs),
                    "n_texts": len(docs.texts),
                }
            )
            if args.save_every and indexed_now and indexed_now % args.save_every == 0:
                save_docs(docs, args.cache, meta, created_at=created_at)
                write_report(args.report, report)
                print(
                    f"      saved cache ({human_mb(args.cache.stat().st_size)})",
                    flush=True,
                )

    save_docs(docs, args.cache, meta, created_at=created_at)
    report.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "n_docs": len(docs.docs),
            "n_texts": len(docs.texts),
            "cache_bytes": args.cache.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        }
    )
    write_report(args.report, report)

    print("\n=== DONE ===")
    print(f"Docs in cache:   {len(docs.docs):,}")
    print(f"Chunks in cache: {len(docs.texts):,}")
    print(f"Cache size:      {human_mb(args.cache.stat().st_size)}")
    print(f"Errors:          {len(report['errors'])}")
    print(f"Duplicates:      {len(duplicates)}")
    print(f"Report:          {args.report}")
    print("\nQuery with:")
    print(
        '  .\\.venv\\Scripts\\python.exe ask_index.py "your question here"'
    )


if __name__ == "__main__":
    asyncio.run(main())
