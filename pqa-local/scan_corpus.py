"""Read-only corpus scan: page counts, extractable text, duplicate detection.

This does not call any API. It estimates how many chunks paper-qa will create
at the default chunk_chars=5000 / overlap=250 and flags PDFs with little or no
extractable text (likely scanned images), plus exact duplicate files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE_DIR.parent.parent / "PDF"
CHUNK_CHARS = 5000
OVERLAP = 250
WORKERS = 4


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def file_md5(path: Path) -> str:
    h = md5()  # noqa: S324 - content identity, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def display_path(path: Path, roots: tuple[str, ...]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def scan(path_str: str, roots: tuple[str, ...]) -> dict:
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    path = Path(path_str)
    record: dict = {
        "path": display_path(path, roots),
        "bytes": path.stat().st_size,
        "md5": file_md5(path),
    }
    try:
        reader = PdfReader(str(path))
        record["pages"] = len(reader.pages)
        chars = 0
        empty_pages = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            n = len(text.strip())
            chars += n
            if n < 20:
                empty_pages += 1
        record["chars"] = chars
        record["empty_pages"] = empty_pages
        record["chars_per_page"] = (
            round(chars / len(reader.pages), 1) if reader.pages else 0
        )
        record["est_chunks"] = max(
            0, -(-max(chars - OVERLAP, 0) // (CHUNK_CHARS - OVERLAP))
        )
    except Exception as e:  # noqa: BLE001 - report per-file, keep scanning
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def main() -> None:
    configure_stdio()
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Read-only corpus scan (no API calls): pages, chars, "
        "estimated chunks/embedding tokens, duplicates, low-text PDFs."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        default=[DEFAULT_CORPUS],
        help="One or more directories containing PDFs (default: ../../PDF).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "corpus_scan.json",
        help="Where to write the JSON report (default: ./corpus_scan.json).",
    )
    args = parser.parse_args()

    roots = tuple(str(c.resolve()) for c in args.corpus)
    missing = [r for r in roots if not Path(r).is_dir()]
    if missing:
        sys.exit(f"Corpus directory not found: {', '.join(missing)}")

    pdfs = sorted(
        {p for r in roots for p in Path(r).rglob("*.pdf")},
        key=lambda p: str(p).lower(),
    )
    if not pdfs:
        sys.exit(f"No PDFs under {', '.join(roots)}")
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(scan, str(p), roots): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            print(
                f"[{i}/{len(pdfs)}] {rec['path']}: "
                f"{rec.get('pages', '?')}p, {rec.get('chars', 0):,} chars"
                + (f", ERROR {rec['error']}" if "error" in rec else ""),
                flush=True,
            )
    records.sort(key=lambda r: r["path"])

    ok = [r for r in records if "error" not in r]
    total_pages = sum(r["pages"] for r in ok)
    total_chars = sum(r["chars"] for r in ok)
    total_chunks = sum(r["est_chunks"] for r in ok)
    hashes: dict[str, list[str]] = {}
    for r in records:
        hashes.setdefault(r["md5"], []).append(r["path"])
    duplicates = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    low_text = [r for r in ok if r["chars_per_page"] < 200]

    summary = {
        "corpus": list(roots),
        "pdf_count": len(pdfs),
        "parsed_ok": len(ok),
        "errors": len(records) - len(ok),
        "total_pages": total_pages,
        "total_chars": total_chars,
        "est_chunks": total_chunks,
        "est_embedding_tokens": round(total_chars / 1.6),
        "est_embedding_vector_gb_f32_1024d": round(total_chunks * 1024 * 4 / 1e9, 3),
        "duplicate_groups": duplicates,
        "low_text_pdfs": [
            {"path": r["path"], "chars_per_page": r["chars_per_page"]} for r in low_text
        ],
        "records": records,
    }
    out = args.out
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(
        f"PDFs: {summary['pdf_count']} "
        f"(ok {summary['parsed_ok']}, errors {summary['errors']})"
    )
    print(f"Pages: {total_pages:,}")
    print(f"Extractable chars: {total_chars:,}")
    print(f"Estimated chunks @5000/250: {total_chunks:,}")
    print(f"Estimated embedding tokens: {summary['est_embedding_tokens']:,}")
    print(
        "Estimated vectors (1024-d f32): "
        f"{summary['est_embedding_vector_gb_f32_1024d']} GB"
    )
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Low-text PDFs (<200 chars/page): {len(low_text)}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
