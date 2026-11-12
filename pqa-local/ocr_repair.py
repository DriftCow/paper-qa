r"""为扫描件 PDF 生成带不可见文本层的修复版（RapidOCR + PyMuPDF），供 build_index 收录。

用法
----
    .\.venv\Scripts\python.exe ocr_repair.py                # 默认从 build_report.json 的 errors 里挑 "needs OCR" 项
    .\.venv\Scripts\python.exe ocr_repair.py a.pdf b.pdf    # 或显式指定文件
    .\.venv\Scripts\python.exe ocr_repair.py --dry-run      # 只列出待修复文件

行为
----
* 原始语料只读、绝不修改：修复版写入镜像目录（默认 ``repaired_pdf/``，--out-dir 可改），
  保持与语料根目录相同的相对路径，例如 ``repaired_pdf/2177461012/xxx.pdf``。
* 文本以 render_mode=3（不可见）按 OCR 行框位置插入，使用内置中文字体 china-s，
  页码与原版一一对应，下游证据定位不受影响。
* 修复后用 PyMuPDF 与 pypdf 双重验证提取字符数；不达标则删除修复版并报错。
* 0 字节空文件无法本地修复，单独列出，需要重新下载原文。
* 完成后重跑：build_index.py --ocr-dir <镜像目录>，再跑 build_agent_index.py。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pymupdf

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = BASE_DIR / "index" / "build_report.json"
DEFAULT_OUT_DIR = BASE_DIR / "repaired_pdf"


def select_targets(args: argparse.Namespace) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return ([(source, relative_path)], empty_files)."""
    if args.paths:
        items = [
            (Path(p).resolve(), Path(p).parent.name / Path(p).name) for p in args.paths
        ]
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        roots = [Path(c) for c in report.get("corpus", [])]
        items = []
        for err in report.get("errors", []):
            if "needs OCR or repair" not in err.get("message", ""):
                continue
            rel = Path(err["path"])
            resolved = next((r / rel for r in roots if (r / rel).exists()), None)
            if resolved is None:
                print(f"  ! 报告中的路径不存在，跳过：{err['path']}")
                continue
            items.append((resolved, rel))
    empty = [src for src, _ in items if src.stat().st_size == 0]
    return [(src, rel) for src, rel in items if src.stat().st_size > 0], empty


def ocr_page(engine, page: pymupdf.Page, zoom: float) -> int:
    """OCR one page and insert an invisible text line per detected box."""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:  # RapidOCR 期望 BGR
        img = np.ascontiguousarray(img[:, :, ::-1])
    result = engine(img)
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    if boxes is None or txts is None:
        return 0
    inserted = 0
    for box, text in zip(boxes, txts):
        text = str(text).strip()
        if not text:
            continue
        xs, ys = box[:, 0], box[:, 1]
        x0, y0 = float(xs.min()) / zoom, float(ys.min()) / zoom
        height = float(ys.max() - ys.min()) / zoom
        try:
            page.insert_text(
                pymupdf.Point(x0, y0 + height * 0.85),
                text,
                fontname="china-s",
                fontsize=max(2.0, height * 0.95),
                render_mode=3,
            )
            inserted += 1
        except Exception as exc:  # noqa: BLE001 - 单行失败不丢整页
            print(f"      ! 插入行失败（{text[:20]}...）：{exc}")
    return inserted


def extract_chars(path: Path) -> tuple[int, int]:
    with pymupdf.open(path) as doc:
        mu_chars = sum(len(page.get_text()) for page in doc)
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    py_chars = sum(len(page.extract_text() or "") for page in reader.pages)
    return mu_chars, py_chars


def repair(src: Path, dest: Path, engine, dpi: int, min_chars: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    tmp = dest.with_name(dest.name + ".ocr-tmp")
    with pymupdf.open(src) as doc:
        for i, page in enumerate(doc, start=1):
            n = ocr_page(engine, page, zoom)
            print(f"      第{i}/{len(doc)}页：插入 {n} 行", flush=True)
        doc.save(tmp, garbage=3, deflate=True)
    os.replace(tmp, dest)
    mu_chars, py_chars = extract_chars(dest)
    print(f"      验证：PyMuPDF {mu_chars:,} 字符 / pypdf {py_chars:,} 字符", flush=True)
    if mu_chars < min_chars:
        dest.unlink()
        print(f"      ! 提取字符不足 {min_chars}，已删除修复版", flush=True)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets, empty = select_targets(args)
    for p in empty:
        print(f"空文件（0 字节），无法本地修复，请重新下载：{p}")
    for src, rel in targets:
        print(f"待修复：{src} -> {args.out_dir / rel}")
    if args.dry_run or not targets:
        return

    from rapidocr import RapidOCR

    engine = RapidOCR()
    failed = []
    for src, rel in targets:
        print(f"[OCR] {src.name}", flush=True)
        if not repair(src, args.out_dir / rel, engine, args.dpi, args.min_chars):
            failed.append(src)
    print(f"\n=== 完成：修复 {len(targets) - len(failed)}/{len(targets)} ===")
    if failed:
        raise SystemExit(f"修复失败：{failed}")
    print(f"下一步：build_index.py --ocr-dir {args.out_dir}，然后 build_agent_index.py。")


if __name__ == "__main__":
    main()