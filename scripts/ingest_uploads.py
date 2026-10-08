# -*- coding: utf-8 -*-
"""
批量导入 data/uploads 下的论文 PDF 到知识库（命令行，绕过 Streamlit 界面）。

为什么需要这个脚本：
    app.upload_dir 只是「界面上传文件的落盘目录」，系统启动时**不会**自动扫描它。
    直接把 PDF 丢进 data/uploads/ 不会被发现，必须显式调用 ingest 才会
    解析 -> 分块 -> 向量化 -> 写入 SQLite，之后 list_documents() 才能看到。

用法：
    python scripts/ingest_uploads.py                     # 导入全部 PDF，chunk=256
    python scripts/ingest_uploads.py --chunk-size 512     # 改用配置的默认 512
    python scripts/ingest_uploads.py --dir ./data/uploads
    python scripts/ingest_uploads.py --dry-run            # 只列文件，不导入

注意 chunk_size 口径：
    config.yaml 默认 512，但交付书 §4.1 的评测数据（46 块）来自 256。
    为复现评测结果，本脚本默认显式覆盖为 256。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.system import ResearchAssistantSystem  # noqa: E402
from src.utils.config import Config  # noqa: E402

SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


def main() -> int:
    ap = argparse.ArgumentParser(description="批量导入 data/uploads 下的论文")
    ap.add_argument("--dir", default=str(ROOT / "data" / "uploads"),
                    help="待导入目录，默认 data/uploads")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="分块大小，默认 256（与交付书评测口径一致）")
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--dry-run", action="store_true", help="只列文件，不实际导入")
    ap.add_argument("--reset", action="store_true",
                    help="导入前先清空知识库（清掉历史测试数据，保证库里只有这批论文）")
    args = ap.parse_args()

    src = Path(args.dir)
    if not src.exists():
        print(f"[ERROR] 目录不存在: {src}")
        return 1

    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in src.glob(pattern)
                   if p.is_file() and p.suffix.lower() in SUFFIXES)

    if not files:
        print(f"[WARN] {src} 下没有可导入的文件（{sorted(SUFFIXES)}）")
        return 1

    total_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
    print(f"待导入 {len(files)} 个文件，共 {total_mb:.1f} MB")
    for p in files:
        print(f"  - {p.name}")
    if args.dry_run:
        print("\n[dry-run] 未执行导入")
        return 0

    cfg = Config.load(str(ROOT / "config.yaml"), reload=True)
    cfg.set("chunking.chunk_size", args.chunk_size)
    print(f"\nchunk_size = {args.chunk_size}")

    print("初始化系统 ...")
    system = ResearchAssistantSystem(config=cfg)

    if args.reset:
        before = len(system.list_documents())
        system.clear_knowledge_base()
        print(f"已清空知识库（原有 {before} 篇文档）\n")

    t0 = time.time()
    done = {"n": 0}

    def on_progress(cur: int, total: int, name: str, status: str) -> None:
        if status != "ok":
            print(f"  ({cur}/{total}) {name} -> {status}")
        done["n"] = cur

    print("开始导入 ...\n")
    summary = system.ingest_files([str(p) for p in files],
                                  on_progress=on_progress)

    print("\n" + "=" * 56)
    print("导入结果")
    print("=" * 56)
    for k in ("total", "success", "failed", "skipped"):
        if k in summary:
            print(f"  {k:10s}: {summary[k]}")
    print(f"  chunks    : {summary.get('total_chunks', system.retriever.size)}")
    print(f"  耗时      : {time.time() - t0:.1f}s")

    docs = system.list_documents()
    print(f"\n知识库现有 {len(docs)} 篇文档：")
    for d in docs:
        print(f"  - {d.get('title', d.get('id'))} "
              f"(chunks={d.get('chunk_count', '?')})")

    if summary.get("failed"):
        print("\n失败明细：")
        for item in summary.get("results", []):
            if item.get("status") != "ok":
                print(f"  {item}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
