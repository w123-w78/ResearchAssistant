# -*- coding: utf-8 -*-
"""
侧边栏组件：知识库管理（上传 / 列表 / 删除 / 统计）+ 会话 + 系统状态
"""
from __future__ import annotations

import time
from html import escape
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from .style import section


def render_sidebar(system) -> None:
    """渲染左侧知识库面板。"""
    # ---------- 上传 ----------
    section("知识库", "📚")
    uploaded = st.file_uploader(
        "上传论文（PDF / Word / Markdown / TXT）",
        type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
        help="支持批量上传；同名同大小的文件会自动跳过",
    )
    if uploaded:
        paths = _save_uploads(uploaded, system)
        c1, c2 = st.columns([0.62, 0.38])
        with c1:
            if st.button(f"开始导入 ({len(paths)})", type="primary",
                         use_container_width=True):
                _do_ingest(system, paths)
        with c2:
            if st.button("清空选择", use_container_width=True):
                st.rerun()

    st.divider()

    # ---------- 文档列表 ----------
    docs = system.list_documents()
    section(f"已收录文档 · {len(docs)}", "📄")
    if not docs:
        st.caption("暂无文档，请先上传论文")
    for d in docs:
        card, btn = st.columns([0.84, 0.16])
        with card:
            st.markdown(
                '<div class="ra-doc">'
                f'<div class="t">{escape(d["title"][:40])}</div>'
                f'<div class="m">{d.get("num_chunks", 0)} 块 · '
                f'{escape(str(d.get("file_type", "")))} · '
                f'{escape(str(d.get("status", "")))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        with btn:
            st.button("🗑", key=f"del_{d['doc_id']}", help="删除该文档",
                      on_click=_delete_doc, args=(system, d["doc_id"]))

    st.divider()

    # ---------- 会话管理 ----------
    section("会话", "💬")
    if st.button("＋ 新建会话", use_container_width=True):
        mem = system.memory.create_session()
        st.session_state.session_id = mem.session_id
        st.session_state.messages = []
        st.session_state.last_steps = []
        st.session_state.last_meta = {}
        st.rerun()

    sessions = system.memory.list_sessions(limit=20)
    if not sessions:
        st.caption("暂无历史会话")
    for s in sessions:
        active = s["session_id"] == st.session_state.get("session_id")
        if st.button(
            f"{'▶ ' if active else ''}{s['title'][:18]}",
            key=f"sess_{s['session_id']}",
            use_container_width=True,
            type="primary" if active else "secondary",
        ):
            st.session_state.session_id = s["session_id"]
            st.session_state.messages = _load_history(system, s["session_id"])
            st.session_state.last_steps = []
            st.session_state.last_meta = {}
            st.rerun()

    st.divider()

    # ---------- 系统状态 ----------
    _render_health(system)


# ------------------------------------------------------------------ 内部 ----
def _do_ingest(system, paths: List[str]) -> None:
    """执行导入并展示进度。"""
    bar = st.progress(0.0)
    status = st.empty()

    def on_progress(cur, total, path, state):
        bar.progress(cur / total)
        icon = {"ok": "✓", "fail": "✗", "skip": "⊘"}.get(state, "·")
        status.caption(f"{icon} [{cur}/{total}] {path.split('/')[-1]}")

    with st.spinner("正在解析与向量化..."):
        summary = system.ingest_files(paths, on_progress=on_progress)
    bar.empty()
    status.empty()
    st.success(f"导入完成：成功 {summary['success']}，"
               f"失败 {summary['failed']}，跳过 {summary['skipped']}")
    if summary["errors"]:
        with st.expander("失败详情"):
            for e in summary["errors"]:
                st.text(f"{e['path']}: {e['error']}")
    st.rerun()


def _delete_doc(system, doc_id: str) -> None:
    """删除文档（回调形式，避免按钮 rerun 顺序问题）。"""
    system.delete_document(doc_id)


def _render_health(system) -> None:
    """系统状态（走会话缓存，默认折叠）。"""
    with st.expander("🔧 系统状态", expanded=False):
        health = _health_cached(system)
        rows = []
        for key, label in (("llm", "LLM"), ("embedding", "Embedding"),
                           ("vector_store", "向量库"), ("database", "数据库")):
            item = health.get(key) or {}
            ok = item.get("status") == "ok"
            rows.append(f'<div class="ra-doc"><div class="t">{"🟢" if ok else "🔴"} {label}'
                        f'</div><div class="m">{escape(_brief(item))}</div></div>')
        st.markdown("".join(rows), unsafe_allow_html=True)

        c1, c2 = st.columns([0.5, 0.5])
        with c1:
            if st.button("重新检测", use_container_width=True):
                st.session_state["_health_cache"] = (0.0, {})
                st.rerun()
        with c2:
            if st.button("查看原始 JSON", use_container_width=True):
                st.session_state["_show_raw_health"] = not st.session_state.get(
                    "_show_raw_health", False)

        if st.session_state.get("_show_raw_health"):
            st.json(health)


def _brief(item: Dict[str, Any]) -> str:
    """把健康检查结果压缩成一行摘要。"""
    if not item:
        return "无数据"
    if item.get("status") != "ok":
        return f"异常：{item.get('error', '未知')}"
    bits = []
    for k in ("provider", "model", "dim", "latency_ms", "num_vectors",
              "docs", "chunks", "sessions"):
        if k in item:
            bits.append(f"{k}={item[k]}")
    return " · ".join(bits) if bits else "正常"


def _health_cached(system) -> Dict[str, Any]:
    """
    健康检查结果缓存 120 秒（与 app.py 共用同一份 session_state 缓存）。

    health_check() 会真实调用一次 LLM 与 Embedding，逐帧执行会明显卡顿。
    """
    now = time.time()
    cache = st.session_state.get("_health_cache")
    if cache and now - cache[0] < 120:
        return cache[1]
    try:
        health = system.health_check()
    except Exception as exc:  # noqa: BLE001
        health = {"error": str(exc)}
    st.session_state["_health_cache"] = (now, health)
    return health


def _save_uploads(uploaded, system) -> List[str]:
    """保存上传文件到磁盘，返回路径列表。"""
    out_dir = Path(system.config.get("app.upload_dir", "./data/uploads"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploaded:
        p = out_dir / f.name
        if not p.exists() or p.stat().st_size != f.size:
            p.write_bytes(f.getbuffer())
        paths.append(str(p))
    return paths


def _load_history(system, session_id: str) -> List[Dict[str, Any]]:
    """从 SQLite 恢复会话历史（供界面展示）。"""
    rows = system.memory.get_messages(session_id, limit=200)
    return [{"role": r["role"], "content": r["content"]} for r in rows]
