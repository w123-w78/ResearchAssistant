# -*- coding: utf-8 -*-
"""
推理轨迹面板：可视化 Agent 的 Thought -> Action -> Observation
"""
from __future__ import annotations

from html import escape
from typing import Any, Dict, List

import streamlit as st

from .style import metric_grid


def render_trace(steps: List[Dict[str, Any]], meta: Dict[str, Any] | None = None) -> None:
    """在右侧面板渲染 Agent 推理轨迹。"""
    if not steps:
        st.caption("本次问答未调用工具（直答或路由直达）")
        return

    for s in steps:
        n = s.get("iteration", "?")
        with st.container(border=True):
            if s.get("thought"):
                st.markdown(f"**🧠 第 {n} 轮 · Thought**")
                st.caption(s["thought"])

            if s.get("action"):
                parallel = " ⚡并行" if s.get("parallel") else ""
                st.markdown(f"**🔧 Action{parallel}**　`{s['action']}`")
                inp = s.get("action_input")
                if inp:
                    st.code(_fmt(inp), language="json")

            if s.get("error"):
                st.error(f"⚠ {s['error']}")

            obs = s.get("observation", "")
            if obs:
                with st.expander(f"👁 Observation（{len(obs)} 字）", expanded=False):
                    st.text(obs[:1500] + ("..." if len(obs) > 1500 else ""))

            tail = []
            if s.get("latency"):
                tail.append(f"耗时 {s['latency']}s")
            if s.get("tokens"):
                tail.append(f"{s['tokens']} tokens")
            if tail:
                st.caption(" · ".join(tail))


def render_metrics(meta: Dict[str, Any]) -> None:
    """渲染本次请求的指标（2×2 紧凑网格，适配窄栏）。"""
    if not meta:
        return

    metric_grid([
        {"label": "⏱ 耗时", "value": f"{meta.get('latency', 0):.2f}s"},
        {"label": "🔁 迭代", "value": meta.get("iterations", 0)},
        {"label": "🛠 工具调用", "value": meta.get("tool_calls", 0)},
        {"label": "🔤 Token", "value": meta.get("tokens", 0)},
    ], cols=2)

    extras = []
    if meta.get("route"):
        extras.append(f"路由 `{meta['route']}`")
    if meta.get("terminated_by"):
        extras.append(f"终止 `{meta['terminated_by']}`")
    if meta.get("degraded"):
        extras.append(f"⚠ 降级：{meta.get('degrade_reason', '')}")
    if extras:
        st.caption(" · ".join(extras))


def render_citations(citations: List[Dict[str, Any]]) -> None:
    """渲染引用来源卡片。"""
    if not citations:
        return

    with st.expander(f"📎 引用来源（{len(citations)}）"):
        for c in citations:
            title = c.get("title", "")
            page = c.get("page", "")
            meta_bits = []
            if page:
                meta_bits.append(f"第 {page} 页")
            if c.get("score"):
                meta_bits.append(f"相关度 {c['score']}")
            st.markdown(
                '<div class="ra-doc" style="margin-bottom:.4rem">'
                f'<div class="t">[{c.get("index")}] {escape(str(title))}</div>'
                f'<div class="m">{escape(" · ".join(meta_bits))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if c.get("text"):
                st.caption(c["text"][:300])


def _fmt(obj) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)[:500]
    except Exception:  # noqa: BLE001
        return str(obj)[:500]
