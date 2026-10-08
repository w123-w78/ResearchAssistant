# -*- coding: utf-8 -*-
"""
全局样式与通用 UI 片段
----------------------
集中管理 Streamlit 的 CSS 注入，避免各组件里散落 `st.markdown("<style>")`。

配色尽量复用 Streamlit 自带的 CSS 变量（--text-color / --background-color /
--secondary-background-color / --primary-color），从而同时适配浅色与深色主题；
只有强调色（如状态徽章）使用固定色值。
"""
from __future__ import annotations

from html import escape
from typing import Any, Dict, List, Sequence

import streamlit as st

# ---------------------------------------------------------------- CSS ----
_CSS = """
<style>
:root{
  --ra-radius: 14px;
  --ra-border: rgba(128,145,170,.22);
  --ra-shadow: 0 1px 2px rgba(15,23,42,.04), 0 6px 18px rgba(15,23,42,.06);
  --ra-muted: rgba(128,145,170,.9);
}

/* 字体与全局留白 */
html, body, [class*="css"], .stMarkdown, .stText, .stCaption, .stCode{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", "Hiragino Sans GB", "Helvetica Neue", Arial, sans-serif;
}
.block-container{ padding-top: 1.0rem !important; padding-bottom: .3rem !important; }
[data-testid="stHeader"]{ background: transparent; height: 0; }
[data-testid="stDecoration"]{ display: none; }
footer{ visibility: hidden; }
#MainMenu{ visibility: hidden; }

/* 顶部标题条 */
.ra-header{
  display:flex; align-items:flex-end; justify-content:space-between;
  gap:1rem; flex-wrap:wrap; margin:0 0 .45rem;
}
.ra-title{ font-size:1.45rem; font-weight:700; margin:0; line-height:1.25; }
.ra-sub{ font-size:.8rem; color: var(--ra-muted); margin-top:.12rem; }
.ra-badges{ display:flex; gap:.35rem; flex-wrap:wrap; justify-content:flex-end; }
.ra-badge{
  font-size:.72rem; line-height:1.6; padding:.12rem .55rem; border-radius:999px;
  border:1px solid var(--ra-border); white-space:nowrap;
}
.ra-badge.ok   { color:#0f766e; border-color:rgba(15,118,110,.4);  background:rgba(15,118,110,.08); }
.ra-badge.err  { color:#b91c1c; border-color:rgba(185,28,28,.4);   background:rgba(185,28,28,.08); }
.ra-badge.info { color:#1d4ed8; border-color:rgba(29,78,216,.35);  background:rgba(29,78,216,.08); }
.ra-badge.plain{ color: var(--ra-muted); }

/* 分栏容器：卡片化 + 独立滚动 */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: var(--ra-radius) !important;
  border: 1px solid var(--ra-border) !important;
  box-shadow: var(--ra-shadow);
  overflow: auto;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar{ width:6px; }
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb{
  background: rgba(128,145,170,.35); border-radius: 99px; }

/* 对话气泡 */
[data-testid="stChatMessage"]{
  background: var(--secondary-background-color);
  border:1px solid var(--ra-border);
  border-radius: var(--ra-radius);
  padding: .75rem .95rem .65rem;
  margin-bottom: .5rem;
}
[data-testid="stChatMessage"]:has(.ra-user-flag){
  background: rgba(37,99,235,.07);
  border-color: rgba(37,99,235,.28);
}
[data-testid="stChatMessage"] p{ margin-bottom:.35rem; }
[data-testid="stChatMessage"] p:last-child{ margin-bottom:0; }
.ra-user-flag{ display:none; }

/* 分区小标题 */
.ra-sec{
  font-size:.76rem; font-weight:600; color: var(--ra-muted);
  margin:.15rem 0 .45rem; display:flex; align-items:center; gap:.4rem;
}
.ra-sec::after{ content:""; flex:1; height:1px; background: var(--ra-border); }

/* 文档卡片 */
.ra-doc{
  padding:.45rem .6rem; border:1px solid var(--ra-border); border-radius:10px;
  background: var(--secondary-background-color); min-height:2.9rem;
  display:flex; flex-direction:column; justify-content:center;
}
.ra-doc .t{ font-size:.85rem; font-weight:600; line-height:1.35;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ra-doc .m{ font-size:.71rem; color: var(--ra-muted); margin-top:.12rem; }

/* 指标小卡 */
.ra-metric{
  border:1px solid var(--ra-border); border-radius:10px; padding:.5rem .55rem;
  background: var(--secondary-background-color); text-align:center;
}
.ra-metric .v{ font-size:1.02rem; font-weight:700; line-height:1.25; }
.ra-metric .k{ font-size:.7rem; color: var(--ra-muted); margin-top:.1rem; }

/* 空状态 */
.ra-empty{ text-align:center; padding:2.6rem 1rem; color: var(--ra-muted); }
.ra-empty .big{ font-size:2rem; }
.ra-empty .t{ font-size:.95rem; font-weight:600; color: var(--text-color); margin:.5rem 0 .25rem; }
.ra-empty .d{ font-size:.8rem; line-height:1.6; }
.ra-chips{ display:flex; flex-wrap:wrap; gap:.4rem; justify-content:center; margin-top:1rem; }
.ra-chip{
  font-size:.78rem; padding:.28rem .7rem; border-radius:999px;
  border:1px dashed var(--ra-border); color: var(--ra-muted);
}

/* 细节 */
hr{ margin:.7rem 0 !important; border-color: var(--ra-border) !important; }
.stExpander{ border-radius:10px !important; }
.stButton>button{ border-radius:10px !important; }
.stButton>button[data-testid="baseButton-secondary"]{ font-size:.8rem; }
</style>
"""


def inject_global_css() -> None:
    """注入全局样式（每次运行只需一次）。"""
    st.markdown(_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------ 片段 ----
def section(text: str, icon: str = "") -> None:
    """分区小标题（带右侧分隔线）。"""
    st.markdown(
        f'<div class="ra-sec">{escape(icon)} {escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, badges: Sequence[Dict[str, str]]) -> None:
    """
    顶部标题条：左侧标题+副标题，右侧状态徽章。

    badges: [{"text": "...", "kind": "ok|err|info|plain"}, ...]
    """
    chips = "".join(
        f'<span class="ra-badge {escape(b.get("kind", "plain"))}">{escape(str(b["text"]))}</span>'
        for b in badges
    )
    st.markdown(
        '<div class="ra-header">'
        f'  <div><h1 class="ra-title">{escape(title)}</h1>'
        f'  <div class="ra-sub">{escape(subtitle)}</div></div>'
        f'  <div class="ra-badges">{chips}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def metric_grid(items: Sequence[Dict[str, Any]], cols: int = 2) -> None:
    """紧凑指标网格（替代 st.metric，避免在窄栏里挤成一团）。"""
    if not items:
        return
    cells = "".join(
        f'<div class="ra-metric"><div class="v">{escape(str(i["value"]))}</div>'
        f'<div class="k">{escape(str(i["label"]))}</div></div>'
        for i in items
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
        f'gap:.4rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, desc: str, chips: List[str] | None = None) -> None:
    """居中空状态卡片，可选附一组示例问题 chip。"""
    chip_html = ""
    if chips:
        chip_html = '<div class="ra-chips">' + "".join(
            f'<span class="ra-chip">{escape(c)}</span>' for c in chips
        ) + "</div>"
    st.markdown(
        '<div class="ra-empty">'
        f'<div class="big">{escape(icon)}</div>'
        f'<div class="t">{escape(title)}</div>'
        f'<div class="d">{desc}</div>'
        f"{chip_html}"
        "</div>",
        unsafe_allow_html=True,
    )
