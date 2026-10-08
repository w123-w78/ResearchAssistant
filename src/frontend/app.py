# -*- coding: utf-8 -*-
"""
模块四：Streamlit 前端主程序
----------------------------
三栏布局（每栏独立滚动，互不挤压）：
  左侧  知识库管理（上传 / 文档列表 / 会话切换 / 系统状态）
  中部  对话窗口（流式打字机 + Markdown 渲染 + 引用标记）
  右侧  Agent 推理过程可视化（Thought/Action/Observation 轨迹 + 指标）

启动： streamlit run src/frontend/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# 保证以 `streamlit run` 启动时也能 import 项目根下的 src 包
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frontend.components import (render_citations, render_metrics,  # noqa: E402
                                     render_sidebar, render_trace)
from src.frontend.components.style import (empty_state, inject_global_css,  # noqa: E402
                                           render_header)
from src.system import ResearchAssistantSystem  # noqa: E402
from src.utils.llm_client import LLMError  # noqa: E402

# 主体三栏的固定高度（px）：让每栏内部滚动，而不是把整页无限拉长
BODY_HEIGHT = 620

EXAMPLE_QUESTIONS = [
    "Transformer 为什么要使用多头注意力？",
    "对比 BERT 和 GPT 的预训练目标",
    "这篇论文的位置编码是怎么设计的？",
    "总结一下论文的实验设置与主要结论",
]


@st.cache_resource(show_spinner="正在初始化系统...")
def get_system() -> ResearchAssistantSystem:
    """全局唯一的系统实例（避免每次交互重复加载模型与索引）。"""
    sys_obj = ResearchAssistantSystem()
    sys_obj.load_index()      # 启动时尝试恢复已构建的向量索引
    return sys_obj


def init_state() -> None:
    st.session_state.setdefault("session_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_meta", {})
    st.session_state.setdefault("last_steps", [])


def _in_streamlit_runtime() -> bool:
    """
    判断是否通过 `streamlit run` 启动。

    直接 `python app.py` 时 Streamlit 处于 bare mode：
    没有 ScriptRunContext，所有 st.xxx 都是空操作，浏览器不会打开。
    这里提前检测并给出明确指引，避免用户看到一堆警告却不知所以。
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001
        return True   # 无法判断时放行，不阻塞正常运行


def _print_usage_hint() -> None:
    app_path = str(Path(__file__).resolve())
    print("\n" + "=" * 66)
    print(" 本应用是 Streamlit 应用，不能直接用 python 启动")
    print("=" * 66)
    print("\n请改用以下命令：\n")
    print(f"    streamlit run {app_path}\n")
    print("说明：Streamlit 需要由 `streamlit run` 启动服务器进程来创建")
    print("运行时上下文（会话状态、WebSocket、组件渲染）。直接 python 运行时")
    print("所有界面调用都会变成空操作，因此不会打开浏览器。\n")
    print("若需命令行交互，可改用：")
    print(f"    python {str(Path(app_path).parents[2] / 'scripts' / 'cli_demo.py')} -i")
    print("=" * 66 + "\n")


def _status_badges(system: ResearchAssistantSystem) -> list[dict]:
    """顶部状态徽章（走会话内缓存，避免每帧都访问远端服务）。"""
    health = _cached_health(system)
    badges = []

    llm = health.get("llm", {}) or {}
    badges.append({
        "text": f"LLM {llm.get('provider', '-')}" +
                ("" if llm.get("status") == "ok" else " 异常"),
        "kind": "ok" if llm.get("status") == "ok" else "err",
    })

    emb = health.get("embedding", {}) or {}
    badges.append({
        "text": f"Embedding {emb.get('model', '-')}" +
                ("" if emb.get("status") == "ok" else " 异常"),
        "kind": "ok" if emb.get("status") == "ok" else "err",
    })

    vs = health.get("vector_store", {}) or {}
    badges.append({
        "text": f"索引 {vs.get('num_vectors', vs.get('total_vectors', 0))} 块",
        "kind": "info" if vs.get("status") == "ok" else "err",
    })

    tools = health.get("tools", {}) or {}
    badges.append({"text": f"工具 {tools.get('count', 0)} 个", "kind": "plain"})
    return badges


def _cached_health(system: ResearchAssistantSystem) -> dict:
    """
    健康检查结果缓存 120 秒。

    health_check() 会真实调用一次 LLM 与 Embedding，若每次重绘都执行，
    页面交互会明显卡顿；因此放进 session_state 并配 TTL。
    """
    now = time.time()
    cache = st.session_state.get("_health_cache")
    if cache and now - cache[0] < 120:
        return cache[1]
    try:
        health = system.health_check()
    except Exception as exc:  # noqa: BLE001
        health = {"overall": "error", "error": str(exc)}
    st.session_state["_health_cache"] = (now, health)
    return health


def main() -> None:
    if not _in_streamlit_runtime():
        _print_usage_hint()
        return

    cfg_default = {"title": "智能科研助理 · 论文知识库问答", "page_icon": "🔬",
                   "layout": "wide"}
    st.set_page_config(page_title="智能科研助理", page_icon="🔬", layout="wide")
    inject_global_css()
    init_state()

    # 系统初始化失败（如未配置 API Key）时给出明确指引
    try:
        system = get_system()
    except LLMError as exc:
        st.error("❌ 系统初始化失败")
        st.code(str(exc))
        st.info("配置好 DASHSCOPE_API_KEY 后重新运行：`streamlit run src/frontend/app.py`")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ 系统初始化失败：{type(exc).__name__}: {exc}")
        return

    cfg = system.config
    title = cfg.get("frontend.title", cfg_default["title"])
    icon = cfg.get("frontend.page_icon", cfg_default["page_icon"])
    render_header(
        f"{icon} {title}",
        "基于 RAG + Agent 的论文知识库问答系统 · 通义千问 + SQLite",
        _status_badges(system),
    )

    if not st.session_state.session_id:
        st.session_state.session_id = system.memory.create_session().session_id

    # ---------------- 三栏主体（每栏独立滚动） ----------------
    left, center, right = st.columns([0.24, 0.50, 0.26], gap="small")

    with left:
        with st.container(height=BODY_HEIGHT):
            render_sidebar(system)

    with center:
        with st.container(height=BODY_HEIGHT):
            _render_history()

    with right:
        with st.container(height=BODY_HEIGHT):
            _render_trace_panel()

    # 输入框置于三栏之外，由 Streamlit 固定在视口底部
    prompt = st.chat_input("输入关于论文的问题，Enter 发送")
    if prompt:
        _handle_query(system, prompt)


def _render_history() -> None:
    """渲染历史消息；无消息时展示引导空状态。"""
    if not st.session_state.messages:
        empty_state(
            "🔬",
            "开始你的论文问答",
            "在左侧上传 PDF / Word / Markdown 论文并完成导入，<br/>"
            "然后在下方输入框提问，右侧会实时展示 Agent 的推理链路。",
            EXAMPLE_QUESTIONS,
        )
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown('<span class="ra-user-flag"></span>', unsafe_allow_html=True)
            st.markdown(m["content"])


def _render_trace_panel() -> None:
    """右侧推理轨迹面板。"""
    from src.frontend.components.style import section

    section("Agent 推理过程", "🔍")
    meta = st.session_state.get("last_meta", {})
    if not meta:
        empty_state(
            "🧭",
            "等待提问",
            "发起提问后，这里会展示<br/>Thought → Action → Observation 全链路轨迹。",
        )
        return

    render_metrics(meta)
    render_trace(st.session_state.get("last_steps", []), meta)
    render_citations(meta.get("citations", []))


def _handle_query(system: ResearchAssistantSystem, prompt: str) -> None:
    """处理一次提问：流式输出 + 轨迹记录。"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown('<span class="ra-user-flag"></span>', unsafe_allow_html=True)
        st.markdown(prompt)

    steps, meta = [], {}
    with st.chat_message("assistant"):
        try:
            def token_gen():
                for ev in system.ask_stream(prompt, session_id=st.session_state.session_id):
                    if ev["type"] == "step":
                        steps.append(ev["step"])
                    elif ev["type"] == "token":
                        yield ev["content"]
                    elif ev["type"] == "done":
                        meta.update(ev["meta"])

            answer = st.write_stream(token_gen())
        except Exception as exc:  # noqa: BLE001
            answer = f"❌ 生成失败：{type(exc).__name__}: {exc}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_steps = steps
    st.session_state.last_meta = meta
    if meta:
        st.rerun()


if __name__ == "__main__":
    main()
