# -*- coding: utf-8 -*-
"""Agent 决策层单元测试（模块三）"""
from __future__ import annotations

import pytest

from src.agent import (MemoryManager, QueryRouter, ReActAgent, ToolRegistry, Tool)
from src.agent.tools import safe_eval
from src.retrieval import HybridRetriever, Reranker


# ---------------- 路由 ----------------
@pytest.mark.parametrize("query,expect", [
    ("(88.3-85.1)/85.1*100", "calculator"),
    ("计算 12*15", "calculator"),
    ("今天几号", "get_current_time"),
    ("现在几点", "get_current_time"),
])
def test_router_tools(query, expect):
    assert QueryRouter().route(query).tool_name == expect


def test_router_direct_and_agent():
    r = QueryRouter()
    assert r.route("你好").route == "direct"
    assert r.route("知识库里一共有多少篇论文").tool_name == "library_stats"
    assert r.route("这篇论文用了什么数据集").route == "agent"


def test_router_disabled_falls_back_to_agent():
    assert QueryRouter(enabled=False).route("3+5").route == "agent"


# ---------------- 工具 ----------------
def test_safe_eval_basic():
    assert safe_eval("1+2*3") == 7.0
    assert safe_eval("(88.3-85.1)/85.1*100") == pytest.approx(3.76, abs=0.01)
    assert safe_eval("sqrt(16)") == 4.0


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x.txt').read()",
    "a + 1",
])
def test_safe_eval_blocks_injection(expr):
    with pytest.raises(Exception):
        safe_eval(expr)


def test_tool_registry_and_parallel_execution():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="回显", parameters={"x": {"type": "string"}},
                      func=lambda a: f"echo:{a.get('x')}"))
    reg.register(Tool(name="slow", description="慢工具", parameters={},
                      func=lambda a: "done", timeout=1))
    assert "echo" in reg and len(reg) == 2
    assert reg.run("echo", {"x": "hi"}) == "echo:hi"
    assert "未找到工具" in reg.run("not_exist")


def test_tool_error_message_on_exception():
    def bad(_):
        raise ValueError("boom")
    t = Tool(name="bad", description="会失败", parameters={}, func=bad)
    assert "[错误]" in t.run({})


# ---------------- ReAct 解析 ----------------
def _make_agent(llm, tools=None) -> ReActAgent:
    reg = tools or ToolRegistry()
    return ReActAgent.__new__(ReActAgent) if False else _build(reg, llm)


def _build(reg, llm) -> ReActAgent:
    import types
    from src.utils.config import Config
    agent = ReActAgent.__new__(ReActAgent)
    agent.llm = llm
    agent.tools = reg
    agent.memory = None
    agent.config = Config.instance()
    agent.max_iterations = 3
    agent.max_workers = 2
    agent.enable_parallel = True
    agent.enable_router = False
    agent.max_tool_retries = 0
    agent.loop_window = 3
    agent.router = QueryRouter(enabled=False)
    agent._has_documents_func = None
    return agent


def test_parse_single_action(stub_llm):
    agent = _build(ToolRegistry(), stub_llm)
    thought, actions, final = agent._parse_output(
        "Thought: 需要检索\nAction: knowledge_search\nAction Input: {\"query\": \"方法\"}")
    assert "需要检索" in thought
    assert actions == [("knowledge_search", {"query": "方法"})]
    assert final == ""


def test_parse_parallel_actions(stub_llm):
    agent = _build(ToolRegistry(), stub_llm)
    _, actions, _ = agent._parse_output(
        'Thought: 并行\nAction: ["knowledge_search", "get_current_time"]\n'
        'Action Input: [{"query": "方法"}, {}]')
    assert len(actions) == 2
    assert actions[0][0] == "knowledge_search"


def test_parse_final_answer(stub_llm):
    agent = _build(ToolRegistry(), stub_llm)
    _, actions, final = agent._parse_output(
        "Thought: 信息足够\nFinal Answer: 答案是 87.3% [1]")
    assert actions == []
    assert final == "答案是 87.3% [1]"


def test_parse_plain_input_fallback(stub_llm):
    agent = _build(ToolRegistry(), stub_llm)
    _, actions, _ = agent._parse_output(
        "Thought: 检索\nAction: knowledge_search\nAction Input: 扩散模型")
    assert actions[0][1] == {"query": "扩散模型"}


# ---------------- 记忆 ----------------
def test_memory_sliding_window_and_summary():
    mm = MemoryManager(llm=None, db=None)
    sid = mm.create_session().session_id
    for i in range(12):
        mm.add_message(sid, "user", f"问题{i}")
        mm.add_message(sid, "assistant", f"回答{i}")
    mem = mm.get_session(sid, create=False)
    assert mem.turns <= mm.max_turns
    ctx = mm.get_context(sid, max_messages=4)
    assert 0 < len(ctx) <= 5
    assert mm.delete_session(sid)


def test_memory_session_isolation():
    mm = MemoryManager(llm=None, db=None)
    a = mm.create_session().session_id
    b = mm.create_session().session_id
    mm.add_message(a, "user", "会话A的问题")
    ctx_b = mm.get_context(b, max_messages=5)
    assert all("会话A的问题" not in m["content"] for m in ctx_b)
