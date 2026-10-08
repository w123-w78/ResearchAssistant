# -*- coding: utf-8 -*-
"""
模块三 5.3.1：ReAct 核心循环（手写实现，约 200 行核心逻辑）

ReAct = Reasoning + Acting，由 Yao et al. (ICLR 2023) 提出。
核心思想：让模型在"推理"和"行动"之间交替，用外部工具的真实反馈
纠正推理偏差，而不是一次性"凭空"生成答案。

循环结构（本项目实现）：

    Thought（思考：我现在知道什么？下一步该做什么？）
      ↓
    Action（行动：调用哪个工具，参数是什么）
      ↓
    Observation（观察：工具返回了什么）
      ↓
    ... 循环，直到产出 Final Answer 或达到最大迭代次数

为什么不直接用 LangChain 的 Agent：
  课程目标是"理解原理而非调用 API"。手写实现才能真正掌握
  Prompt 结构、输出解析、终止条件、错误恢复这些关键设计点。

本实现的工程增强（超出原始 ReAct）：
  1. 并行工具调用：一次决策里的多个独立工具并发执行
  2. 错误恢复：工具失败自动重试，重试仍失败则换策略并记录
  3. 死循环检测：重复动作超过阈值强制终止并给出已有结论
  4. 全链路轨迹记录：每一步都结构化保存，供前端可视化与 Bad Case 分析
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.config import Config
from ..utils.llm_client import BaseLLM, LLMError
from ..utils.text_utils import estimate_tokens
from ..utils.logger import RequestLogger, Timer, get_logger, metrics

from .memory import MemoryManager
from .router import QueryRouter
from .tools import ToolRegistry

logger = get_logger("research_assistant.agent")


# ==================================================================
# Agent System Prompt（决定 Agent 表现的关键）
# ==================================================================
AGENT_SYSTEM_PROMPT = """你是一个智能科研助理的"决策中枢"，负责通过调用工具来回答用户关于学术论文的问题。

## 可用工具
{tools}

## 工作方式
你必须严格按 ReAct 格式逐步推理，每轮输出以下三部分（缺一不可）：

Thought: 分析当前状态——我已经知道什么？还缺什么？下一步该调用哪个工具？
Action: 工具名称（必须是上面列表中的一个）
Action Input: 工具参数（JSON 格式，如 {{"query": "注意力机制"}}）

当你已经收集到足够信息时，输出：
Thought: 我已经获得足够信息，可以回答用户了
Final Answer: 你的最终回答（需标注引用来源，如 [1]）

## 重要规则
1. 每轮只输出一个 Thought + Action + Action Input（或 Final Answer），不要输出多轮。
2. 涉及论文内容的问题，必须先调用 knowledge_search 获取原文依据，禁止凭记忆编造。
3. 工具返回"未找到"时，换用更具体/更抽象的关键词重试一次，不要立刻放弃。
4. 若连续两次检索均无结果，直接说明"知识库中未找到相关资料"，不要无限重试。
5. 需要计算数值时用 calculator，不要心算。
6. 最终回答要标注引用：在句末用 [1]、[2] 对应检索结果的编号。
7. 严禁输出 Action 之外的自由文本，严禁编造工具返回的内容。

## 并行调用（可选）
若需要多个互不依赖的工具，可用列表格式一次性调用：
Action: ["knowledge_search", "get_current_time"]
Action Input: [{{"query": "方法"}}, {{}}]

现在开始。"""


# ==================================================================
# 数据结构
# ==================================================================
@dataclass
class AgentStep:
    """ReAct 的单步轨迹（Thought -> Action -> Observation）。"""
    iteration: int
    thought: str = ""
    action: str = ""
    action_input: Any = None
    observation: str = ""
    parallel: bool = False
    error: str = ""
    latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation[:1000],
            "parallel": self.parallel,
            "error": self.error,
            "latency": round(self.latency, 3),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"Step#{self.iteration} [{self.action}] {self.thought[:40]}"


@dataclass
class AgentResult:
    """Agent 运行结果。"""
    query: str
    answer: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    iterations: int = 0
    latency: float = 0.0
    tokens: int = 0
    tool_calls: int = 0
    success: bool = True
    error: str = ""
    route: str = ""                     # direct / tool / agent
    terminated_by: str = "final_answer"  # final_answer / max_iterations / loop_detected / error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "iterations": self.iterations, "latency": round(self.latency, 3),
            "tokens": self.tokens, "tool_calls": self.tool_calls,
            "success": self.success, "error": self.error,
            "route": self.route, "terminated_by": self.terminated_by,
        }

    @property
    def trace_text(self) -> str:
        """人类可读的推理轨迹（前端展示）。"""
        lines = []
        for s in self.steps:
            lines.append(f"【第{s.iteration}轮】")
            if s.thought:
                lines.append(f"  Thought: {s.thought}")
            if s.action:
                lines.append(f"  Action: {s.action}  Input: {s.action_input}")
            if s.observation:
                obs = s.observation[:300] + ("..." if len(s.observation) > 300 else "")
                lines.append(f"  Observation: {obs}")
            if s.error:
                lines.append(f"  Error: {s.error}")
        return "\n".join(lines)


# ==================================================================
# ReAct Agent
# ==================================================================
class ReActAgent:
    """
    手写 ReAct Agent。

    用法：
        agent = ReActAgent(llm=llm, tools=registry, memory=memory)
        result = agent.run("这篇论文用了什么数据集？", session_id="abc")
    """

    def __init__(self, llm: BaseLLM, tools: ToolRegistry,
                 memory: Optional[MemoryManager] = None,
                 config: Optional[Config] = None,
                 router: Optional[QueryRouter] = None,
                 has_documents_func: Optional[Callable[[], bool]] = None,
                 db=None):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.config = config or Config.instance()
        self.db = db
        a_cfg = self.config.section("agent")

        self.max_iterations = int(a_cfg.get("max_iterations", 6))
        self.max_workers = int(a_cfg.get("max_tool_workers", 4))
        self.enable_parallel = bool(a_cfg.get("enable_parallel", True))
        self.enable_router = bool(a_cfg.get("enable_router", True))
        self.max_tool_retries = int(a_cfg.get("max_tool_retries", 2))
        self.loop_window = int(a_cfg.get("loop_detect_window", 3))

        self.router = router or QueryRouter(enabled=self.enable_router, llm=llm)
        self._has_documents_func = has_documents_func
        self.request_logger = RequestLogger(db=self.db)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, query: str, session_id: str = "default",
            on_step: Optional[Callable[[AgentStep], None]] = None,
            force_agent: bool = False) -> AgentResult:
        """
        执行一次 Agent 推理。

        :param on_step: 每一步的回调（前端流式展示推理轨迹用）
        :param force_agent: 跳过路由，强制走 ReAct（对比实验用）
        """
        start = time.perf_counter()
        result = AgentResult(query=query)
        has_docs = self._has_documents()

        # ---- 0) 快速路由（省 Token） ----
        if self.enable_router and not force_agent:
            route = self.router.route(query, has_documents=has_docs)
            result.route = route.route
            logger.info("路由决策: %s -> %s (%s)", query[:30], route.route, route.reason)
            if route.route == "tool" and route.tool_name:
                obs = self.tools.run(route.tool_name, route.tool_args)
                step = AgentStep(iteration=1, thought=route.reason,
                                 action=route.tool_name, action_input=route.tool_args,
                                 observation=obs)
                result.steps.append(step)
                result.tool_calls += 1
                if on_step:
                    on_step(step)
                result.answer = self._compose_from_tool(query, route.tool_name, obs)
                result.iterations = 1
                result.latency = time.perf_counter() - start
                result.tokens = estimate_tokens(query + obs + result.answer)
                self._finalize(result, session_id)
                return result
            if route.route == "direct":
                result.answer = self._direct_answer(query, has_docs)
                result.iterations = 0
                result.latency = time.perf_counter() - start
                result.tokens = estimate_tokens(query + result.answer)
                self._finalize(result, session_id)
                return result

        result.route = result.route or "agent"

        # ---- 1) 构建初始消息 ----
        messages = self._build_messages(query, session_id)
        action_history: List[str] = []

        # ---- 2) ReAct 循环 ----
        for iteration in range(1, self.max_iterations + 1):
            step = AgentStep(iteration=iteration)
            try:
                raw = self.llm.chat(messages, temperature=0.1)
            except LLMError as exc:
                logger.error("Agent LLM 调用失败: %s", exc)
                result.error = str(exc)
                result.success = False
                result.terminated_by = "error"
                result.answer = ("抱歉，模型服务不可用，无法完成推理。"
                                 "请检查 Ollama 是否已启动或 API 配置是否正确。")
                break

            result.tokens += estimate_tokens(raw)
            thought, actions, final = self._parse_output(raw)
            step.thought = thought
            result.iterations = iteration

            # ---- 2a) 终止：Final Answer ----
            if final:
                result.answer = final
                result.terminated_by = "final_answer"
                result.steps.append(step)
                if on_step:
                    on_step(step)
                break

            # ---- 2b) 解析失败：强制引导 ----
            if not actions:
                step.error = "未能解析出有效的 Action，已提示模型修正输出格式"
                step.observation = "（输出格式错误，请严格按 Thought/Action/Action Input 格式输出）"
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    "你的输出格式不正确。请严格按以下格式输出：\n"
                    "Thought: ...\nAction: 工具名\nAction Input: {\"参数\": \"值\"}\n"}) 
                result.steps.append(step)
                if on_step:
                    on_step(step)
                continue

            # ---- 2c) 死循环检测 ----
            signature = self._signature(actions)
            action_history.append(signature)
            if action_history.count(signature) >= self.loop_window:
                logger.warning("检测到重复动作 %s，触发死循环保护", signature)
                step.error = f"检测到重复动作 {self.loop_window} 次，已强制终止循环"
                result.steps.append(step)
                result.answer = self._force_answer(query, messages, result.steps)
                result.terminated_by = "loop_detected"
                break

            # ---- 2d) 执行工具（支持并行） ----
            t0 = time.perf_counter()
            if len(actions) > 1 and self.enable_parallel:
                observations = self._run_parallel(actions)
                step.parallel = True
            else:
                observations = [self._run_single(a) for a in actions]
            step.latency = time.perf_counter() - t0
            result.tool_calls += len(actions)

            step.action = ", ".join(a[0] for a in actions)
            step.action_input = [a[1] for a in actions] if len(actions) > 1 else actions[0][1]
            step.observation = "\n\n".join(observations)
            result.steps.append(step)
            if on_step:
                on_step(step)

            # 把观察结果反馈给模型
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {step.observation}"})

        # ---- 3) 达到最大迭代次数仍未收敛 ----
        if not result.answer:
            logger.warning("达到最大迭代次数 %s，强制收敛", self.max_iterations)
            result.answer = self._force_answer(query, messages, result.steps)
            result.terminated_by = "max_iterations"

        result.latency = time.perf_counter() - start
        metrics.incr("agent.tokens", result.tokens)
        metrics.observe("agent.latency", result.latency)
        metrics.incr("agent.iterations", result.iterations)
        self._finalize(result, session_id)
        return result

    # ------------------------------------------------------------------
    # 输出解析
    # ------------------------------------------------------------------
    def _parse_output(self, text: str) -> Tuple[str, List[Tuple[str, Dict[str, Any]]], str]:
        """
        解析 LLM 输出，返回 (thought, actions, final_answer)。

        容错设计：真实模型经常不严格遵循格式，这里做了多层兜底：
          - "Thought:" / "思考:" 多种前缀
          - Action Input 可能是 JSON 也可能是裸字符串
          - Final Answer 可能在文本中间出现
        """
        text = (text or "").strip()
        thought = self._extract(r"(?:Thought|思考)[:：]\s*(.*?)(?=\n(?:Action|行动|Final|最终)|$)", text)
        final = self._extract(r"(?:Final Answer|最终答案|Final)[:：]\s*(.*)", text)

        actions: List[Tuple[str, Dict[str, Any]]] = []
        action_raw = self._extract(r"(?:Action|行动)[:：]\s*(.*?)(?=\n(?:Action Input|参数|输入)|$)", text)
        input_raw = self._extract(r"(?:Action Input|参数|输入)[:：]\s*(.*?)(?=\n(?:Thought|思考|Observation|观察|Final|最终)|$)", text)

        if action_raw and action_raw.lower() not in ("none", "无", ""):
            # 并行格式：Action 是 JSON 数组
            if action_raw.strip().startswith("["):
                try:
                    names = json.loads(action_raw.strip())
                    inputs = json.loads(input_raw.strip()) if input_raw.strip().startswith("[") else []
                    for i, n in enumerate(names):
                        args = inputs[i] if i < len(inputs) and isinstance(inputs[i], dict) else {}
                        actions.append((str(n), args))
                except json.JSONDecodeError:
                    pass
            else:
                name = action_raw.strip().strip('"\'').strip()
                args = self._parse_input(input_raw)
                actions.append((name, args))

        return thought, actions, final

    @staticmethod
    def _extract(pattern: str, text: str) -> str:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _parse_input(raw: str) -> Dict[str, Any]:
        """解析 Action Input：优先 JSON，失败则尝试 key=value，最后当 query 处理。"""
        raw = (raw or "").strip()
        if not raw:
            return {}
        # 去掉可能的 Markdown 代码块
        raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            pass
        # key=value 形式
        if "=" in raw and ":" not in raw:
            out = {}
            for part in raw.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip().strip('"\'')
            if out:
                return out
        # 裸字符串，默认作为 query
        return {"query": raw.strip('"\'')}

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _run_single(self, action: Tuple[str, Dict[str, Any]]) -> str:
        """执行单个工具（带重试与错误恢复）。"""
        name, args = action
        tool = self.tools.get(name)
        if tool is None:
            return (f"[错误] 工具 '{name}' 不存在。可用工具：{', '.join(self.tools.names)}。"
                    f"请重新选择工具。")
        obs = tool.run(args)
        # 错误恢复：重试
        for attempt in range(self.max_tool_retries):
            if not obs.startswith("[错误]"):
                break
            logger.warning("工具 %s 第 %s 次重试", name, attempt + 1)
            obs = tool.run(args)
        return obs

    def _run_parallel(self, actions: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """并行执行多个独立工具（降低端到端延迟）。"""
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(actions))) as ex:
            futures = [ex.submit(self._run_single, a) for a in actions]
            return [f.result() for f in futures]

    @staticmethod
    def _signature(actions: List[Tuple[str, Dict[str, Any]]]) -> str:
        """生成动作指纹（死循环检测用）。"""
        return "|".join(f"{n}:{json.dumps(a, ensure_ascii=False, sort_keys=True)}"
                        for n, a in actions)

    # ------------------------------------------------------------------
    # 消息构建与收尾
    # ------------------------------------------------------------------
    def _build_messages(self, query: str, session_id: str) -> List[Dict[str, str]]:
        system = AGENT_SYSTEM_PROMPT.format(tools=self.tools.describe())
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if self.memory is not None:
            messages.extend(self.memory.get_context(session_id, max_messages=6))
        messages.append({"role": "user", "content": query})
        return messages

    def _force_answer(self, query: str, messages: List[Dict[str, str]],
                      steps: List[AgentStep]) -> str:
        """强制收敛：基于已收集的观察生成总结性回答。"""
        observations = "\n\n".join(
            f"[{s.iteration}] {s.action}: {s.observation[:600]}" for s in steps if s.observation
        )
        if not observations:
            return ("未能获取足够信息来回答该问题。建议：换用更具体的关键词，"
                    "或确认相关论文已上传到知识库。")
        prompt = (
            "以下是你在前几轮工具调用中收集到的信息。请基于这些信息直接回答用户问题，"
            "不要再调用工具。若信息仍不足，如实说明缺少什么。\n\n"
            f"用户问题：{query}\n\n已收集信息：\n{observations}\n\n"
            "请直接给出最终回答（标注引用编号）："
        )
        try:
            return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2).strip()
        except Exception as exc:  # noqa: BLE001
            return f"推理未能收敛，且总结生成失败：{exc}\n\n已收集信息：\n{observations[:800]}"

    def _compose_from_tool(self, query: str, tool_name: str, observation: str) -> str:
        """
        路由直达工具时，把工具原始输出润色成自然语言答案。

        为什么需要：直接把 JSON 抛给用户（如 {"expression":"...","result":3.76}）
        体验很差。这里用一次低成本 LLM 调用转成"答案是 3.76"这类表述，
        失败时回退到原始输出，保证功能不中断。
        """
        if self.llm is None:
            return f"【调用工具：{tool_name}】\n\n{observation}"
        prompt = (
            f"用户问题：{query}\n\n"
            f"系统已调用工具「{tool_name}」并得到如下结果：\n{observation}\n\n"
            "请用简洁的中文把该结果组织成对用户问题的直接回答（1-3 句）。\n"
            "要求：不要出现“根据工具返回”这类套话，不要编造结果之外的内容，"
            "涉及数值时保留原始精度。"
        )
        try:
            return self.llm.chat([{"role": "user", "content": prompt}],
                                 temperature=0.2).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("工具结果润色失败，回退原始输出: %s", exc)
            return f"【调用工具：{tool_name}】\n\n{observation}"

    def _direct_answer(self, query: str, has_docs: bool) -> str:
        """不检索，直接由模型回答（通用概念 / 寒暄）。"""
        hint = ""
        if not has_docs:
            hint = ("\n\n（提示：当前知识库为空。如需基于私有论文问答，"
                    "请先在左侧上传 PDF/Word 文档。）")
        try:
            ans = self.llm.chat([
                {"role": "system", "content": "你是智能科研助理。请用简洁专业的中文回答。"
                                              "这是通用知识问题，无需引用私有文献。"},
                {"role": "user", "content": query},
            ], temperature=0.3)
            return ans.strip() + hint
        except Exception as exc:  # noqa: BLE001
            logger.error("直答失败: %s", exc)
            return f"模型服务暂时不可用（{exc}）。请检查 LLM 配置。"

    def _has_documents(self) -> bool:
        if self._has_documents_func:
            try:
                return bool(self._has_documents_func())
            except Exception:  # noqa: BLE001
                return True
        return True

    def _finalize(self, result: AgentResult, session_id: str) -> None:
        """写记忆 + 结构化日志。"""
        if self.memory is not None:
            self.memory.add_message(session_id, "user", result.query)
            self.memory.add_message(session_id, "assistant", result.answer,
                                    steps=len(result.steps), route=result.route)
        self.request_logger.log_agent(
            query=result.query, answer=result.answer,
            steps=[s.to_dict() for s in result.steps],
            latency=result.latency, tokens=result.tokens,
            extra={"route": result.route, "tool_calls": result.tool_calls,
                   "iterations": result.iterations, "terminated_by": result.terminated_by,
                   "success": result.success},
        )

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.name,
            "tools": self.tools.names,
            "max_iterations": self.max_iterations,
            "router_enabled": self.enable_router,
            "parallel_enabled": self.enable_parallel,
        }
