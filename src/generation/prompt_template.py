# -*- coding: utf-8 -*-
"""
模块二 5.2.1：Prompt 工程与模板

Prompt 是 RAG 系统里"性价比最高"的优化点（改几行字就能显著影响答案质量），
本模块集中管理所有模板，便于 A/B 对比实验。

模板设计要点：
  1. 角色设定明确："严谨的科研助理"，限定语气与行为边界
  2. 强约束引用：要求每个事实性陈述后标注 [n]，杜绝无来源断言
  3. 允许诚实拒答："资料未提及就说不知道" —— 这是降低幻觉的关键一条
  4. 上下文编号：[1] [2] ... 与引用标记一一对应，便于程序化解析
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..retrieval.base import RetrievedChunk

# ==================================================================
# 系统角色设定
# ==================================================================
SYSTEM_ROLE = """你是一名严谨的科研助理，服务于学术论文知识库问答场景。

你的行为准则：
1. 只依据【检索到的文献片段】回答，不得引入片段之外的 factual 断言。
2. 每一个事实性陈述都必须在句末标注引用编号，格式为 [1]、[2]。
3. 若检索片段不足以回答问题，请如实说明"根据当前知识库无法确定"，而不是编造内容。
4. 回答使用简洁、学术化的中文；涉及方法名、数据集名、指标名时保留英文原文。
5. 如果问题涉及多篇论文，请分点对比，避免混淆不同论文的结论。"""

# ==================================================================
# RAG 主模板
# ==================================================================
RAG_TEMPLATE = """{system_role}

【检索到的文献片段】
{context}

【用户问题】
{question}

【输出要求】
1. 先给出直接回答（1-3 句话），再展开要点说明。
2. 引用格式：在对应陈述末尾标注 [编号]，如"该方法在 ImageNet 上达到 88.3% [2]"。
3. 若多个片段结论冲突，请指出冲突并分别标注来源。
4. 若片段不相关或不充分，直接说明并给出建议（例如换关键词、补充上传论文）。

请回答："""

# 无检索结果时的降级模板
NO_CONTEXT_TEMPLATE = """{system_role}

【注意】当前知识库中未检索到与问题相关的文献片段（置信度不足）。

【用户问题】
{question}

【输出要求】
1. 首先明确告知用户："当前知识库中未找到相关文献，以下回答基于模型通用知识，仅供参考。"
2. 然后用你的通用知识给出简短回答。
3. 最后建议用户：上传相关论文，或换用更具体的关键词重新提问。

请回答："""

# 低相关性时的模板
LOW_SCORE_TEMPLATE = """{system_role}

【检索到的文献片段】（注意：相关性较低，请谨慎使用）
{context}

【用户问题】
{question}

【输出要求】
1. 先提示用户"检索到的内容相关性较低，以下回答可能不准确"。
2. 基于片段给出你能确定的部分，不确定的部分明确标注"无法确定"。
3. 展示你参考了哪些片段，供用户自行判断。

请回答："""

# ==================================================================
# 对话摘要压缩模板（模块三记忆管理使用）
# ==================================================================
SUMMARY_TEMPLATE = """请将以下对话历史压缩为一段简洁摘要。
要求：
1. 保留用户关注的核心主题、已确认的事实、已给出的结论。
2. 保留论文名称、方法名、数据集名等关键实体。
3. 丢弃寒暄、重复提问等无效信息。
4. 控制在 200 字以内，使用第三人称客观陈述。

对话历史：
{history}

摘要："""

# ==================================================================
# 关键词抽取模板
# ==================================================================
KEYWORD_TEMPLATE = """从下面的文本中提取 3-6 个核心关键词（用于文献检索）。
要求：只输出 JSON，格式为 {{"keywords": ["词1", "词2"]}}；优先保留专业术语、方法名、数据集名。

文本：
{text}

JSON："""


def build_context(hits: List[RetrievedChunk], max_chars: int = 6000,
                  include_meta: bool = True) -> str:
    """
    上下文智能拼接：
      - 按相关性降序（检索层已排好序）
      - 累积到 max_chars 即截断，防止超出模型上下文窗口
      - 每个片段前标注 [n] 来源（文档名 + 页码），便于模型引用

    :return: 拼接好的上下文字符串
    """
    if not hits:
        return ""
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] 来源：{h.chunk.doc_title}（第 {h.chunk.page} 页）"
        if include_meta:
            score_info = f"[相似度 {h.score:.3f}]"
            header += f" {score_info}"
        body = h.text.strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars and parts:
            parts.append(f"（后续 {len(hits) - i + 1} 个片段因长度限制被省略）")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def build_rag_messages(question: str, hits: List[RetrievedChunk],
                       history: Optional[List[Dict[str, str]]] = None,
                       max_context_chars: int = 6000,
                       low_score: bool = False) -> List[Dict[str, str]]:
    """
    构造完整的 messages（system + history + user）。

    :param low_score: 检索分数偏低时使用 LOW_SCORE_TEMPLATE
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_ROLE}]
    if history:
        # 只保留最近的 N 轮，避免超长
        messages.extend(history[-6:])

    if not hits:
        user_prompt = NO_CONTEXT_TEMPLATE.format(system_role="", question=question)
    elif low_score:
        context = build_context(hits, max_context_chars)
        user_prompt = LOW_SCORE_TEMPLATE.format(system_role="", context=context, question=question)
    else:
        context = build_context(hits, max_context_chars)
        user_prompt = RAG_TEMPLATE.format(system_role="", context=context, question=question)

    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_summary_messages(history: str) -> List[Dict[str, str]]:
    """对话摘要压缩的 messages。"""
    return [
        {"role": "system", "content": "你是一个对话压缩助手，只输出摘要本身，不要解释。"},
        {"role": "user", "content": SUMMARY_TEMPLATE.format(history=history)},
    ]


def build_keyword_messages(text: str) -> List[Dict[str, str]]:
    """关键词抽取的 messages。"""
    return [
        {"role": "system", "content": "你是关键词抽取助手，只输出 JSON，不要其他内容。"},
        {"role": "user", "content": KEYWORD_TEMPLATE.format(text=text[:2000])},
    ]


# ==================================================================
# 引用解析
# ==================================================================
def parse_citations(answer: str, hits: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """
    从答案文本中解析出被引用的编号，映射回真实来源。

    用于前端展示"引用溯源"卡片：答案里的 [2] -> 哪篇论文第几页。
    """
    import re

    nums = re.findall(r"\[(\d+)\]", answer)
    seen = []
    out: List[Dict[str, Any]] = []
    for n in nums:
        idx = int(n)
        if idx in seen or idx < 1 or idx > len(hits):
            continue
        seen.append(idx)
        h = hits[idx - 1]
        out.append({
            "index": idx,
            "doc_id": h.doc_id,
            "title": h.chunk.doc_title,
            "page": h.chunk.page,
            "text": h.text[:200],
            "score": round(h.score, 4),
            "citation": h.citation,
        })
    return out
