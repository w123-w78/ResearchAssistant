# -*- coding: utf-8 -*-
"""
模块五 5.5.1：用 LLM 从语料自动生成评测集（50+ 问答对）
-------------------------------------------------------
为什么让模型出题：
  人工编写 50 道题成本高；而题目必须"不出现论文标题"
  （否则检索时靠标题字面匹配就能命中，指标虚高无区分度）。

题型设计（对齐课程要求的 4 类）：
  事实性  能从单篇直接找到答案（数据集、指标、参数）
  归纳性  需要跨多篇归纳（如"这些论文用到了哪些数据集"）
  推理性  需要结合上下文推断原因
  对比性  比较两篇论文的异同（跨文档，最难）

产出 reports/评测集.json，字段含 relevant_titles（按标题关联文档，
与具体 doc_id 解耦，导入后由 benchmark 自动映射）。

用法：python scripts/build_evalset.py [--per-paper 4] [--pairs 10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_directory          # noqa: E402
from src.utils.config import Config                 # noqa: E402
from src.utils.llm_client import build_llm          # noqa: E402

SINGLE_PROMPT = """你是学术评测集构建专家。下面是一篇论文，请生成 {n} 个问答对。
要求：
1. 问题中【禁止出现论文标题】，用内容特征描述（例如"哪种方法在仅10%标注数据时仍保持性能"）
2. 类型分配：事实性 2 条、归纳性 1 条、推理性 1 条
3. 答案必须能在原文找到依据，1-2 句，保留关键数字
4. 只输出 JSON 数组：[{{"type":"事实性","question":"...","answer":"..."}}]

论文：
{content}

JSON："""

PAIR_PROMPT = """你是学术评测集构建专家。下面是两篇论文的摘要，请生成 1 个【对比性】问答对。
要求：问题明确比较二者的差异（方法、数据集或结果），答案各用一句话概括并给关键数字。
问题中不出现论文标题，用内容特征指代。
只输出 JSON：[{{"type":"对比性","question":"...","answer":"..."}}]

论文A：{a}
论文B：{b}

JSON："""


def parse_json_array(text: str) -> list:
    """从 LLM 输出中抽取 JSON 数组（兼容 ```json 包裹与前后废话）。"""
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def build(corpus: str, per_paper: int, pairs: int, out: str) -> dict:
    cfg = Config.load(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    llm = build_llm(cfg)
    docs = load_directory(corpus)
    if len(docs) < 2:
        raise SystemExit(f"语料至少需要 2 篇，当前 {len(docs)} 篇")

    questions, qid = [], 0
    # ---- 单篇出题 ----
    for d in docs:
        content = d.full_text[:3000]
        raw = llm.chat([{"role": "user", "content": SINGLE_PROMPT.format(n=per_paper, content=content)}],
                       temperature=0.3)
        items = parse_json_array(raw)[:per_paper]
        for it in items:
            qid += 1
            questions.append({
                "id": qid,
                "type": it.get("type", "事实性"),
                "question": it.get("question", "").strip(),
                "answer": it.get("answer", "").strip(),
                "relevant_titles": [d.title],
            })
        print(f"  {d.title[:26]} -> {len(items)} 题")
        if not items:
            print(f"    [警告] 解析失败，原文前 120 字：{raw[:120]}")

    # ---- 对比题（跨文档）----
    for i in range(min(pairs, len(docs))):
        a, b = docs[i], docs[(i + 1) % len(docs)]
        raw = llm.chat([{"role": "user", "content": PAIR_PROMPT.format(
            a=a.full_text[:1200], b=b.full_text[:1200])}], temperature=0.3)
        for it in parse_json_array(raw)[:1]:
            qid += 1
            questions.append({
                "id": qid,
                "type": "对比性",
                "question": it.get("question", "").strip(),
                "answer": it.get("answer", "").strip(),
                "relevant_titles": [a.title, b.title],
            })
    print(f"  对比题 {min(pairs, len(docs))} 条")

    result = {
        "_说明": "由 build_evalset.py 自动生成。relevant_titles 用于关联文档，"
                 "评测时由 benchmark 按标题映射到 doc_id。",
        "_题型说明": {"事实性": "单篇可答", "归纳性": "跨篇归纳",
                      "推理性": "需推断原因", "对比性": "两篇对比"},
        "num_questions": len(questions),
        "questions": [q for q in questions if q["question"]],
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "评测集.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共生成 {result['num_questions']} 题 -> {Path(out) / '评测集.json'}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./data/sample_papers")
    ap.add_argument("--per-paper", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--out", default="./reports")
    a = ap.parse_args()
    build(a.corpus, a.per_paper, a.pairs, a.out)
