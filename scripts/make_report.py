# -*- coding: utf-8 -*-
"""
模块五产出：把评测数据渲染成 HTML 报告（内联 SVG 图表，零第三方依赖）
------------------------------------------------------------------------
为什么不装 matplotlib：matplotlib 与部分环境的 numpy 2.x 二进制不兼容，
且交付物需要一个"打开就能看、能直接贴进文档"的文件。
这里用纯 SVG 手绘柱状图，无依赖、矢量清晰、可直接截图进 PPT。

用法：python scripts/make_report.py
产出：reports/评测报告.html
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "reports" / "性能评测数据.json"
EVAL = ROOT / "reports" / "评测集.json"
OUT = ROOT / "reports" / "评测报告.html"

PALETTE = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]


def bar_chart(labels, series, height=280, unit="") -> str:
    """
    分组柱状图。
    labels: 分组名（x 轴，如三档）
    series: [{"name": "Hit@1", "values": [0.92, 0.94, 0.96]}, ...]
    """
    w, pad_l, pad_b, pad_t = 660, 52, 54, 26
    plot_h = height - pad_b - pad_t
    n_groups, n_series = len(labels), len(series)
    group_w = (w - pad_l - 16) / max(n_groups, 1)
    bar_w = min(26, group_w / (n_series + 1))

    vmax = max(max(s["values"]) for s in series) if series else 1.0
    vmax = min(1.0, vmax * 1.15) if vmax <= 1.0 else vmax * 1.15
    if vmax <= 0:
        vmax = 1.0

    def y_of(v):
        return pad_t + plot_h - (v / vmax) * plot_h

    parts = [f'<svg viewBox="0 0 {w} {height}" width="100%" xmlns="http://www.w3.org/2000/svg">']
    # 网格与 y 轴刻度
    for i in range(5):
        v = vmax * i / 4
        y = y_of(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-8}" y2="{y:.1f}" '
                     f'stroke="#E5E3DC" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="#888780" '
                     f'text-anchor="end">{v:.2f}</text>')
    # 柱
    for gi, label in enumerate(labels):
        gx = pad_l + gi * group_w
        for si, s in enumerate(series):
            v = s["values"][gi] if gi < len(s["values"]) else 0
            x = gx + 8 + si * bar_w
            y = y_of(v)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-4:.1f}" '
                         f'height="{pad_t+plot_h-y:.1f}" rx="3" fill="{PALETTE[si%len(PALETTE)]}"/>')
            parts.append(f'<text x="{x+bar_w/2-2:.1f}" y="{y-5:.1f}" font-size="10" '
                         f'fill="#5F5E5A" text-anchor="middle">{v:.3f}</text>')
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{pad_t+plot_h+20}" font-size="12" '
                     f'fill="#2C2C2A" text-anchor="middle">{label}</text>')
    # 图例
    lx = pad_l
    for si, s in enumerate(series):
        parts.append(f'<rect x="{lx}" y="{height-18}" width="12" height="12" rx="2" '
                     f'fill="{PALETTE[si%len(PALETTE)]}"/>')
        parts.append(f'<text x="{lx+17}" y="{height-8}" font-size="11" fill="#5F5E5A">'
                     f'{s["name"]}{unit}</text>')
        lx += 26 + len(s["name"]) * 11
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers, rows) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;
     max-width:860px;margin:32px auto;padding:0 20px;color:#2C2C2A;line-height:1.7;}
h1{font-size:22px;font-weight:500;border-bottom:2px solid #378ADD;padding-bottom:10px;}
h2{font-size:17px;font-weight:500;margin-top:32px;color:#185FA5;}
h3{font-size:14px;font-weight:500;margin-top:20px;color:#0F6E56;}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}
th,td{border:1px solid #D3D1C7;padding:7px 10px;text-align:left;}
th{background:#F1EFE8;font-weight:500;}
tr:nth-child(even) td{background:#FAFAF8;}
.card{border:1px solid #E5E3DC;border-radius:10px;padding:18px;margin:16px 0;background:#fff;}
.note{background:#FAEEDA;border-left:3px solid #EF9F27;padding:10px 14px;
      font-size:13px;margin:12px 0;border-radius:0 6px 6px 0;}
.bad{background:#FCEBEB;border-left:3px solid #E24B4A;padding:10px 14px;
     font-size:13px;margin:8px 0;border-radius:0 6px 6px 0;}
code{background:#F1EFE8;padding:1px 5px;border-radius:3px;font-size:12px;}
"""


def build() -> str:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    ev = json.loads(EVAL.read_text(encoding="utf-8")) if EVAL.exists() else {}
    rows = data["retrieval"]

    # 三档对比图
    labels = [r["档位"] for r in rows]
    metrics = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    series = [{"name": m, "values": [r.get(m, 0) for r in rows]} for m in metrics]
    chart1 = bar_chart(labels, series)

    # 分块对比表（按 512 一组展示）
    ck = [c for c in data["chunking"] if c.get("chunk_size") == data.get("chunk_size", 256)]
    ck = ck or data["chunking"]
    table_ck = table(
        ["策略", "块数", "平均块长", "最大块长", "最短块长", "标准差", "过短块占比"],
        [[c["strategy"], c["num_chunks"], c["avg_len"], c["max_len"],
          c["min_len"], c["std_len"], f"{c['short_ratio']:.1%}"] for c in ck])

    # 分块策略 × chunk_size 的块数图
    sizes = sorted({c["chunk_size"] for c in data["chunking"]})
    strats = sorted({c["strategy"] for c in data["chunking"]})
    series2 = [{"name": s, "values": [
        next((c["num_chunks"] for c in data["chunking"]
              if c["strategy"] == s and c["chunk_size"] == sz), 0) for sz in sizes]}
        for s in strats]
    chart2 = bar_chart([str(sz) for sz in sizes], series2, height=260, unit=" 块")

    bad = data.get("bad_cases", [])
    bad_html = "".join(
        f'<div class="bad"><b>[{b.get("type","")}]</b> {b["question"]}'
        f'<br><span style="color:#888780">Top1 得分 {b.get("top1_score",0)}</span></div>'
        for b in bad) or '<div class="note">本轮评测未出现未命中案例。</div>'

    from collections import Counter
    type_dist = Counter(q.get("type", "") for q in ev.get("questions", []))

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>智能科研助理 · 检索性能评测报告</title><style>{CSS}</style></head><body>
<h1>智能科研助理 · 检索性能评测报告</h1>
<p style="color:#5F5E5A;font-size:13px">
语料：{data['corpus']}（{data['num_docs']} 篇）｜ 评测集：{data['eval_size']} 题 ｜
评测分块大小：{data.get('chunk_size', 256)}</p>

<h2>一、检索策略三档对比</h2>
<div class="card">{chart1}</div>
{table(["档位"] + metrics + ["块数"],
       [[r["档位"], r.get("Hit@1",0), r.get("Hit@3",0), r.get("Hit@5",0), r["MRR"], r["块数"]]
        for r in rows])}
<div class="note">
<b>结论</b>：混合检索（BM25 + 向量 + RRF）相比纯向量检索，Hit@1 提升
{round((rows[1].get('Hit@1',0)-rows[0].get('Hit@1',0))*100,1)} 个百分点；
再叠加在线重排序后，Hit@1 达 {rows[2].get('Hit@1',0)}、MRR 达 {rows[2]['MRR']}。
Hit@5 已接近饱和（{rows[2].get('Hit@5',0)}），说明本系统使用的
qwen3.7-text-embedding 语义表征能力较强，Top-5 宽松召回几乎必然命中，
因此<b>以 Hit@1 与 MRR 作为主要区分指标</b>。
</div>

<h2>二、分块策略对比实验</h2>
<div class="card">{chart2}</div>
{table_ck}
<div class="note">
递归切分（recursive）在相同 chunk_size 下产生的块长最均匀、标准差最小，
且能保留段落与句子边界；固定切分（fixed）速度快但会在句中切断；
语义切分（semantic）块内连贯性最好但耗时更高；
学术切分（academic）额外保护公式与表格块。
</div>

<h2>三、评测集构成</h2>
{table(["题型", "题数"], [[k, v] for k, v in type_dist.items()])}

<h2>四、Bad Case 分析</h2>
<p style="font-size:13px;color:#5F5E5A">
未命中（Hit@5 = 0）的案例共 {len(bad)} 条，占评测集
{len(bad)/max(data['eval_size'],1):.1%}：</p>
{bad_html}

</body></html>"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"报告已生成: {OUT}")
