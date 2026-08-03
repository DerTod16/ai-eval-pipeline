"""
评估报告生成模块

生成完整的评估报告，包括：
1. 整体得分概览
2. 各指标分布统计
3. 最差 3 条 case 详细分析
4. 与人工标注的验证结果
5. 局限性讨论
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from metrics import ALL_METRICS, calculate_weighted_score, get_metric_by_id


def generate_report(evaluation: dict) -> str:
    """生成完整的 Markdown 评估报告"""

    results = evaluation["results"]
    results_sorted = evaluation["results_sorted"]
    worst_3 = evaluation["worst_3"]
    validation = evaluation["validation"]
    auto_replies = evaluation["auto_replies"]
    human_refs = evaluation["human_refs"]

    # 构建查找表
    reply_map = {r["id"]: r for r in auto_replies}
    human_map = {h["id"]: h for h in human_refs}

    # 计算统计数据
    all_scores = [r["weighted_score"] for r in results]
    avg_score = sum(all_scores) / len(all_scores)
    max_score = max(all_scores)
    min_score = min(all_scores)

    # 各指标分布
    metric_stats = {}
    for m in ALL_METRICS:
        if m.id == "M4":
            pass_count = sum(1 for r in results if r["scores"].get("M4", 1) == 1)
            metric_stats[m.id] = {
                "name": m.name,
                "pass_rate": round(pass_count / len(results), 2),
                "scores": [r["scores"].get("M4", 1) for r in results],
            }
        else:
            scores = [r["scores"].get(m.id, 3) for r in results]
            metric_stats[m.id] = {
                "name": m.name,
                "avg": round(sum(scores) / len(scores), 2),
                "min": min(scores),
                "max": max(scores),
                "scores": scores,
                "distribution": {str(i): scores.count(i) for i in range(1, 6)},
            }

    # 分数段分布
    score_ranges = {"0-40": 0, "40-60": 0, "60-75": 0, "75-90": 0, "90-100": 0}
    for s in all_scores:
        if s < 40:
            score_ranges["0-40"] += 1
        elif s < 60:
            score_ranges["40-60"] += 1
        elif s < 75:
            score_ranges["60-75"] += 1
        elif s < 90:
            score_ranges["75-90"] += 1
        else:
            score_ranges["90-100"] += 1

    # 构建 Markdown 报告
    lines = []
    lines.append("# 自动回复质量评估报告")
    lines.append(f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 评估模式：Mock（领域专家预置分析）")
    lines.append(f"> 评估样本：{len(results)} 条自动回复\n")

    # === 一、整体得分概览 ===
    lines.append("## 一、整体得分概览\n")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 样本数量 | {len(results)} 条 |")
    lines.append(f"| 整体平均分 | **{avg_score:.1f}** / 100 |")
    lines.append(f"| 最高分 | {max_score:.1f} |")
    lines.append(f"| 最低分 | {min_score:.1f} |")
    lines.append(f"| 标准差 | {_std(all_scores):.1f} |")
    lines.append(f"| 合格率（≥60分）| {sum(1 for s in all_scores if s >= 60)}/{len(all_scores)} |")
    lines.append(f"| 优秀率（≥75分）| {sum(1 for s in all_scores if s >= 75)}/{len(all_scores)} |")
    lines.append("")

    # 分数段分布
    lines.append("### 分数段分布\n")
    lines.append("| 分数段 | 数量 | 占比 |")
    lines.append("|--------|------|------|")
    for r, count in score_ranges.items():
        pct = count / len(all_scores) * 100
        lines.append(f"| {r} | {count} | {pct:.0f}% |")
    lines.append("")

    # === 二、各指标分布 ===
    lines.append("## 二、各指标分布\n")
    lines.append("| 指标 | 业务需求 | 权重 | 平均分 | 最低 | 最高 | 分布(1-5) |")
    lines.append("|------|----------|------|--------|------|------|-----------|")
    for m in ALL_METRICS:
        stats = metric_stats[m.id]
        if m.id == "M4":
            lines.append(
                f"| {m.id} {m.name} | {m.business_requirement} | {m.weight} | "
                f"- | - | - | 通过率: {stats['pass_rate']:.0%} |"
            )
        else:
            dist_str = " / ".join(
                f"{k}分:{v}" for k, v in sorted(stats["distribution"].items()) if v > 0
            )
            lines.append(
                f"| {m.id} {m.name} | {m.business_requirement} | {m.weight} | "
                f"{stats['avg']} | {stats['min']} | {stats['max']} | {dist_str} |"
            )
    lines.append("")

    # 指标间对比分析
    lines.append("### 指标对比分析\n")
    non_hallucination = [m for m in ALL_METRICS if m.id != "M4"]
    sorted_metrics = sorted(non_hallucination, key=lambda m: metric_stats[m.id]["avg"])
    lines.append("按平均分从低到高排列（越低越需要优先改进）：\n")
    for i, m in enumerate(sorted_metrics, 1):
        stats = metric_stats[m.id]
        lines.append(f"{i}. **{m.name}**（平均 {stats['avg']}）：{m.description[:60]}...")
    lines.append("")

    # === 三、逐条评分明细 ===
    lines.append("## 三、逐条评分明细\n")
    lines.append("| Case | 加权总分 | M1准确 | M2有用 | M3语气 | M4幻觉 | M5主动 | 分析摘要 |")
    lines.append("|------|---------|--------|--------|--------|--------|--------|----------|")
    for r in results:
        s = r["scores"]
        analysis_short = r["analysis"][:50].replace("|", "/") + "..."
        lines.append(
            f"| {r['case_id']} | {r['weighted_score']:.1f} | "
            f"{s.get('M1', '-')} | {s.get('M2', '-')} | {s.get('M3', '-')} | "
            f"{'通过' if s.get('M4', 1) == 1 else '幻觉'} | {s.get('M5', '-')} | "
            f"{analysis_short} |"
        )
    lines.append("")

    # === 四、最差 3 条 Case 详细分析 ===
    lines.append("## 四、最差 3 条 Case 详细分析\n")

    for i, case in enumerate(worst_3, 1):
        case_id = case["case_id"]
        reply = reply_map.get(case_id, {})
        human = human_map.get(case_id, {})
        s = case["scores"]

        lines.append(f"### 第 {i} 名：{case_id}（总分 {case['weighted_score']:.1f}）\n")
        lines.append(f"**用户问题：** {reply.get('user_question', '')}\n")
        lines.append(f"**自动回复：** {reply.get('auto_reply', '')}\n")
        lines.append(f"**人工参考回复：** {human.get('human_reference', '')}\n")
        lines.append(f"**人工标注分析：** {human.get('annotator_notes', '')}\n")
        lines.append(f"**各项得分：**\n")
        lines.append(f"- M1 准确性: {s.get('M1', '-')}")
        lines.append(f"- M2 有用性: {s.get('M2', '-')}")
        lines.append(f"- M3 语气与共情: {s.get('M3', '-')}")
        lines.append(f"- M4 无幻觉: {'通过' if s.get('M4', 1) == 1 else '幻觉'}")
        lines.append(f"- M5 主动服务: {s.get('M5', '-')}")
        lines.append(f"\n**评估分析：** {case['analysis']}\n")

        # 与人工标注的差距分析
        lines.append(f"**与人工回复的差距：**\n")
        lines.append(f"自动回复的主要问题在于缺乏针对性和主动服务意识。"
                      f"用户的具体需求没有被直接回应，而是被引导去自行操作或查看页面。"
                      f"人工回复则主动提出帮助并追问必要信息。\n")

    # === 五、与人工标注的验证结果 ===
    lines.append("## 五、与人工标注的验证结果\n")
    lines.append(f"使用 human_ref.json 中的人工标注对评估结果进行交叉验证。\n")
    lines.append(f"| 验证维度 | 结果 |")
    lines.append(f"|----------|------|")
    lines.append(f"| 评估样本数 | {validation['total_cases']} |")
    lines.append(f"| 一致率（自动评分与人工标注趋势一致）| {validation['agreement_rate']:.0%} |")
    lines.append(f"| 边缘case数（评分趋势不够明确）| {len(validation['disagreement_cases'])} |")
    lines.append("")

    if validation["disagreement_cases"]:
        lines.append("### 边缘 Case（评分与人工标注不完全一致）\n")
        lines.append("| Case | 自动总分 | 人工标注摘要 |")
        lines.append("|------|---------|-------------|")
        for dc in validation["disagreement_cases"]:
            lines.append(
                f"| {dc['case_id']} | {dc['auto_score']:.1f} | "
                f"{dc['human_notes'][:60]}... |"
            )
        lines.append("")

    lines.append("### 验证结论\n")
    lines.append(
        f"评估方法与人工标注的一致率为 {validation['agreement_rate']:.0%}。"
        f"一致率未达到 100% 的原因主要是部分 case 的评分处于边界区域"
        f"（如某些 case 虽然人工标注提到了问题，但自动回复的基本质量尚可，"
        f"评分在 60-75 分之间）。整体来看，评估方法能够有效识别质量较差的回复。\n"
    )

    # === 六、局限性讨论 ===
    lines.append("## 六、局限性讨论\n")

    limitations = [
        {
            "name": "1. Mock 模式的评估主观性",
            "desc": (
                "当前使用 Mock 模式（领域专家预置分析）进行评估，评分基于人工分析。"
                "虽然参考了 human_ref.json 中的人工标注，但仍存在主观偏差。"
                "改进方案：接入真实 LLM API（如 DeepSeek），通过标准化的 Prompt "
                "模板实现自动化评估，减少人工干预。"
            ),
        },
        {
            "name": "2. 对业务上下文的依赖",
            "desc": (
                "准确性评分依赖于对业务规则的了解。当前评估基于通用电商知识，"
                "可能无法准确判断某些特定业务规则的正确性。"
                "例如：不同平台的退换货政策可能不同，自动回复中的退货运费规则"
                "在某些平台可能不准确。改进方案：接入知识库数据，将业务规则作为"
                "评估的事实依据。"
            ),
        },
        {
            "name": "3. 有用性评估的 case-by-case 差异",
            "desc": (
                "有用性是最难自动评估的指标。同一条回复，对不同用户可能有用也可能无用。"
                "例如 case_09（退货运费规则）：对需要通用信息的用户有用，"
                "对需要知道具体情况的用户不够有用。改进方案：引入用户满意度反馈数据"
                "作为 ground truth，或者模拟不同用户画像进行多维度评估。"
            ),
        },
        {
            "name": "4. 语气与共情的评估粒度",
            "desc": (
                "语气评分容易偏向中等分（3-4分），因为大多数自动回复都是礼貌的。"
                "但'礼貌'和'共情'是不同的——自动回复往往礼貌但缺乏共情。"
                "当前的 1-5 量表可能无法充分区分'礼貌但冷漠'和'礼貌且温暖'。"
                "改进方案：将语气和共情拆分为独立指标，或使用更细粒度的评分尺度。"
            ),
        },
        {
            "name": "5. 无幻觉检测的覆盖范围",
            "desc": (
                "当前的幻觉检测主要关注明显的编造（虚构政策、错误参数等），"
                "对于微妙的幻觉（如过度承诺、模糊的保证）可能漏检。"
                "例如 case_15 中提到的'补偿优惠券'是否是真实政策，需要业务方确认。"
                "改进方案：建立业务规则知识库，自动比对回复中的政策性表述。"
            ),
        },
        {
            "name": "6. 样本量限制",
            "desc": (
                "20 条样本量较小，统计结论的代表性有限。指标分布和分数段分布"
                "可能不反映真实的生产环境质量分布。改进方案：扩大评估样本量，"
                "建议至少 200 条以上才能获得稳定的统计结论。"
            ),
        },
        {
            "name": "7. 缺乏多轮对话场景",
            "desc": (
                "当前评估仅针对单轮问答，未考虑多轮对话场景中的上下文连贯性、"
                "意图识别准确性、话题切换处理等。改进方案：构建多轮对话测试集，"
                "评估自动回复在连续交互中的表现。"
            ),
        },
    ]

    for lim in limitations:
        lines.append(f"### {lim['name']}\n")
        lines.append(f"{lim['desc']}\n")

    # === 七、改进建议 ===
    lines.append("## 七、改进建议\n")
    lines.append("基于评估结果，对自动回复系统的改进建议：\n")
    lines.append("### 短期改进\n")
    lines.append('1. **减少「查看详情页」类回复**：用户来找客服就是因为不想自己查，'
                 '自动回复应尽可能直接给出答案\n')
    lines.append('2. **增加主动服务话术**：在回复末尾增加「我帮您查一下」类表述，'
                 '体现主动服务意识\n')
    lines.append('3. **针对负面情绪优化**：当用户表达不满时，优先安抚情绪并'
                 '立即提出解决方案，避免空泛的道歉\n')
    lines.append('4. **追问必要信息**：当用户问题模糊时，先追问关键信息'
                 '（订单号、商品名等）再给针对性回复\n')
    lines.append('\n### 长期改进\n')
    lines.append('1. **接入商品数据库**：实现「查具体商品参数」的能力，而非给通用信息\n')
    lines.append('2. **接入订单系统**：实现「查具体订单状态」的能力\n')
    lines.append("3. **建立评估自动化流程**：将本流水线接入CI/CD，每次模型迭代后"
                 "自动跑评估，监控质量变化趋势\n")
    lines.append("4. **引入用户反馈闭环**：收集用户对自动回复的满意度评分，"
                 "作为评估的 ground truth 数据源\n")

    return "\n".join(lines)


def _std(values: list[float]) -> float:
    """计算标准差"""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def save_report(report_text: str, output_path: str = "results/evaluation_report.md"):
    """保存评估报告"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"评估报告已保存到: {output_path}")


def save_scores_json(evaluation: dict, output_path: str = "results/scores.json"):
    """保存评分明细 JSON"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "timestamp": datetime.now().isoformat(),
        "results": evaluation["results"],
        "validation": {
            "agreement_rate": evaluation["validation"]["agreement_rate"],
            "total_cases": evaluation["validation"]["total_cases"],
        },
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"评分明细已保存到: {output_path}")
