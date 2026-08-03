"""
评估流水线主模块

支持两种评估模式：
1. LLM 模式：调用 OpenAI 兼容 API（如 DeepSeek、GPT 等）进行自动评分
2. Mock 模式：使用预置的评分结果（基于领域专家分析），无需 API Key

使用方式：
    # Mock 模式（默认）
    python evaluator.py --mode mock

    # LLM 模式（需配置 API）
    python evaluator.py --mode llm --api-key YOUR_KEY --api-base https://api.deepseek.com/v1 --model deepseek-chat
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 将当前目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from metrics import (
    ALL_METRICS,
    calculate_weighted_score,
    get_metric_by_id,
)


# ============================================================
# Mock 评估数据（基于领域专家对 20 条 case 的逐条分析）
# ============================================================
# 评分依据：对比 auto_replies.json 与 human_ref.json 中的人工标注，
# 结合 eval_criteria.md 中的业务需求，逐条分析每条自动回复的质量。

MOCK_SCORES: list[dict[str, Any]] = [
    {
        "case_id": "case_01",
        "scores": {"M1": 4, "M2": 2, "M3": 3, "M4": 1, "M5": 2},
        "analysis": (
            "信息准确但把责任推给用户。用户核心诉求是'取不到快递'，需要的是"
            "主动帮助而非'自己去取/联系快递员'。人工回复会主动提出联系快递公司重新派送。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_02",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "用户问'这个'充电宝（指具体商品），自动回复只给了通用航空规定，"
            "没有查具体商品参数。人工回复直接告知该充电宝74Wh可带上飞机。"
            "典型的'正确但没用'。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_03",
        "scores": {"M1": 4, "M2": 3, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "退款时效信息准确，但用户问的是'我的退款'（具体订单），"
            "自动回复给的是通用说明。人工回复会先查订单再给具体信息。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_04",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "用户已明确说才买三天（在7天质保期内），自动回复却先给一堆排查步骤"
            "让用户自己试，增加了操作负担。应该直接给出退换货方案。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_05",
        "scores": {"M1": 4, "M2": 2, "M3": 3, "M4": 1, "M5": 2},
        "analysis": (
            "道歉了但没有立刻帮用户解决原始问题，还提到'加强客服培训'——"
            "这是内部事项，用户不关心。用户已经等了20分钟，需要的是立刻解决问题。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_06",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "罗列了可能原因但没有帮用户实际排查。人工回复会直接查用户的优惠券"
            "和订单状态来定位问题。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_07",
        "scores": {"M1": 4, "M2": 3, "M3": 3, "M4": 1, "M5": 2},
        "analysis": (
            "防诈骗提醒是好的，但用户最需要的是'帮我确认是不是真的'。"
            "异地登录场景下用户情绪是害怕的，自动回复的情绪安抚不够。"
            "人工回复会直接查账号记录给明确答案。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_08",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "用户问具体商品的具体问题，自动回复说'可以查看商品详情页'——"
            "用户就是因为不想自己翻详情页才来问的。人工回复直接告知TPU软胶材质。"
            "典型的'正确但没用'。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_09",
        "scores": {"M1": 5, "M2": 3, "M3": 4, "M4": 1, "M5": 3},
        "analysis": (
            "退货邮费规则说明正确，但没有追问用户的具体情况来给针对性回答。"
            "人工回复会在给出规则后追问'您的退货是哪种情况？'"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_10",
        "scores": {"M1": 5, "M2": 4, "M3": 4, "M4": 1, "M5": 3},
        "analysis": (
            "基本正确，给出了操作路径和退款时间。作为自动回复，给出明确操作路径"
            "是可接受的。人工回复会直接帮用户操作，差距不大。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_11",
        "scores": {"M1": 4, "M2": 3, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "给出了通用换货流程，但没有追问具体是哪两件商品、要换什么尺码。"
            "人工回复会确认具体信息后分别发起换货申请。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_12",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "罗列了可能原因但没有帮用户查实际物流状态。人工回复会先查物流"
            "再给具体信息，必要时主动帮用户联系快递。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_13",
        "scores": {"M1": 4, "M2": 1, "M3": 3, "M4": 1, "M5": 1},
        "analysis": (
            "用户明确说皮肤敏感，需要个性化关注。自动回复让用户自己去看详情页，"
            "没有体现对用户特殊需求的关心。人工回复会主动查询成分并提供个性化建议。"
            "严重缺乏有用性和主动服务意识。"
        ),
        "worst_case_reason": "最差Case之一：用户有特殊需求（皮肤敏感），回复完全忽视，只让用户自查",
    },
    {
        "case_id": "case_14",
        "scores": {"M1": 5, "M2": 4, "M3": 5, "M4": 1, "M5": 4},
        "analysis": (
            "建议类反馈处理得不错。表达了感谢并承诺反馈给产品团队。"
            "人工回复会更具体一些，但整体质量尚可。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_15",
        "scores": {"M1": 4, "M2": 3, "M3": 3, "M4": 1, "M5": 3},
        "analysis": (
            "用户强调'连续两次'收到坏商品，是需要特殊安抚的场景。自动回复虽然"
            "提到了补偿优惠券但语气和力度不够，没有充分回应用户的不满情绪。"
            "人工回复会给出更具体的补偿方案（50元优惠券）。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_16",
        "scores": {"M1": 4, "M2": 3, "M3": 4, "M4": 1, "M5": 3},
        "analysis": (
            "处理方式基本合理——追问使用场景和预算。但'查看商品详情页的用户评价'"
            "把用户推走了，应该主动帮用户做对比。人工回复会主动提出帮用户对比推荐。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_17",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "用户一条消息问了两个问题，自动回复虽然都回应了但让用户自己去查，"
            "没有主动帮用户处理。人工回复会同时处理多个问题。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_18",
        "scores": {"M1": 5, "M2": 4, "M3": 4, "M4": 1, "M5": 3},
        "analysis": (
            "基本正确，给出了质保期信息和解决方案。人工回复会先追问用户倾向"
            "哪种方案再操作，自动回复没有做这一步确认。整体质量尚可。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_19",
        "scores": {"M1": 4, "M2": 2, "M3": 4, "M4": 1, "M5": 2},
        "analysis": (
            "用户问的是具体商品的补货时间。自动回复没有查具体商品，给了通用建议。"
            "人工回复会实际查询补货计划并主动帮用户设置到货提醒。"
        ),
        "worst_case_reason": None,
    },
    {
        "case_id": "case_20",
        "scores": {"M1": 4, "M2": 1, "M3": 2, "M4": 1, "M5": 1},
        "analysis": (
            "用户说退货流程太复杂搞不懂，自动回复又重复了一遍退货流程——"
            "这恰恰是用户说搞不懂的东西。完全答非所问，加剧了用户的困惑和不满。"
            "人工回复会问具体卡在哪一步然后针对性帮忙。"
        ),
        "worst_case_reason": "最差Case之一：用户说搞不懂流程，回复又重复流程，完全答非所问",
    },
]


# ============================================================
# LLM 评估 Prompt 模板
# ============================================================
EVAL_PROMPT_TEMPLATE = """你是一个客服自动回复质量评估专家。请根据以下信息对自动回复进行评分。

## 业务评估标准
{eval_criteria}

## 评估指标
{metrics_description}

## 待评估的对话
用户问题：{user_question}
自动回复：{auto_reply}
人工参考回复：{human_reference}
人工标注分析：{annotator_notes}

## 评分要求
请对以下每个指标给出 1-5 的整数评分（M4无幻觉为0或1），并简要说明理由。
严格按照评分标准打分，重点关注自动回复与人工参考回复之间的差距。

请以 JSON 格式返回，不要包含其他内容：
```json
{{
  "M1": {{"score": <1-5>, "reason": "<准确性评分理由>"}},
  "M2": {{"score": <1-5>, "reason": "<有用性评分理由>"}},
  "M3": {{"score": <1-5>, "reason": "<语气与共情评分理由>"}},
  "M4": {{"score": <0或1>, "reason": "<无幻觉评分理由>"}},
  "M5": {{"score": <1-5>, "reason": "<主动服务评分理由>"}},
  "overall_analysis": "<整体分析>"
}}
```
"""


def build_metrics_description() -> str:
    """构建指标描述文本（用于 LLM Prompt）"""
    lines = []
    for m in ALL_METRICS:
        lines.append(f"\n### {m.id} {m.name}（{m.business_requirement}）")
        lines.append(f"量表：{m.scale}，权重：{m.weight}")
        lines.append(f"说明：{m.description}")
        lines.append("评分标准：")
        for score, desc in sorted(m.scoring_criteria.items()):
            lines.append(f"  {score}分：{desc}")
    return "\n".join(lines)


def call_llm_api(
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
) -> dict:
    """调用 OpenAI 兼容 API 进行评估"""
    import requests

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # 提取 JSON
    json_str = content
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    return json.loads(json_str.strip())


def evaluate_with_llm(
    auto_replies: list,
    human_refs: list,
    eval_criteria: str,
    api_key: str,
    api_base: str,
    model: str,
) -> list[dict]:
    """使用 LLM API 评估所有 case"""
    metrics_desc = build_metrics_description()
    human_ref_map = {h["id"]: h for h in human_refs}
    results = []

    for reply in auto_replies:
        case_id = reply["id"]
        human_ref = human_ref_map.get(case_id, {})

        prompt = EVAL_PROMPT_TEMPLATE.format(
            eval_criteria=eval_criteria,
            metrics_description=metrics_desc,
            user_question=reply["user_question"],
            auto_reply=reply["auto_reply"],
            human_reference=human_ref.get("human_reference", ""),
            annotator_notes=human_ref.get("annotator_notes", ""),
        )

        try:
            llm_result = call_llm_api(prompt, api_key, api_base, model)
            scores = {}
            analysis_parts = []
            for mid in ["M1", "M2", "M3", "M4", "M5"]:
                if mid in llm_result:
                    scores[mid] = llm_result[mid]["score"]
                    analysis_parts.append(f"{mid}: {llm_result[mid]['reason']}")

            results.append({
                "case_id": case_id,
                "scores": scores,
                "analysis": llm_result.get("overall_analysis", " | ".join(analysis_parts)),
                "worst_case_reason": None,
            })
            print(f"  [{case_id}] scored: {scores}")
        except Exception as e:
            print(f"  [{case_id}] LLM error: {e}, falling back to mock")
            # 回退到 mock
            mock = next(m for m in MOCK_SCORES if m["case_id"] == case_id)
            results.append(mock)

    return results


def evaluate_with_mock(auto_replies: list, human_refs: list) -> list[dict]:
    """使用 Mock 数据评估所有 case（无需 API）"""
    print("  使用 Mock 模式评估（基于领域专家预置分析）")
    mock_map = {m["case_id"]: m for m in MOCK_SCORES}
    results = []

    for reply in auto_replies:
        case_id = reply["id"]
        mock_data = mock_map.get(case_id)
        if mock_data:
            results.append(mock_data.copy())
            print(f"  [{case_id}] scored: {mock_data['scores']}")
        else:
            # 默认中性评分
            results.append({
                "case_id": case_id,
                "scores": {"M1": 3, "M2": 3, "M3": 3, "M4": 1, "M5": 3},
                "analysis": "默认评分（无匹配的 mock 数据）",
                "worst_case_reason": None,
            })

    return results


def validate_with_human_ref(
    results: list[dict],
    human_refs: list,
    auto_replies: list,
) -> dict:
    """
    用人工标注验证评估结果
    
    验证维度：
    1. 相关性：自动评分低的 case 是否在人工标注中也有问题
    2. 一致性：评分趋势是否与人工标注一致
    """
    human_map = {h["id"]: h for h in human_refs}
    reply_map = {r["id"]: r for r in auto_replies}

    validation = {
        "total_cases": len(results),
        "correlation_analysis": [],
        "agreement_rate": 0,
        "disagreement_cases": [],
    }

    agreements = 0
    for result in results:
        case_id = result["case_id"]
        human = human_map.get(case_id, {})
        reply = reply_map.get(case_id, {})
        weighted = calculate_weighted_score(result["scores"])

        # 人工标注中提到的问题关键词
        notes = human.get("annotator_notes", "").lower()
        has_issue = any(
            kw in notes
            for kw in ["没有", "推给用户", "通用", "正确但没用", "不够", "答非所问", "缺乏"]
        )

        # 自动评分低 (<60) 且人工标注有问题 → 一致
        # 自动评分高 (>=75) 且人工标注无明显问题 → 一致
        if weighted < 60 and has_issue:
            agreements += 1
            validation["correlation_analysis"].append({
                "case_id": case_id,
                "auto_score": weighted,
                "human_issue": has_issue,
                "status": "agree_low",
            })
        elif weighted >= 75 and not has_issue:
            agreements += 1
            validation["correlation_analysis"].append({
                "case_id": case_id,
                "auto_score": weighted,
                "human_issue": has_issue,
                "status": "agree_high",
            })
        else:
            validation["disagreement_cases"].append({
                "case_id": case_id,
                "auto_score": weighted,
                "human_notes": notes[:100],
                "status": "marginal",
            })

    validation["agreement_rate"] = round(agreements / len(results), 2)
    return validation


def run_evaluation(
    data_dir: str = "data",
    mode: str = "mock",
    api_key: str = "",
    api_base: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-chat",
) -> dict:
    """运行完整评估流水线"""
    data_path = Path(data_dir)

    # 加载数据
    print("1. 加载数据文件...")
    with open(data_path / "auto_replies.json", encoding="utf-8") as f:
        auto_replies = json.load(f)
    with open(data_path / "human_ref.json", encoding="utf-8") as f:
        human_refs = json.load(f)
    with open(data_path / "eval_criteria.md", encoding="utf-8") as f:
        eval_criteria = f.read()

    print(f"   - auto_replies.json: {len(auto_replies)} 条")
    print(f"   - human_ref.json: {len(human_refs)} 条")
    print(f"   - eval_criteria.md: {len(eval_criteria)} 字符")

    # 评估
    print(f"\n2. 执行评估（模式: {mode}）...")
    if mode == "llm" and api_key:
        results = evaluate_with_llm(
            auto_replies, human_refs, eval_criteria, api_key, api_base, model
        )
    else:
        results = evaluate_with_mock(auto_replies, human_refs)

    # 计算加权总分
    print("\n3. 计算加权总分...")
    for r in results:
        r["weighted_score"] = calculate_weighted_score(r["scores"])

    # 验证
    print("\n4. 与人工标注交叉验证...")
    validation = validate_with_human_ref(results, human_refs, auto_replies)
    print(f"   一致率: {validation['agreement_rate']}")
    print(f"   边缘case数: {len(validation['disagreement_cases'])}")

    # 排序
    results_sorted = sorted(results, key=lambda x: x["weighted_score"])

    # 找最差3条
    worst_3 = results_sorted[:3]

    return {
        "results": results,
        "results_sorted": results_sorted,
        "worst_3": worst_3,
        "validation": validation,
        "auto_replies": auto_replies,
        "human_refs": human_refs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动回复质量评估流水线")
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock",
                        help="评估模式：mock（默认）或 llm（需API Key）")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""),
                        help="LLM API Key")
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1",
                        help="LLM API Base URL")
    parser.add_argument("--model", default="deepseek-chat",
                        help="LLM 模型名称")
    parser.add_argument("--data-dir", default="data",
                        help="数据目录路径")

    args = parser.parse_args()

    evaluation = run_evaluation(
        data_dir=args.data_dir,
        mode=args.mode,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)

    all_scores = [r["weighted_score"] for r in evaluation["results"]]
    avg_score = sum(all_scores) / len(all_scores)
    print(f"\n整体平均分: {avg_score:.1f} / 100")
    print(f"最高分: {max(all_scores):.1f}")
    print(f"最低分: {min(all_scores):.1f}")

    print(f"\n最差 3 条 Case:")
    for i, case in enumerate(evaluation["worst_3"], 1):
        print(f"  {i}. {case['case_id']} (得分: {case['weighted_score']})")
        print(f"     {case['analysis'][:80]}...")

    print(f"\n人工标注验证一致率: {evaluation['validation']['agreement_rate']}")
