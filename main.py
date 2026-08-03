#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动回复质量评估流水线 - 主入口

使用方式:
    python main.py                          # Mock 模式（默认）
    python main.py --mode llm --api-key XXX # LLM 模式
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import run_evaluation
from report import generate_report, save_report, save_scores_json


def main():
    parser = argparse.ArgumentParser(
        description="自动回复质量评估流水线 - 晓多AI测评 Task 0109"
    )
    parser.add_argument(
        "--mode", choices=["mock", "llm"], default="mock",
        help="评估模式: mock=使用预置分析(默认), llm=调用LLM API"
    )
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--api-base", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    # 确定项目根目录
    project_root = Path(__file__).parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir

    print("=" * 60)
    print("  自动回复质量评估流水线")
    print("  Task 0109 - 晓多AI测评")
    print("=" * 60)

    # 运行评估
    evaluation = run_evaluation(
        data_dir=str(data_dir),
        mode=args.mode,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model,
    )

    # 生成报告
    print("\n5. 生成评估报告...")
    report_text = generate_report(evaluation)

    # 保存结果
    save_report(report_text, str(output_dir / "evaluation_report.md"))
    save_scores_json(evaluation, str(output_dir / "scores.json"))

    # 打印最终摘要
    print("\n" + "=" * 60)
    print("  评估完成!")
    print("=" * 60)

    all_scores = [r["weighted_score"] for r in evaluation["results"]]
    avg = sum(all_scores) / len(all_scores)

    print(f"\n  整体平均分: {avg:.1f} / 100")
    print(f"  最高分: {max(all_scores):.1f}")
    print(f"  最低分: {min(all_scores):.1f}")
    print(f"  人工标注验证一致率: {evaluation['validation']['agreement_rate']:.0%}")

    print(f"\n  报告文件: {output_dir / 'evaluation_report.md'}")
    print(f"  评分明细: {output_dir / 'scores.json'}")

    # 最差3条
    print(f"\n  最差 3 条 Case:")
    for i, case in enumerate(evaluation["worst_3"], 1):
        print(f"    {i}. {case['case_id']} (得分: {case['weighted_score']:.1f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
