from __future__ import annotations

import argparse

from .common import load_config
from .load import load_analysis_data
from .report import build_report
from .report_eda2 import build_eda2_report
from .report_eda3 import build_eda3_report
from . import (
    step0_profile,
    step1_targets,
    step2_segments,
    step3_numeric,
    step4_season,
    step5_food_groups,
    step6_sources_countries,
    step7_group_season,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="디노테푸란 통합원장 EDA")
    parser.add_argument(
        "--step",
        choices=["all", "0", "1", "2", "3", "4", "5", "6", "7"],
        default="all",
        help="실행 단계",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Parquet 캐시를 무시하고 CSV를 다시 읽음",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    df, source_path = load_analysis_data(force=args.force)
    print(f"[로드] {len(df):,}건 · {source_path}")

    selected = {"0", "1", "2", "3", "4", "5", "6", "7"} if args.step == "all" else {args.step}
    outputs = {}
    if "0" in selected:
        outputs.update(step0_profile.run(df, config))
        print("[완료] 0단계 구조·품질 점검")
    if "1" in selected:
        outputs.update(step1_targets.run(df, config))
        print("[완료] 1단계 목표 라벨 분석")
    if "2" in selected:
        outputs.update(step2_segments.run(df, config))
        print("[완료] 2단계 품목·국가·지역 분석")
    if "3" in selected:
        outputs.update(step3_numeric.run(df, config))
        print("[완료] 3단계 결과값·MRL 분석")
    if "4" in selected:
        outputs.update(step4_season.run(df, config))
        print("[완료] 4단계 계절·월별 분석")
    if "5" in selected:
        outputs.update(step5_food_groups.run(df, config))
        print("[완료] 5단계 원천 품목군·세부 품목 분석")
    if "6" in selected:
        outputs.update(step6_sources_countries.run(df, config))
        print("[완료] 6단계 출처·연도·원산국 분석")
    if "7" in selected:
        outputs.update(step7_group_season.run(df, config))
        print("[완료] 7단계 품목군×계절 분석")

    if args.step == "all":
        report_path = build_report(df, outputs, source_path, config)
        print(f"[완료] 기록 문서 · {report_path}")
        report_eda2_path = build_eda2_report(df, outputs, source_path, config)
        print(f"[완료] 2차 기록 문서 · {report_eda2_path}")
        report_eda3_path = build_eda3_report(df, outputs, source_path, config)
        print(f"[완료] 3차 기록 문서 · {report_eda3_path}")


if __name__ == "__main__":
    main()
