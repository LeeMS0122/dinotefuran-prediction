"""Fit-on-train feature standardization and enrichment for dinotefuran."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TARGETS = ("occurrence", "screening", "noncompliance")
SPLITS = ("historical_excluded", "train", "validation", "test", "oot_monitor")


GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("비식품·환경시료", ("토양", "환경시료", "전체")),
    ("가공식품", ("가공", "면류", "밀가루", "음료", "고춧가루", "농림가공")),
    ("차·커피", ("다류", "커피", "차류")),
    ("버섯류", ("버섯",)),
    ("채소류", ("채류", "과채", "엽", "근채", "조미채소", "양채", "산채", "고추", "무(", "호박", "가지", "상추", "브로콜리", "양파", "마늘", "부추", "아스파라거스", "오크라")),
    ("두류", ("두류", "콩", "팥", "녹두", "대두", "완두")),
    ("서류", ("서류", "감자", "고구마", "토란", "얌", "카사바", "마(")),
    ("곡류", ("곡류", "미곡", "쌀", "보리", "옥수수", "메밀", "조", "율무", "귀리", "기장")),
    ("견과·종실류", ("견과", "종실", "땅콩", "참깨", "들깨", "호두", "아몬드", "씨", "피마자", "유채", "브라질넛", "마카다미아", "잣", "케슈너트")),
    ("과실류", ("과실", "과일", "장과", "인과", "핵과", "감귤", "열대과일", "베리", "수실", "포도", "감", "대추", "오미자", "구기자", "복분자", "무화과", "오디", "리치", "람부탄", "커런트")),
    ("향신·약용작물", ("향신", "허브", "약용", "인삼", "수삼", "정향", "생강", "서양박하", "둥굴레", "호프", "스테비아")),
    ("수산·해조류", ("미역", "꼬시래기")),
    ("기타 식물성", ("기타식물", "농산물 종자", "특용작물", "조사료", "초화류")),
    ("기타 식품", ("벌꿀", "기타식품", "기타기준규격외")),
]


NAME_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("채소류", ("파프리카", "상추", "오이", "깻잎", "시금치", "참외", "수박", "가지", "토마토", "당근", "고추", "배추", "호박", "대파", "양파", "무", "브로콜리", "부추", "마늘", "열무", "양배추", "셀러리")),
    ("과실류", ("사과", "포도", "단감", "떫은감", "망고", "복숭아", "바나나", "배", "딸기", "귤", "감귤", "체리", "블루베리", "자두", "매실", "키위", "레몬", "오렌지", "파인애플")),
    ("버섯류", ("버섯", "표고", "느타리", "팽이", "새송이", "목이")),
    ("곡류", ("쌀", "보리", "옥수수", "밀", "메밀", "귀리", "기장", "율무")),
    ("두류", ("콩", "팥", "녹두", "완두")),
    ("서류", ("감자", "고구마", "토란", "얌", "카사바")),
    ("견과·종실류", ("땅콩", "참깨", "들깨", "호두", "아몬드", "잣", "해바라기씨", "호박씨")),
    ("차·커피", ("차", "티", "커피", "루이보스", "캐모마일", "페퍼민트")),
    ("향신·약용작물", ("인삼", "생강", "계피", "정향", "허브", "바질", "로즈마리", "월계수")),
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def normalize_scalar(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text or None


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_scalar).astype("string")


def clean_product_name(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    normalized = normalize_series(series)
    composite = normalized.fillna("").str.count(r"[,;]").gt(0).astype("int8")
    cleaned = normalized.str.replace(r"\([^()]*\)", " ", regex=True)
    cleaned = cleaned.str.replace(r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|개|입|팩)\b", " ", regex=True, flags=re.IGNORECASE)
    cleaned = cleaned.str.replace(r"[^0-9A-Za-z가-힣]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA).astype("string")
    return cleaned, composite


def normalize_group(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    out = normalize_series(series).str.replace(r"\s+", "", regex=True)
    alias_norm = {re.sub(r"\s+", "", normalize_scalar(k) or ""): v for k, v in aliases.items()}
    return out.replace(alias_norm).astype("string")


def classify_text(value: object, rules: list[tuple[str, tuple[str, ...]]]) -> str | None:
    text = normalize_scalar(value)
    if not text:
        return None
    compact = text.replace(" ", "")
    for label, keywords in rules:
        if any(keyword.replace(" ", "") in compact for keyword in keywords):
            return label
    return None


def _stable_mapping(frame: pd.DataFrame, key_col: str, value_col: str,
                    min_support: int, min_purity: float) -> tuple[dict[str, str], pd.DataFrame]:
    valid = frame[[key_col, value_col]].dropna().copy()
    if valid.empty:
        empty = pd.DataFrame(columns=[key_col, "mapped_value", "support", "top_count", "purity"])
        return {}, empty
    counts = valid.groupby([key_col, value_col], observed=True).size().rename("n").reset_index()
    totals = counts.groupby(key_col, observed=True)["n"].sum().rename("support")
    top = counts.sort_values([key_col, "n", value_col], ascending=[True, False, True]).drop_duplicates(key_col)
    top = top.rename(columns={value_col: "mapped_value", "n": "top_count"}).merge(totals, on=key_col)
    top["purity"] = top["top_count"] / top["support"]
    accepted = top.loc[(top["support"] >= min_support) & (top["purity"] >= min_purity)].copy()
    mapping = dict(zip(accepted[key_col].astype(str), accepted["mapped_value"].astype(str)))
    return mapping, accepted


def _valid_iso2(series: pd.Series) -> pd.Series:
    out = normalize_series(series).str.upper()
    return out.where(out.str.fullmatch(r"[A-Z]{2}", na=False))


def _normalize_province(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    out = normalize_series(series)
    return out.replace(aliases).astype("string")


@dataclass
class FoodMappings:
    code_l1: dict[str, str]
    code_l2: dict[str, str]
    name_l1: dict[str, str]
    name_l2: dict[str, str]
    details: pd.DataFrame


def fit_food_mappings(train: pd.DataFrame, config: dict[str, Any]) -> FoodMappings:
    cfg = config["food_group_mapping"]
    work = train.copy()
    work["product_code_key"] = normalize_series(work["product_code"])
    work["product_name_key_v1"], _ = clean_product_name(work["product_name_std"])
    work["food_group_l2_raw_std"] = normalize_group(work["product_group_raw"], cfg["l2_aliases"])
    work["food_group_l1_raw_std"] = work["food_group_l2_raw_std"].map(lambda x: classify_text(x, GROUP_RULES)).astype("string")
    tables = []
    mappings: dict[str, dict[str, str]] = {}
    for map_name, key_col, value_col in [
        ("code_l1", "product_code_key", "food_group_l1_raw_std"),
        ("code_l2", "product_code_key", "food_group_l2_raw_std"),
        ("name_l1", "product_name_key_v1", "food_group_l1_raw_std"),
        ("name_l2", "product_name_key_v1", "food_group_l2_raw_std"),
    ]:
        mapping, detail = _stable_mapping(work, key_col, value_col, int(cfg["min_support"]), float(cfg["min_purity"]))
        detail.insert(0, "mapping_type", map_name)
        detail = detail.rename(columns={key_col: "mapping_key"})
        tables.append(detail)
        mappings[map_name] = mapping
    return FoodMappings(mappings["code_l1"], mappings["code_l2"], mappings["name_l1"], mappings["name_l2"], pd.concat(tables, ignore_index=True))


def transform_features(frame: pd.DataFrame, mappings: FoodMappings, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    food_cfg = config["food_group_mapping"]
    unknown = food_cfg["unclassified_label"]
    out["product_code_key"] = normalize_series(out["product_code"])
    out["product_name_key_v1"], out["product_is_composite_v1"] = clean_product_name(out["product_name_std"])
    raw_l2 = normalize_group(out["product_group_raw"], food_cfg["l2_aliases"])
    raw_l1 = raw_l2.map(lambda x: classify_text(x, GROUP_RULES)).astype("string")
    code_l2 = out["product_code_key"].map(mappings.code_l2).astype("string")
    name_l2 = out["product_name_key_v1"].map(mappings.name_l2).astype("string")
    code_l1 = out["product_code_key"].map(mappings.code_l1).astype("string")
    name_l1 = out["product_name_key_v1"].map(mappings.name_l1).astype("string")
    keyword_l1 = out["product_name_key_v1"].map(lambda x: classify_text(x, NAME_RULES)).astype("string")

    out["food_group_l2_std"] = raw_l2.fillna(code_l2).fillna(name_l2).fillna(unknown).astype("string")
    out["food_group_l1_std"] = raw_l1.fillna(code_l1).fillna(name_l1).fillna(keyword_l1).fillna(unknown).astype("string")
    source = pd.Series("unclassified", index=out.index, dtype="string")
    source = source.mask(raw_l1.notna(), "raw_group")
    source = source.mask(raw_l1.isna() & code_l1.notna(), "train_code_map")
    source = source.mask(raw_l1.isna() & code_l1.isna() & name_l1.notna(), "train_name_map")
    source = source.mask(raw_l1.isna() & code_l1.isna() & name_l1.isna() & keyword_l1.notna(), "name_keyword")
    out["food_group_mapping_source"] = source

    province_aliases = config["province_aliases"]
    collection_province = _normalize_province(out["collection_province"], province_aliases)
    cultivation_province = _normalize_province(out["cultivation_province"], province_aliases)
    out["domestic_province_std"] = cultivation_province.fillna(collection_province).astype("string")
    location_basis = pd.Series("unknown", index=out.index, dtype="string")
    location_basis = location_basis.mask(collection_province.notna(), "collection_province")
    location_basis = location_basis.mask(cultivation_province.notna(), "cultivation_province")
    out["domestic_location_basis"] = location_basis
    out["domestic_macro_region"] = out["domestic_province_std"].map(config["province_macro_region"]).fillna("미상").astype("string")

    production = _valid_iso2(out["production_country_code"])
    origin_code = _valid_iso2(out["origin_country_code"])
    origin_name = normalize_series(out["origin_country_name"]).map(config["country_name_to_iso2"]).astype("string")
    import_code = _valid_iso2(out["import_country_code"])
    iso2 = production.fillna(origin_code).fillna(origin_name).fillna(import_code)
    out["origin_iso2_std"] = iso2.fillna("ZZ").astype("string")
    country_source = pd.Series("unknown", index=out.index, dtype="string")
    country_source = country_source.mask(import_code.notna(), "import_country_code")
    country_source = country_source.mask(origin_name.notna(), "origin_country_name")
    country_source = country_source.mask(origin_code.notna(), "origin_country_code")
    country_source = country_source.mask(production.notna(), "production_country_code")
    out["origin_country_mapping_source"] = country_source
    out["origin_region_std"] = out["origin_iso2_std"].map(config["iso2_region"]).fillna("미상").astype("string")
    route = pd.Series("불명", index=out.index, dtype="string")
    route = route.mask(out["domestic_province_std"].notna(), "국내")
    route = route.mask(out["origin_iso2_std"].ne("ZZ") & out["origin_iso2_std"].ne("KR"), "수입")
    route = route.mask(out["origin_iso2_std"].eq("KR"), "국내")
    out["supply_route_v1"] = route

    event_date = pd.to_datetime(out["event_date"], errors="coerce")
    day = event_date.dt.dayofyear.astype(float)
    week = event_date.dt.isocalendar().week.astype("Int16")
    out["event_half"] = np.where(event_date.dt.month.le(6), "H1", "H2")
    out.loc[event_date.isna(), "event_half"] = pd.NA
    out["event_half"] = out["event_half"].astype("string")
    out["event_week"] = week.astype("string")
    out["event_dayofyear_sin"] = np.sin(2 * math.pi * day / 365.25).astype("float32")
    out["event_dayofyear_cos"] = np.cos(2 * math.pi * day / 365.25).astype("float32")
    out["event_week_sin"] = np.sin(2 * math.pi * week.astype(float) / 52.1775).astype("float32")
    out["event_week_cos"] = np.cos(2 * math.pi * week.astype(float) / 52.1775).astype("float32")
    out["weather_join_ready_v1"] = (event_date.notna() & out["domestic_province_std"].notna()).astype("int8")
    return out.drop(columns=["product_code_key"])


def coverage_summary(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        part = frame.loc[frame["split"] == split]
        n = len(part)
        rows.append({
            "target": target_name,
            "split": split,
            "n_rows": n,
            "raw_food_group_coverage": float(part["product_group_raw"].notna().mean()) if n else np.nan,
            "food_group_l1_classified_rate": float(part["food_group_l1_std"].ne("미분류").mean()) if n else np.nan,
            "food_group_l2_classified_rate": float(part["food_group_l2_std"].ne("미분류").mean()) if n else np.nan,
            "origin_iso2_known_rate": float(part["origin_iso2_std"].ne("ZZ").mean()) if n else np.nan,
            "domestic_province_known_rate": float(part["domestic_province_std"].notna().mean()) if n else np.nan,
            "weather_join_ready_rate": float(part["weather_join_ready_v1"].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def l1_summary(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    out = (frame.loc[frame["split"].isin(["train", "validation", "test"])]
           .groupby(["split", "food_group_l1_std"], observed=True)
           .agg(n_rows=("target", "size"), n_positive=("target", "sum"))
           .reset_index())
    out["positive_rate"] = out["n_positive"] / out["n_rows"]
    out["target"] = target_name
    return out[["target", "split", "food_group_l1_std", "n_rows", "n_positive", "positive_rate"]]


def mapping_source_summary(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    out = (frame.groupby(["split", "food_group_mapping_source"], observed=True)
           .size().rename("n_rows").reset_index())
    out["share"] = out["n_rows"] / out.groupby("split", observed=True)["n_rows"].transform("sum")
    out["target"] = target_name
    return out[["target", "split", "food_group_mapping_source", "n_rows", "share"]]


def cardinality_summary(before: pd.DataFrame, after: pd.DataFrame, target_name: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "validation", "test"]:
        b = before.loc[before["split"] == split]
        a = after.loc[after["split"] == split]
        rows.extend([
            {"target": target_name, "split": split, "variable": "product_name_std", "n_unique": int(b["product_name_std"].nunique())},
            {"target": target_name, "split": split, "variable": "product_name_key_v1", "n_unique": int(a["product_name_key_v1"].nunique())},
            {"target": target_name, "split": split, "variable": "product_group_raw", "n_unique": int(b["product_group_raw"].nunique())},
            {"target": target_name, "split": split, "variable": "food_group_l2_std", "n_unique": int(a["food_group_l2_std"].nunique())},
            {"target": target_name, "split": split, "variable": "food_group_l1_std", "n_unique": int(a["food_group_l1_std"].nunique())},
        ])
    return pd.DataFrame(rows)


def unseen_category_summary(frame: pd.DataFrame, target_name: str, features: list[str]) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"]
    rows = []
    for feature in features:
        known = set(train[feature].dropna().astype(str))
        for split in ["validation", "test"]:
            part = frame.loc[frame["split"] == split, feature].dropna().astype(str)
            unseen = ~part.isin(known)
            rows.append({"target": target_name, "feature": feature, "split": split,
                         "n_nonnull": len(part), "n_unseen": int(unseen.sum()),
                         "unseen_rate": float(unseen.mean()) if len(part) else np.nan})
    return pd.DataFrame(rows)


def feature_catalog(config: dict[str, Any]) -> pd.DataFrame:
    descriptions = {
        "food_group_l1_std": "원천군·학습기간 코드/품목명 대응·품목명 키워드로 생성한 14개 대분류",
        "food_group_l2_std": "공백·별칭을 정규화하고 학습기간 대응표로 보강한 원천군 수준",
        "product_name_key_v1": "괄호·중량·구두점을 제거한 품목명 키",
        "product_is_composite_v1": "쉼표·세미콜론이 포함된 복합 품목명 여부",
        "origin_iso2_std": "생산국→원산국 코드→원산국명→수입국 코드 순서로 통합한 ISO2 키",
        "origin_region_std": "ISO2를 권역으로 묶은 값",
        "supply_route_v1": "국가·국내 지역 근거의 국내/수입/불명 구분",
        "domestic_province_std": "재배지 우선, 수거지 보조의 표준 시도명",
        "domestic_macro_region": "표준 시도를 6개 국내 권역으로 묶은 값",
        "event_half": "상·하반기",
        "event_dayofyear_sin": "연중 일자의 순환형 사인 값",
        "event_dayofyear_cos": "연중 일자의 순환형 코사인 값",
        "event_week_sin": "주차의 순환형 사인 값",
        "event_week_cos": "주차의 순환형 코사인 값",
        "food_group_mapping_source": "식품군 보강 경로; 출처 프록시 위험으로 진단 전용",
        "origin_country_mapping_source": "국가 통합에 사용한 원천 열; 진단 전용",
        "domestic_location_basis": "재배지/수거지 선택 근거; 진단 전용",
        "weather_join_ready_v1": "날짜와 국내 시도가 모두 있는지 확인하는 진단 플래그",
    }
    rows = []
    for role, features in config["feature_sets"].items():
        for feature in features:
            rows.append({"feature": feature, "role": role, "description": descriptions.get(feature, "기존 학습 데이터 변수")})
    return pd.DataFrame(rows).drop_duplicates("feature")


def external_readiness_table() -> pd.DataFrame:
    return pd.DataFrame([
        ["식품군 표준화", "부분 적용", "품목코드·품목명·원천 식품군", "record_id 내부 열", "train-fit 대응표 적용 완료", "중", "베이스라인과 확장모델 비교"],
        ["농약 등록정보", "후속 조건부", "농약명·작물명·병해충·희석배수·사용량·PHI·횟수", "표준 작물명 + 적용 시작/종료일", "API 확보 표기, 역사시점 키 미확인", "높음", "현재 등록표를 과거행에 일괄 연결하지 말고 유효기간 확보"],
        ["MRL", "후속 조건부", "품목별 잔류허용기준", "표준 품목코드 + effective_from/to + 검사일", "현재 원장은 라벨 생성과 얽힘", "매우 높음", "독립 버전 규정표 재구축 전 모델 입력 금지"],
        ["기상 ASOS/AWS/AAOS", "국내 부분집합 파일럿", "기온·강수·습도·일사·토양수분 등", "생산 위치 + 생산/수확 기준일 + 관측소", "현재는 국내 시도와 검사계열 날짜만 일부 보유", "높음", "재배지와 수확일이 확인되는 행에서만 과거창 생성"],
        ["병해충 발생", "후속 조건부", "작물·병해충·지역·발생시점", "표준 작물 + 지역/PNU + 발생일", "API 확보 표기, 현재 검사원장에 PNU 없음", "중", "지역·시점 해상도 확인 후 월/주 단위 과거값만 연결"],
        ["토양·팜맵", "현재 보류", "토양산도·수분·필지 공간정보", "PNU/좌표 + 관측일", "검사원장에 PNU/좌표 없음", "중", "주소 지오코딩 또는 원천 PNU 확보 후 재검토"],
        ["농약 판매기록", "현재 보류", "품목·판매일·판매량·사용작물", "작물 + 지역 + 판매일", "미확보 표기", "중", "자료 확보 후 검사일 이전 누적량으로만 생성"],
    ], columns=["candidate", "status", "available_fields", "required_join_key", "current_readiness", "leakage_risk", "recommendation"])


def _save_figures(coverage: pd.DataFrame, l1: pd.DataFrame, cardinality: pd.DataFrame,
                  unseen: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    focus = coverage.loc[coverage["split"].isin(["train", "validation", "test"])].copy()
    long = focus.melt(id_vars=["target", "split"], value_vars=["raw_food_group_coverage", "food_group_l1_classified_rate"], var_name="measure", value_name="rate")
    labels = {"raw_food_group_coverage": "원천 식품군", "food_group_l1_classified_rate": "표준 대분류"}
    long["measure"] = long["measure"].replace(labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    target_labels = {"occurrence": "잔류 발생", "screening": "MRL 10% 관심농도", "noncompliance": "기준 부적합"}
    for ax, target in zip(axes, TARGETS):
        part = long.loc[long["target"] == target]
        pivot = part.pivot(index="split", columns="measure", values="rate").reindex(["train", "validation", "test"])
        pivot.plot(kind="bar", ax=ax, color=["#A0AEC0", "#4C78A8"])
        ax.set_title(target_labels[target])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("가용률")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("식품군 표준화 전후 가용성")
    fig.savefig(figure_dir / "food_group_standardization_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    occ = l1.loc[(l1["target"] == "occurrence") & (l1["split"] == "train")].sort_values("n_rows")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(occ["food_group_l1_std"], occ["n_rows"], color="#4C78A8")
    ax.set_xlabel("검사 건수")
    ax.set_title("잔류 발생 학습 데이터의 표준 식품군 구성")
    for y, value in enumerate(occ["n_rows"]):
        ax.text(value, y, f" {value:,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "food_group_l1_train_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    card = cardinality.loc[(cardinality["target"] == "occurrence") & (cardinality["split"] == "train")]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(card["variable"], card["n_unique"], color=["#A0AEC0", "#4C78A8", "#A0AEC0", "#72B7B2", "#F58518"])
    ax.set_yscale("log")
    ax.set_ylabel("고유값 수(로그축)")
    ax.set_title("표준화 전후 범주 수")
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(card["n_unique"]):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_cardinality_before_after.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = unseen.sort_values("unseen_rate", ascending=False).head(20).sort_values("unseen_rate")
    labels = top["target"] + " · " + top["split"] + " · " + top["feature"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, 100 * top["unseen_rate"], color="#F58518")
    ax.set_xlabel("신규범주 비율(%)")
    ax.set_title("검증·테스트 신규범주율 상위 변수")
    fig.tight_layout()
    fig.savefig(figure_dir / "category_unseen_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 해당 없음."
    rendered = df.copy()
    for col in rendered.select_dtypes(include=["float", "float32", "float64"]).columns:
        rendered[col] = rendered[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines = ["| " + " | ".join(map(str, rendered.columns)) + " |", "| " + " | ".join(["---"] * len(rendered.columns)) + " |"]
    for row in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def write_report(coverage: pd.DataFrame, l1: pd.DataFrame, cardinality: pd.DataFrame,
                 unseen: pd.DataFrame, external: pd.DataFrame, docs_dir: Path) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    focus = coverage.loc[coverage["split"].isin(["train", "validation", "test"])].copy()
    focus_pct = focus.copy()
    rate_cols = [c for c in focus.columns if c.endswith("_rate") or c.endswith("_coverage")]
    for col in rate_cols:
        focus_pct[col] = 100 * focus_pct[col]
    occ_card = cardinality.loc[(cardinality["target"] == "occurrence") & (cardinality["split"] == "train")]
    top_unseen = unseen.nlargest(12, "unseen_rate").copy()
    top_unseen["unseen_rate_pct"] = 100 * top_unseen["unseen_rate"]
    report = f"""# 변수 확장 및 고도화 1차

## 결론

- 내부 데이터 기반 변수 확장 완료.
- 식품군 대응표는 각 목표의 학습기간(2015–2024년)에서만 적합.
- 검증·테스트 식품군 보강에 미래 구간의 분포나 라벨을 사용하지 않음.
- 국가·국내 시도·권역·공급경로·주기형 시기 변수 생성.
- 결과값·판정값·MRL·목표 라벨은 파생변수 생성에 사용하지 않음.
- 현재 MRL·농약 등록·기상·병해충·토양은 연결키와 시점 조건이 충족될 때만 후속 적용.

## 식품군 표준화 방식

- 1순위: 원천 `product_group_raw` 정규화.
- 2순위: 학습기간의 안정적인 품목코드→식품군 대응.
- 3순위: 학습기간의 안정적인 정제 품목명→식품군 대응.
- 4순위: 품목명 키워드 대분류.
- 나머지: `미분류` 유지. 임의로 가장 흔한 식품군을 넣지 않음.
- 대응표 채택 기준: 학습기간 3건 이상, 최빈 식품군 순도 80% 이상.

## 목표·구간별 가용성

{_markdown_table(focus_pct)}

## 품목 변수 카디널리티

{_markdown_table(occ_card)}

## 신규 범주 발생 위험

{_markdown_table(top_unseen[["target", "feature", "split", "n_nonnull", "n_unseen", "unseen_rate_pct"]])}

## 생성 변수 사용정책

- 기본 변수: `food_group_l1_std`, 월·분기·계절·반기, 일자·주차 순환형 변수.
- 확장 비교 변수: 식품군 소분류, 국가·권역·공급경로, 국내 시도·권역.
- 조건부 비교 변수: 정제 품목명, 품목코드, 업무·수거·재배 관련 변수.
- 진단 전용: 식품군/국가/지역 보강 출처, 기상 연결 가능 플래그, `source_system`, 원본 날짜와 연도.
- `facility_type`: 2015–2024년은 텍스트, 2025–2026년은 숫자 코드로 구조가 바뀌어 코드북 확보 전 모델 입력에서 제외.
- 고카디널리티 품목명은 CatBoost 등 범주형 처리 모델에서만 별도 비교.
- 검증·테스트 신규범주는 `__UNK__` 또는 모델의 미지범주 처리로 보존.

## 외부변수 준비도

{_markdown_table(external)}

## 외부변수 적용 원칙

- 검사 이후 확정된 값은 사용 금지.
- MRL·등록정보는 검사일 당시 유효했던 버전만 연결.
- 기상은 생산지와 생산·수확 기준일을 확보한 경우에만 과거 관측창 생성.
- 검사일 또는 접수일 주변 기상을 생산환경으로 해석하지 않음.
- 병해충·판매량은 검사일 이전 값만 집계.
- 외부변수 연결 전후의 행 수, 연결률, 중복증폭, 시점 역전을 자동 검증.

## 참고자료와 품질 주의

- 참고: 연구 과정의 위해요소별 중요인자·데이터 출처 후보 목록.
- 해당 시트는 33행의 후보 데이터 목록이며 실제 관측값 테이블이 아님.
- ASOS·AWS·AAOS 항목이 중복 기재되어 있어 외부자료 전수 개수로 해석하지 않음.
- 확보(API/CSV)는 접근 수단이 있다는 의미이며 현재 검사원장과 연결됐다는 의미가 아님.

## 다음 단계

- 내부 기본 변수와 확장 변수를 이용한 베이스라인 모델 비교.
- `core` 대 `core+extended` 성능 및 출처·연도·품목군별 안정성 비교.
- 외부변수는 베이스라인 이후 별도 실험으로 추가.
- 부적합 목표는 테스트 양성 24건이므로 신뢰구간과 보조 백테스트 병행.
"""
    path = docs_dir / "01_변수확장_고도화_1차.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_feature_engineering(input_dir: Path, output_dir: Path, docs_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = docs_dir / "table"
    figure_dir = docs_dir / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)
    coverages, l1_tables, sources, cards, unseen_tables, mapping_details = [], [], [], [], [], []
    manifests: dict[str, Any] = {}
    extended_features = (config["feature_sets"]["core_categorical"] + config["feature_sets"]["extended_categorical"] + config["feature_sets"]["conditional_categorical"])

    for target_name in TARGETS:
        input_path = input_dir / f"{target_name}.parquet"
        before = pd.read_parquet(input_path)
        mappings = fit_food_mappings(before.loc[before["split"] == "train"], config)
        after = transform_features(before, mappings, config)
        output_path = output_dir / f"{target_name}_features_v1.parquet"
        after.to_parquet(output_path, index=False)
        detail = mappings.details.copy()
        detail.insert(0, "target", target_name)
        mapping_details.append(detail)
        coverages.append(coverage_summary(after, target_name))
        l1_tables.append(l1_summary(after, target_name))
        sources.append(mapping_source_summary(after, target_name))
        cards.append(cardinality_summary(before, after, target_name))
        unseen_tables.append(unseen_category_summary(after, target_name, extended_features))
        map_payload = {
            "code_l1": mappings.code_l1, "code_l2": mappings.code_l2,
            "name_l1": mappings.name_l1, "name_l2": mappings.name_l2,
        }
        (output_dir / f"{target_name}_food_mappings_v1.json").write_text(
            json.dumps(map_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifests[target_name] = {
            "input": str(input_path), "output": str(output_path), "rows": len(after),
            "columns": len(after.columns), "feature_sets": config["feature_sets"],
            "mapping_fit_split": "train", "mapping_fit_period": "2015-2024",
        }

    coverage = pd.concat(coverages, ignore_index=True)
    l1 = pd.concat(l1_tables, ignore_index=True)
    source = pd.concat(sources, ignore_index=True)
    cardinality = pd.concat(cards, ignore_index=True)
    unseen = pd.concat(unseen_tables, ignore_index=True)
    mapping = pd.concat(mapping_details, ignore_index=True)
    external = external_readiness_table()
    catalog = feature_catalog(config)
    for df, name in [
        (coverage, "feature_coverage_by_target_split.csv"),
        (l1, "food_group_l1_by_target_split.csv"),
        (source, "food_group_mapping_source.csv"),
        (cardinality, "feature_cardinality_before_after.csv"),
        (unseen, "category_unseen_rate.csv"),
        (mapping, "train_fitted_food_mapping_quality.csv"),
        (external, "external_feature_readiness.csv"),
        (catalog, "feature_catalog.csv"),
    ]:
        df.to_csv(table_dir / name, index=False, encoding="utf-8-sig")
    _save_figures(coverage, l1, cardinality, unseen, figure_dir)
    report_path = write_report(coverage, l1, cardinality, unseen, external, docs_dir)
    manifest = {"version": "features_v1", "config": str(config_path), "report": str(report_path), "targets": manifests}
    (output_dir / "feature_engineering_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"coverage": coverage, "l1": l1, "mapping_source": source, "cardinality": cardinality,
            "unseen": unseen, "mapping": mapping, "external": external, "catalog": catalog,
            "report_path": report_path, "manifest": manifest}






