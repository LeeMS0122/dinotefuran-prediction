from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
DOCS_DIR = ROOT / "docs" / "eda_1차"
FIG_DIR = DOCS_DIR / "fig"
TABLE_DIR = DOCS_DIR / "table"
EDA2_DOCS_DIR = ROOT / "docs" / "eda_2차"
EDA2_FIG_DIR = EDA2_DOCS_DIR / "fig"
EDA2_TABLE_DIR = EDA2_DOCS_DIR / "table"
EDA3_DOCS_DIR = ROOT / "docs" / "eda_3차"
EDA3_FIG_DIR = EDA3_DOCS_DIR / "fig"
EDA3_TABLE_DIR = EDA3_DOCS_DIR / "table"
OUTPUT_DIR = ROOT / "output" / "eda"
CACHE_DIR = OUTPUT_DIR / "cache"


def configure_runtime() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180


def load_dotenv(path: Path | None = None) -> None:
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dirs() -> None:
    for path in (
        DOCS_DIR,
        FIG_DIR,
        TABLE_DIR,
        EDA2_DOCS_DIR,
        EDA2_FIG_DIR,
        EDA2_TABLE_DIR,
        EDA3_DOCS_DIR,
        EDA3_FIG_DIR,
        EDA3_TABLE_DIR,
        OUTPUT_DIR,
        CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def data_path_from_env(config: dict) -> Path:
    load_dotenv()
    key = config["data"]["env_key"]
    raw = os.getenv(key)
    if not raw:
        raise RuntimeError(f"{key}가 없습니다. train/.env를 확인하세요.")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"통합원장 파일이 없습니다: {path}")
    return path


def save_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    ensure_dirs()
    path = TABLE_DIR / filename
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_dirs()
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_eda2_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    ensure_dirs()
    path = EDA2_TABLE_DIR / filename
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_eda2_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_dirs()
    path = EDA2_FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_eda3_table(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    ensure_dirs()
    path = EDA3_TABLE_DIR / filename
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def save_eda3_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_dirs()
    path = EDA3_FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def percent(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else float("nan")


configure_runtime()
