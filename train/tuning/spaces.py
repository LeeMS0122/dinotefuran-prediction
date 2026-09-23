from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import yaml


def load_search_spaces(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    models = payload.get("models", {})
    expected = {"logistic", "catboost", "lightgbm", "xgboost", "lstm"}
    if set(models) != expected:
        raise ValueError(f"탐색공간 모델 불일치: {sorted(models)}")
    return payload


def sample_params(
    trial: optuna.Trial,
    model: str,
    spaces: dict[str, Any],
) -> dict[str, Any]:
    spec = spaces["models"][model]
    params: dict[str, Any] = {}
    for name, rule in spec.items():
        if name == "fixed":
            continue
        kind = rule["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(rule["choices"]))
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(rule["low"]),
                int(rule["high"]),
                step=int(rule.get("step", 1)),
                log=bool(rule.get("log", False)),
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(rule["low"]),
                float(rule["high"]),
                step=rule.get("step"),
                log=bool(rule.get("log", False)),
            )
        else:
            raise ValueError(f"{model}/{name}: 지원하지 않는 탐색공간 유형 {kind}")
    params.update(deepcopy(spec.get("fixed", {})))
    return params


def apply_tabular_params(
    base_config: dict[str, Any],
    model: str,
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["random_seed"] = int(seed)
    section = "logistic_regression" if model == "logistic" else model
    model_params = {key: value for key, value in params.items() if key != "tree_min_frequency"}
    config[section].update(model_params)
    if "tree_min_frequency" in params:
        config["tree_encoding"]["min_frequency"] = int(params["tree_min_frequency"])
    return config


def apply_lstm_params(
    base_config: dict[str, Any],
    params: dict[str, Any],
    seed: int,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config["seed"] = int(seed)
    architecture_keys = {
        "hidden_size",
        "num_layers",
        "bidirectional",
        "embedding_max_dim",
        "mlp_hidden_size",
        "dropout",
    }
    training_keys = set(params).difference(architecture_keys)
    for key in architecture_keys.intersection(params):
        config["architecture"][key] = params[key]
    for key in training_keys:
        config["training"][key] = params[key]
    if max_epochs_override is not None:
        config["training"]["max_epochs"] = int(max_epochs_override)
    return config
