import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def _coerce_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _load_accelerator_kwargs() -> Dict[str, Any]:
    raw_config = os.environ.get("AIT_ACCELERATOR_CONFIG")
    kwargs: Dict[str, Any] = {}

    if raw_config:
        config_path = Path(raw_config)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                text = handle.read()
                try:
                    kwargs = yaml.safe_load(text) or {}
                except yaml.YAMLError:
                    kwargs = json.loads(text)
        else:
            try:
                kwargs = json.loads(raw_config)
            except json.JSONDecodeError:
                try:
                    kwargs = yaml.safe_load(raw_config) or {}
                except yaml.YAMLError:
                    kwargs = {}

    env_overrides = {
        "mixed_precision": os.environ.get("AIT_ACCELERATOR_MIXED_PRECISION"),
        "gradient_accumulation_steps": os.environ.get("AIT_ACCELERATOR_GRADIENT_ACCUMULATION"),
        "cpu": os.environ.get("AIT_ACCELERATOR_CPU"),
        "device_placement": os.environ.get("AIT_ACCELERATOR_DEVICE_PLACEMENT"),
        "split_batches": os.environ.get("AIT_ACCELERATOR_SPLIT_BATCHES"),
        "even_batches": os.environ.get("AIT_ACCELERATOR_EVEN_BATCHES"),
        "dynamo_backend": os.environ.get("AIT_ACCELERATOR_DYNAMO_BACKEND"),
    }

    for key, value in env_overrides.items():
        if value is None:
            continue
        if key in {"cpu", "device_placement", "split_batches", "even_batches"}:
            kwargs[key] = _coerce_bool(value)
        elif key == "gradient_accumulation_steps":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value

    if "kwargs_handlers" in kwargs and not isinstance(kwargs["kwargs_handlers"], list):
        raise ValueError("Accelerator kwargs 'kwargs_handlers' must be a list of handler configs")

    return kwargs


def get_accelerator() -> Accelerator:
    global global_accelerator
    if global_accelerator is None:
        accelerator_kwargs = _load_accelerator_kwargs()
        global_accelerator = Accelerator(**accelerator_kwargs)
    return global_accelerator

def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
