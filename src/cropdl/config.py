"""Config loading: YAML -> nested dataclass with CLI overrides.

A single experiment is fully described by one YAML file. The same config object
drives data prep, training, evaluation, dehazing and profiling so that every run
is reproducible from one artifact (logged to W&B).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"          # produced by prepare.py
    crop: str = "potato"                          # which crop subset to train on
    image_size: int = 224
    fold: int = 0                                 # which CV fold is the val set
    num_folds: int = 5
    batch_size: int = 32
    num_workers: int = 8
    # augmentation (train only)
    aug_rotation: int = 30
    aug_translate: float = 0.1
    aug_scale_min: float = 0.8
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_color_jitter: float = 0.2
    # haze evaluation (dehazing experiments)
    haze_eval: bool = False                       # synth-haze the val set
    haze_beta: float = 1.5                        # scattering coefficient
    haze_airlight: float = 0.8
    dehaze_method: str = "none"                   # none | dcp | aodnet


@dataclass
class ModelConfig:
    name: str = "densenet121"                     # see models/factory.py registry
    pretrained: bool = True
    drop_rate: float = 0.2
    # densexnet head (paper's custom head on a timm backbone)
    densexnet_backbone: str = "densenet121"
    densexnet_hidden: int = 256


@dataclass
class OptimConfig:
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"                      # adamw | sgd | adam
    scheduler: str = "cosine"                     # cosine | plateau | none
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    use_class_weights: bool = True                # counter class imbalance
    early_stop_patience: int = 12
    mixed_precision: bool = True
    grad_clip: float = 1.0


@dataclass
class WandbConfig:
    project: str = "crop_disease"
    entity: str | None = "ashu-1069-rochester-institute-of-technology"
    mode: str = "online"                          # online | offline | disabled
    group: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    experiment: str = "default"
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_name(self) -> str:
        return f"{self.experiment}-{self.data.crop}-{self.model.name}-f{self.data.fold}"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(section_cls, data: dict):
    return section_cls(**{k: v for k, v in data.items() if k in section_cls.__dataclass_fields__})


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config, optionally deep-merging a dict of overrides.

    Overrides use dotted keys, e.g. {"data.fold": 1, "model.name": "vit_small"}.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        dotted: dict = {}
        for key, val in overrides.items():
            cur = dotted
            parts = key.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val
        raw = _merge(raw, dotted)

    return Config(
        experiment=raw.get("experiment", "default"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "outputs"),
        device=raw.get("device", "cuda"),
        data=_build(DataConfig, raw.get("data", {})),
        model=_build(ModelConfig, raw.get("model", {})),
        optim=_build(OptimConfig, raw.get("optim", {})),
        wandb=_build(WandbConfig, raw.get("wandb", {})),
    )


def parse_cli_overrides(pairs: list[str]) -> dict:
    """Turn ['data.fold=1', 'optim.lr=1e-4'] into a typed override dict."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be key=value")
        key, val = pair.split("=", 1)
        out[key] = yaml.safe_load(val)   # infers int/float/bool/str
    return out
