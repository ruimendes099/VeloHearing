from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class TranscribeConfig:
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True


@dataclass(frozen=True)
class Config:
    data_dir: Path
    outputs_dir: Path
    sample_rate: int
    channels: int
    transcribe: TranscribeConfig = TranscribeConfig()


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    base = Path(path).resolve().parent

    def resolve(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else base / p

    return Config(
        data_dir=resolve(raw["paths"]["data"]),
        outputs_dir=resolve(raw["paths"]["outputs"]),
        sample_rate=int(raw["ingest"]["sample_rate"]),
        channels=int(raw["ingest"]["channels"]),
        transcribe=TranscribeConfig(**(raw.get("transcribe") or {})),
    )
