from dataclasses import dataclass, field
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


DEFAULT_AGENTS = {
    "A": {"engine": "whisper", "model": "large-v3-turbo", "beam_size": 5,
          "vad_filter": True, "condition_on_previous_text": True},
    "B": {"engine": "whisper", "model": "medium", "beam_size": 5,
          "vad_filter": False, "condition_on_previous_text": False},
    "C": {"engine": "wav2vec2", "model": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"},
}


@dataclass(frozen=True)
class AnalysisConfig:
    language: str | None = "pt"
    agents: dict = field(default_factory=lambda: DEFAULT_AGENTS)  # first agent = reference
    match_threshold: float = 80       # rapidfuzz ratio (0-100) for two words to count as equal
    time_tolerance_s: float = 1.0     # how far apart two agents' timestamps may be
    min_confidence: float = 0.5       # below this mean confidence, agreement is still flagged
    merge_gap_s: float = 1.0          # flagged words closer than this merge into one span
    review_model: str = "claude-opus-5"
    review_effort: str = "high"
    review_batch_size: int = 40
    review_context_s: float = 20.0


@dataclass(frozen=True)
class Config:
    data_dir: Path
    outputs_dir: Path
    sample_rate: int
    channels: int
    transcribe: TranscribeConfig = TranscribeConfig()
    analysis: AnalysisConfig = AnalysisConfig()


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
        analysis=AnalysisConfig(**(raw.get("analysis") or {})),
    )
