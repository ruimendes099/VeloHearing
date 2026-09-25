import json

import numpy as np
import pytest
import soundfile as sf

from velohearing.config import Config
from velohearing.ingest import IngestError, ingest


@pytest.fixture
def cfg(tmp_path):
    return Config(data_dir=tmp_path / "data", outputs_dir=tmp_path / "out",
                  sample_rate=16000, channels=1)


@pytest.fixture
def stereo_44k(tmp_path):
    sr = 44100
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "hearing.flac"
    sf.write(path, np.stack([tone, tone], axis=1), sr)
    return path


def test_ingest_normalizes_and_records(cfg, stereo_44k):
    rec = ingest(stereo_44k, "case-001", cfg)
    wav = cfg.data_dir / "case-001" / rec["audio"]
    info = sf.info(wav)
    assert (info.samplerate, info.channels) == (16000, 1)
    assert rec["duration_s"] == pytest.approx(2.0, abs=0.01)

    manifest = json.loads((cfg.data_dir / "case-001" / "manifest.json").read_text())
    assert [r["id"] for r in manifest["recordings"]] == [rec["id"]]


def test_reingest_is_noop(cfg, stereo_44k):
    first = ingest(stereo_44k, "case-001", cfg)
    second = ingest(stereo_44k, "case-001", cfg)
    assert first == second
    manifest = json.loads((cfg.data_dir / "case-001" / "manifest.json").read_text())
    assert len(manifest["recordings"]) == 1


def test_rejects_bad_case_id(cfg, stereo_44k):
    with pytest.raises(IngestError):
        ingest(stereo_44k, "../escape", cfg)


def test_rejects_non_audio(cfg, tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not audio")
    with pytest.raises(IngestError):
        ingest(bogus, "case-001", cfg)
