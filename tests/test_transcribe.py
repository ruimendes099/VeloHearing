import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from velohearing.config import Config
from velohearing.ingest import ingest
from velohearing.transcribe import transcribe_case


class FakeModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kwargs):
        self.calls += 1
        segs = [SimpleNamespace(start=0.0, end=1.2, text=" Good morning. "),
                SimpleNamespace(start=1.2, end=2.0, text=" Please be seated.")]
        info = SimpleNamespace(language="en", language_probability=0.99, duration=2.0)
        return iter(segs), info


def make_case(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", outputs_dir=tmp_path / "out",
                 sample_rate=16000, channels=1)
    src = tmp_path / "a.wav"
    sf.write(src, np.zeros(32000), 16000)
    rec = ingest(src, "c1", cfg)
    return cfg, rec


def test_writes_json_and_txt(tmp_path):
    cfg, rec = make_case(tmp_path)
    model = FakeModel()
    [out] = transcribe_case("c1", cfg, model=model, log=lambda *_: None)

    data = json.loads(out.read_text())
    assert data["source_sha256"] == rec["source_sha256"]
    assert data["settings"]["model"] == "small"
    assert [s["text"] for s in data["segments"]] == ["Good morning.", "Please be seated."]
    assert out.with_suffix(".txt").read_text() == (
        "[00:00:00] Good morning.\n[00:00:01] Please be seated.\n")


def test_skips_existing_unless_forced(tmp_path):
    cfg, _ = make_case(tmp_path)
    model = FakeModel()
    transcribe_case("c1", cfg, model=model, log=lambda *_: None)
    transcribe_case("c1", cfg, model=model, log=lambda *_: None)
    assert model.calls == 1
    transcribe_case("c1", cfg, model=model, force=True, log=lambda *_: None)
    assert model.calls == 2
