"""Transcribe a case's ingested recordings with faster-whisper.

Writes outputs/<case>/<recording id>.json (segments with timestamps, plus the model and
settings used) and a plain-text .txt next to it.
"""
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, TranscribeConfig
from .ingest import case_dir, load_manifest


def load_model(tc: TranscribeConfig):
    from faster_whisper import WhisperModel  # heavy import; only when transcribing
    return WhisperModel(tc.model, device=tc.device, compute_type=tc.compute_type)


def transcribe_file(model, audio: Path, tc: TranscribeConfig) -> dict:
    segments, info = model.transcribe(
        str(audio), language=tc.language, beam_size=tc.beam_size, vad_filter=tc.vad_filter,
    )
    segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments  # generator: decoding happens here
    ]
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_s": round(info.duration, 3),
        "segments": segs,
    }


def fmt_ts(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_outputs(result: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(out_json)
    lines = [f"[{fmt_ts(s['start'])}] {s['text']}" for s in result["segments"]]
    out_json.with_suffix(".txt").write_text("\n".join(lines) + "\n")


def transcribe_case(case_id: str, cfg: Config, force: bool = False, model=None, log=print):
    """Transcribe every recording in the case that has no transcript yet."""
    cdir = case_dir(cfg, case_id)
    manifest = load_manifest(cdir / "manifest.json", case_id)
    if not manifest["recordings"]:
        log(f"no recordings ingested for case {case_id}")
        return []

    tc = cfg.transcribe
    out_dir = cfg.outputs_dir / case_id
    done = []
    for rec in manifest["recordings"]:
        out_json = out_dir / f"{rec['id']}.json"
        if out_json.exists() and not force:
            log(f"{rec['id']}  skip (already transcribed)")
            continue
        if model is None:
            log(f"loading model {tc.model} ({tc.device}/{tc.compute_type})")
            model = load_model(tc)

        t0 = time.monotonic()
        result = transcribe_file(model, cdir / rec["audio"], tc)
        elapsed = time.monotonic() - t0

        result = {
            "case_id": case_id,
            "recording_id": rec["id"],
            "source_name": rec["source_name"],
            "source_sha256": rec["source_sha256"],
            "settings": asdict(tc),
            "transcribed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "processing_s": round(elapsed, 1),
            **result,
        }
        write_outputs(result, out_json)
        rtf = elapsed / result["duration_s"] if result["duration_s"] else 0
        log(f"{rec['id']}  {result['duration_s']:.0f}s audio in {elapsed:.0f}s "
            f"(x{rtf:.2f} real time, lang={result['language']})  -> {out_json}")
        done.append(out_json)
    return done
