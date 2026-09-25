"""Normalize source recordings to ASR-ready WAV and record them in a case manifest.

Source files are never modified. Each ingest stores a SHA-256 of the original so
normalized audio and transcripts can be traced back to the exact recording.
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from .config import Config

CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class IngestError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_dir(cfg: Config, case_id: str) -> Path:
    if not CASE_ID_RE.match(case_id):
        raise IngestError(f"invalid case id: {case_id!r}")
    return cfg.data_dir / case_id


def load_manifest(path: Path, case_id: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"case_id": case_id, "recordings": []}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def normalize(src: Path, dst: Path, cfg: Config) -> None:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", str(cfg.channels), "-ar", str(cfg.sample_rate),
        "-c:a", "pcm_s16le", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        dst.unlink(missing_ok=True)
        raise IngestError(f"ffmpeg failed on {src}: {result.stderr.strip()}")


def ingest(src: Path, case_id: str, cfg: Config) -> dict:
    """Ingest one recording into a case. Re-ingesting identical audio is a no-op."""
    src = Path(src).resolve()
    if not src.is_file():
        raise IngestError(f"not a file: {src}")

    cdir = case_dir(cfg, case_id)
    audio_dir = cdir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cdir / "manifest.json"
    manifest = load_manifest(manifest_path, case_id)

    digest = sha256_file(src)
    for rec in manifest["recordings"]:
        if rec["source_sha256"] == digest:
            return rec

    rec_id = digest[:12]
    dst = audio_dir / f"{rec_id}.wav"
    normalize(src, dst, cfg)

    info = sf.info(dst)
    if info.frames == 0:
        dst.unlink()
        raise IngestError(f"no audio decoded from {src}")

    rec = {
        "id": rec_id,
        "source_name": src.name,
        "source_path": str(src),
        "source_sha256": digest,
        "source_bytes": src.stat().st_size,
        "audio": str(dst.relative_to(cdir)),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_s": round(info.frames / info.samplerate, 3),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest["recordings"].append(rec)
    save_manifest(manifest_path, manifest)
    return rec
