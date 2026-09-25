# VeloHearing

Transcription pipeline for hearing recordings.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

A fresh server can be provisioned with `scripts/setup-server.sh` (run as root).

## Usage

Ingest recordings into a case. Each file is converted to 16 kHz mono WAV and recorded in
`data/<case>/manifest.json` along with the original file's SHA-256 hash. Source files
are never modified, and ingesting the same recording again does nothing.

```bash
.venv/bin/python -m velohearing ingest --case CASE-001 /path/to/recording.mp3
```

Settings are in `config.yaml`. `data/`, `outputs/` and all audio formats are gitignored,
because recordings must never be committed.

## Tests

```bash
.venv/bin/python -m pytest
```
