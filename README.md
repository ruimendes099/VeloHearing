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

Transcribe everything ingested for a case (runs on the CPU with faster-whisper; the model downloads on first use):

```bash
.venv/bin/python -m velohearing transcribe --case CASE-001 [--model medium] [--language pt] [--force]
```

This writes `outputs/<case>/<recording id>.json` (timestamped segments, model settings and the source file hash) and a readable `.txt` next to it.

### Multi-agent analysis and imperceptibility report

```bash
.venv/bin/python -m velohearing analyze --case CASE-001 [--no-review] [--force]
```

1. **Three ASR agents** that fail in different ways transcribe each recording with
   word timestamps: Whisper `large-v3-turbo` (reference), Whisper `medium` with different
   decoding (no VAD, no conditioning on previous text), and a wav2vec2 CTC model that
   transcribes acoustically with no language model. Their output is cached in
   `outputs/<case>/<id>/agents/`.
2. **Concordance**: words are aligned by time and fuzzy spelling. A word heard by a
   majority with enough confidence is accepted. Words without a majority, low-confidence
   agreements, and words only the non-reference agents heard become disputed spans.
3. **Text reviewer** (Claude API, text only): for each disputed span it picks one agent's
   exact version when it is clearly coherent, or marks it imperceptible. Unanswered spans
   stay imperceptible. It then reads the final transcript and flags doubtful words the
   agents agreed on (e.g. deformed place names) with a suggestion, without changing them.
4. **Report** `outputs/<case>/<id>.relatorio.md`: every imperceptible span as
   `HH:MM:SS–HH:MM:SS` with each agent's version, the doubtful spans, the spans the review
   resolved, and the final transcript. `<id>.analysis.json` holds everything plus provenance
   (hashes, models, settings, API request ids).

The reviewer needs `ANTHROPIC_API_KEY`, from the environment or a `.env` file in the repo
root (gitignored). `--no-review` keeps everything on the server and lists disputed spans
as "por rever". On 4 CPU cores the three agents take about 2× the recording's length.

## Tests

```bash
.venv/bin/python -m pytest
```
