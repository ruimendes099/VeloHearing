"""Transcription agents: independent ASR methods that each produce timed words.

Every agent returns a list of words {"start", "end", "word", "prob"} (seconds, text as
heard, confidence 0-1). The agents are chosen to fail differently:

- whisper: faster-whisper with word timestamps. Two instances with different models and
  decoding settings (VAD on/off, conditioning on previous text on/off).
- wav2vec2: a CTC acoustic model. It transcribes character by character with no language
  model, so it does not "fill in" plausible words the way Whisper can.
"""
import gc

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class WhisperAgent:
    def __init__(self, name: str, model: str, language: str | None, device: str = "cpu",
                 compute_type: str = "int8", beam_size: int = 5, vad_filter: bool = True,
                 condition_on_previous_text: bool = True):
        self.name = name
        self.settings = {
            "engine": "whisper", "model": model, "language": language, "device": device,
            "compute_type": compute_type, "beam_size": beam_size, "vad_filter": vad_filter,
            "condition_on_previous_text": condition_on_previous_text,
        }

    def run(self, audio_path) -> list[dict]:
        from faster_whisper import WhisperModel

        s = self.settings
        model = WhisperModel(s["model"], device=s["device"], compute_type=s["compute_type"])
        segments, _ = model.transcribe(
            str(audio_path), language=s["language"], beam_size=s["beam_size"],
            vad_filter=s["vad_filter"],
            condition_on_previous_text=s["condition_on_previous_text"],
            word_timestamps=True,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                if not w.word.strip():
                    continue
                # Whisper splits "Situa-se" and "7,5" into " Situa" + "-se", " 7" + ",5":
                # a piece with no leading space belongs to the previous word.
                if words and not w.word[0].isspace():
                    prev = words[-1]
                    prev["word"] += w.word.strip()
                    prev["end"] = round(w.end, 2)
                    prev["prob"] = round(min(prev["prob"], w.probability), 3)
                    continue
                words.append({"start": round(w.start, 2), "end": round(w.end, 2),
                              "word": w.word.strip(), "prob": round(w.probability, 3)})
        del model
        gc.collect()
        return words


class Wav2Vec2Agent:
    """CTC model run over overlapping windows; words are kept only from each window's core
    so nothing is cut at a window edge."""

    def __init__(self, name: str, model: str, chunk_s: float = 30.0, pad_s: float = 3.0):
        self.name = name
        self.settings = {"engine": "wav2vec2", "model": model, "chunk_s": chunk_s,
                         "pad_s": pad_s}

    def run(self, audio_path) -> list[dict]:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        s = self.settings
        processor = Wav2Vec2Processor.from_pretrained(s["model"])
        model = Wav2Vec2ForCTC.from_pretrained(s["model"]).eval()
        sec_per_frame = model.config.inputs_to_logits_ratio / SAMPLE_RATE

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError(f"expected 16 kHz mono audio, got {sr} Hz, shape {audio.shape}")

        chunk, pad = int(s["chunk_s"] * sr), int(s["pad_s"] * sr)
        words = []
        for core in range(0, len(audio), chunk):
            lo, hi = max(0, core - pad), min(len(audio), core + chunk + pad)
            piece = audio[lo:hi]
            if len(piece) < sr // 10:
                continue
            inputs = processor(piece, sampling_rate=sr, return_tensors="pt")
            with torch.inference_mode():
                logits = model(inputs.input_values).logits[0]
            frame_prob = logits.softmax(-1).max(-1).values.numpy()
            out = processor.decode(logits.argmax(-1), output_word_offsets=True)
            offset = lo / sr
            for wo in out.word_offsets:
                t0 = offset + wo["start_offset"] * sec_per_frame
                t1 = offset + wo["end_offset"] * sec_per_frame
                if not core / sr <= (t0 + t1) / 2 < (core + chunk) / sr:
                    continue
                span = frame_prob[wo["start_offset"]:max(wo["end_offset"], wo["start_offset"] + 1)]
                words.append({"start": round(float(t0), 2), "end": round(float(t1), 2), "word": wo["word"],
                              "prob": round(float(np.mean(span)), 3)})
        del model
        gc.collect()
        return words


def build_agents(agent_cfgs: dict, language: str | None) -> list:
    agents = []
    for name, c in agent_cfgs.items():
        c = dict(c)
        engine = c.pop("engine")
        if engine == "whisper":
            agents.append(WhisperAgent(name, language=language, **c))
        elif engine == "wav2vec2":
            agents.append(Wav2Vec2Agent(name, **c))
        else:
            raise ValueError(f"unknown engine {engine!r} for agent {name}")
    return agents
