"""Concordance between agents: decide word by word where the agents agree.

The first agent is the reference timeline. Every other agent's words are aligned to it
by time (within a tolerance) and spelling (fuzzy match, so "Olívença"/"olivença" or small
CTC misspellings still count as agreement). Then:

- a reference word heard by a majority of agents with enough confidence is accepted;
- a reference word without a majority is flagged "discordância";
- a word with a majority but low mean confidence is flagged "baixa confiança";
- words the reference missed but at least two other agents heard at the same time are
  flagged "omitido pela referência".

Flagged words close together are merged into spans. Spans are what the reviewer and the
final report deal with.
"""
import re
import unicodedata

from rapidfuzz import fuzz

DISAGREE = "discordância"
LOW_CONF = "baixa confiança"
MISSED = "omitido pela referência"


def norm(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", w)


def similar(a: str, b: str, threshold: float) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if min(len(a), len(b)) <= 3:
        return a == b
    return fuzz.ratio(a, b) >= threshold


def center(w: dict) -> float:
    return (w["start"] + w["end"]) / 2


def align(ref: list[dict], hyp: list[dict], threshold: float, tol: float) -> list[int | None]:
    """For each ref word, the index of its matching hyp word (or None). Monotonic greedy
    search over hyp words that overlap the ref word's time window."""
    matches: list[int | None] = []
    j0 = 0
    for w in ref:
        lo, hi = w["start"] - tol, w["end"] + tol
        while j0 < len(hyp) and hyp[j0]["end"] < lo:
            j0 += 1
        best, best_score = None, -1.0
        j = j0
        while j < len(hyp) and hyp[j]["start"] <= hi:
            if similar(w["word"], hyp[j]["word"], threshold):
                score = fuzz.ratio(norm(w["word"]), norm(hyp[j]["word"])) - abs(center(w) - center(hyp[j]))
                if score > best_score:
                    best, best_score = j, score
            j += 1
        matches.append(best)
        if best is not None:
            j0 = best + 1
    return matches


def _missed_by_ref(others: dict[str, list[dict]], used: dict[str, set], threshold: float,
                   tol: float) -> list[tuple[float, float]]:
    """Time ranges where two or more non-reference agents heard the same word that the
    reference did not."""
    names = list(others)
    found = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            extra_b = [w for k, w in enumerate(others[b]) if k not in used[b]]
            for k, wa in enumerate(others[a]):
                if k in used[a]:
                    continue
                for wb in extra_b:
                    if abs(center(wa) - center(wb)) <= tol and similar(wa["word"], wb["word"], threshold):
                        found.append((min(wa["start"], wb["start"]), max(wa["end"], wb["end"])))
                        break
    return found


def concord(words: dict[str, list[dict]], match_threshold: float = 80, time_tolerance_s: float = 1.0,
            min_confidence: float = 0.5, merge_gap_s: float = 1.0) -> dict:
    """words: agent name -> timed words, first entry is the reference agent."""
    names = list(words)
    ref_name, ref = names[0], words[names[0]]
    others = {n: words[n] for n in names[1:]}
    majority = len(names) // 2 + 1

    matches = {n: align(ref, hyp, match_threshold, time_tolerance_s) for n, hyp in others.items()}
    used = {n: {m for m in matches[n] if m is not None} for n in others}

    ref_words, flags = [], []
    for i, w in enumerate(ref):
        agreeing = [ref_name] + [n for n in others if matches[n][i] is not None]
        probs = [w["prob"]] + [others[n][matches[n][i]]["prob"] for n in agreeing[1:]]
        conf = sum(probs) / len(probs)
        if len(agreeing) < majority:
            status = DISAGREE
        elif conf < min_confidence:
            status = LOW_CONF
        else:
            status = "ok"
        ref_words.append({**w, "votes": agreeing, "confidence": round(conf, 3), "status": status})
        if status != "ok":
            flags.append((w["start"], w["end"], status))

    if len(others) >= 2:
        flags += [(s, e, MISSED) for s, e in _missed_by_ref(others, used, match_threshold, time_tolerance_s)]

    spans = []
    for s, e, reason in sorted(flags):
        if spans and s - spans[-1]["end"] <= merge_gap_s:
            sp = spans[-1]
            sp["end"] = max(sp["end"], e)
            if reason not in sp["reasons"]:
                sp["reasons"].append(reason)
        else:
            spans.append({"start": s, "end": e, "reasons": [reason]})

    for n, sp in enumerate(spans, 1):
        sp["id"] = n
        sp["start"], sp["end"] = round(sp["start"], 2), round(sp["end"], 2)
        sp["alternatives"] = {name: words_in(words[name], sp["start"], sp["end"]) for name in names}

    total = len(ref_words) or 1
    return {
        "reference": ref_name,
        "words": ref_words,
        "spans": spans,
        "stats": {
            "ref_words": len(ref_words),
            "agreement": {n: round(sum(m is not None for m in matches[n]) / total, 3) for n in others},
            "flagged_words": sum(w["status"] != "ok" for w in ref_words),
        },
    }


def words_in(words: list[dict], start: float, end: float, margin: float = 0.2) -> str:
    return " ".join(w["word"] for w in words if start - margin <= center(w) <= end + margin)
