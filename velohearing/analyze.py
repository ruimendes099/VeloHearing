"""Multi-agent analysis of a case: run the ASR agents, compute their concordance, have the
text reviewer settle disputed spans, and report every imperceptible stretch by timestamp.

Per recording it writes to outputs/<case>/:
  <id>/agents/<name>.json   each agent's timed words (reused unless settings change or --force)
  <id>.analysis.json        concordance, spans, review decisions, provenance
  <id>.relatorio.md         the report: imperceptible spans and the final transcript
"""
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz import fuzz

from .agents import build_agents
from .concordance import concord, norm
from .config import REPO_ROOT, Config
from .ingest import case_dir, load_manifest
from .review import IMPERCEPTIBLE, load_api_key, review_spans, scan_transcript
from .transcribe import fmt_ts

RESOLVED, UNREVIEWED = "resolvido", "por rever"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_agent(agent, audio: Path, out: Path, force: bool, log) -> list[dict]:
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if cached["settings"] == agent.settings:
            log(f"  agent {agent.name}: cached ({len(cached['words'])} words)")
            return cached["words"]
    log(f"  agent {agent.name}: {agent.settings['engine']} {agent.settings['model']} ...")
    t0 = time.monotonic()
    words = agent.run(audio)
    elapsed = round(time.monotonic() - t0, 1)
    _write_json(out, {"agent": agent.name, "settings": agent.settings,
                      "processing_s": elapsed, "words": words})
    log(f"  agent {agent.name}: {len(words)} words in {elapsed:.0f}s")
    return words


def apply_review(spans: list[dict], decisions: dict | None) -> None:
    for sp in spans:
        if decisions is None:
            sp["status"], sp["verdict"], sp["justification"] = UNREVIEWED, None, ""
            continue
        d = decisions.get(sp["id"])
        if d is None:
            sp["status"], sp["verdict"] = IMPERCEPTIBLE, IMPERCEPTIBLE
            sp["justification"] = "sem decisão do revisor; assinalado por precaução"
        else:
            sp["verdict"], sp["justification"] = d["verdict"], d["justification"]
            sp["status"] = IMPERCEPTIBLE if d["verdict"] == IMPERCEPTIBLE else RESOLVED


def build_lines(words: list[dict], spans: list[dict], max_words: int = 25,
                pause_s: float = 1.5) -> list[list[dict]]:
    """Reference words with every span replaced by its outcome, grouped into lines.
    Each line is a list of tokens {"start", "end", "text", "span"}."""
    tokens, placed = [], set()
    for w in words:
        c = (w["start"] + w["end"]) / 2
        sp = next((s for s in spans if s["start"] - 0.2 <= c <= s["end"] + 0.2), None)
        if sp is None:
            tokens.append({"start": w["start"], "end": w["end"], "text": w["word"], "span": None})
        elif sp["id"] not in placed:
            placed.add(sp["id"])
            tokens.append({"start": sp["start"], "end": sp["end"], "text": span_text(sp),
                           "span": sp["id"]})
    # spans with no reference words (only the other agents heard something there)
    tokens += [{"start": sp["start"], "end": sp["end"], "text": span_text(sp), "span": sp["id"]}
               for sp in spans if sp["id"] not in placed]
    tokens = sorted((t for t in tokens if t["text"]), key=lambda t: t["start"])

    lines, cur = [], []
    for t in tokens:
        if cur and (len(cur) >= max_words or t["start"] - cur[-1]["start"] > pause_s
                    or cur[-1]["text"].endswith((".", "?", "!"))):
            lines.append(cur)
            cur = []
        cur.append(t)
    if cur:
        lines.append(cur)
    return lines


def line_text(line: list[dict]) -> str:
    return f"[{fmt_ts(line[0]['start'])}] " + " ".join(t["text"] for t in line)


def fmt_range(start: float, end: float) -> str:
    """Whole seconds, widened outwards so a short span never collapses to one instant."""
    return f"{fmt_ts(math.floor(start))}–{fmt_ts(math.ceil(end))}"


def locate(phrase: str, line: list[dict], threshold: float = 80) -> list[dict] | None:
    """The run of plain (non-span) tokens in the line that best matches the phrase."""
    target = [norm(w) for w in phrase.split() if norm(w)]
    plain = [t for t in line if t["span"] is None]
    best, best_score = None, threshold
    for i in range(len(plain) - len(target) + 1):
        window = plain[i:i + len(target)]
        score = sum(fuzz.ratio(a, norm(t["text"])) for a, t in zip(target, window)) / len(target)
        if score >= best_score:
            best, best_score = window, score
    return best


def mark_doubtful(lines: list[list[dict]], suspects: list[dict]) -> list[dict]:
    doubtful = []
    for s in suspects:
        if not 0 <= s["line"] < len(lines):
            continue
        window = locate(s["phrase"], lines[s["line"]])
        if not window:
            continue
        d = {"id": f"D{len(doubtful) + 1}", "start": window[0]["start"], "end": window[-1]["end"],
             "text": " ".join(t["text"] for t in window), "reason": s["reason"],
             "suggestion": s["suggestion"], "model": s["model"]}
        window[-1]["text"] += f" [?{d['id']}]"
        doubtful.append(d)
    return doubtful


def span_text(sp: dict) -> str:
    rng = fmt_range(sp["start"], sp["end"])
    if sp["status"] == RESOLVED:
        return sp["alternatives"][sp["verdict"]]
    if sp["status"] == UNREVIEWED:
        return f"[por rever {rng}: {sp['alternatives'][next(iter(sp['alternatives']))]}]"
    return f"[imperceptível {rng}]"


def _cell(text: str) -> str:
    return (text or "—").replace("|", "\\|").replace("\n", " ")


def render_report(a: dict) -> str:
    spans, names = a["spans"], list(a["agents"])
    imp = [s for s in spans if s["status"] == IMPERCEPTIBLE]
    res = [s for s in spans if s["status"] == RESOLVED]
    unrev = [s for s in spans if s["status"] == UNREVIEWED]
    dur = a["duration_s"] or 1
    imp_s = sum(s["end"] - s["start"] for s in imp)

    out = [f"# Relatório de perceptibilidade — {a['source_name']}", "",
           f"- **Caso:** {a['case_id']}  ",
           f"- **Gravação:** {a['recording_id']} (SHA-256 `{a['source_sha256']}`)  ",
           f"- **Duração:** {fmt_ts(a['duration_s'])}  ",
           f"- **Analisado em:** {a['analyzed_at']}  "]
    for n in names:
        s = a["agents"][n]
        out.append(f"- **Agente {n}:** {s['engine']} `{s['model']}`" +
                   (" (referência)" if n == a["concordance"]["reference"] else "") + "  ")
    rv = a.get("review")
    out.append(f"- **Revisor de texto:** {rv['model']}  " if rv else
               "- **Revisor de texto:** não executado  ")
    out += ["", "## Resumo", "",
            f"- Trechos imperceptíveis: **{len(imp)}** ({imp_s:.1f} s, "
            f"{100 * imp_s / dur:.1f}% da gravação)",
            f"- Trechos duvidosos assinalados pelo revisor: {len(a.get('doubtful') or [])}",
            f"- Trechos em disputa resolvidos pela revisão: {len(res)}"]
    if unrev:
        out.append(f"- Trechos em disputa **por rever** (revisão não executada): {len(unrev)}")
    agr = a["concordance"]["stats"]["agreement"]
    out.append("- Concordância com a referência: " +
               ", ".join(f"{n} {100 * v:.0f}%" for n, v in agr.items()))

    def table(title: str, rows: list[dict]) -> None:
        out.extend(["", f"## {title}", ""])
        if not rows:
            out.append("Nenhum.")
            return
        out.append("| # | Início | Fim | Dur. | Motivo | " + " | ".join(names) + " | Revisão |")
        out.append("|---|---|---|---|---|" + "---|" * len(names) + "---|")
        for s in rows:
            verdict = (f"**{s['verdict']}**: " if s["status"] == RESOLVED else "") + s["justification"]
            a_, b_ = fmt_range(s["start"], s["end"]).split("–")
            out.append(f"| {s['id']} | {a_} | {b_} | "
                       f"{s['end'] - s['start']:.1f}s | {', '.join(s['reasons'])} | " +
                       " | ".join(_cell(s["alternatives"][n]) for n in names) +
                       f" | {_cell(verdict)} |")

    table("Trechos imperceptíveis", imp)
    if unrev:
        table("Trechos por rever", unrev)
    out.extend(["", "## Trechos duvidosos (assinalados pelo revisor de texto)", "",
                "Os agentes concordaram, mas o texto parece mal reconhecido. A transcrição "
                "mantém o que foi ouvido; a sugestão é só para verificação humana.", ""])
    doubtful = a.get("doubtful") or []
    if not doubtful:
        out.append("Nenhum." if a.get("review") else "Não verificado (revisão não executada).")
    else:
        out.append("| # | Início | Fim | Texto transcrito | Sugestão | Motivo |")
        out.append("|---|---|---|---|---|---|")
        for d in doubtful:
            a_, b_ = fmt_range(d["start"], d["end"]).split("–")
            out.append(f"| {d['id']} | {a_} | {b_} | {_cell(d['text'])} | "
                       f"{_cell(d['suggestion'])} | {_cell(d['reason'])} |")
    table("Trechos resolvidos pela revisão", res)
    out += ["", "## Transcrição final", "", "```", *a["transcript"], "```", ""]
    return "\n".join(out)


def analyze_case(case_id: str, cfg: Config, review: bool = True, force: bool = False,
                 log=print, reviewer=review_spans, scanner=scan_transcript,
                 agents=None) -> list[Path]:
    ac = cfg.analysis
    cdir = case_dir(cfg, case_id)
    manifest = load_manifest(cdir / "manifest.json", case_id)
    if not manifest["recordings"]:
        log(f"no recordings ingested for case {case_id}")
        return []
    if review:
        load_api_key(REPO_ROOT / ".env")

    agents = agents or build_agents(ac.agents, ac.language)
    out_dir = cfg.outputs_dir / case_id
    done = []
    for rec in manifest["recordings"]:
        log(f"{rec['id']}  {rec['source_name']}")
        audio = cdir / rec["audio"]
        words = {ag.name: run_agent(ag, audio, out_dir / rec["id"] / "agents" / f"{ag.name}.json",
                                    force, log) for ag in agents}
        conc = concord(words, ac.match_threshold, ac.time_tolerance_s, ac.min_confidence,
                       ac.merge_gap_s)
        log(f"  concordance: {conc['stats']['flagged_words']} flagged words, "
            f"{len(conc['spans'])} spans")

        review_meta = None
        if review and conc["spans"]:
            r = reviewer(conc["spans"], conc["words"], list(words), model=ac.review_model,
                         effort=ac.review_effort, batch_size=ac.review_batch_size,
                         context_s=ac.review_context_s, log=log)
            apply_review(conc["spans"], r["decisions"])
            review_meta = {"model": ac.review_model, "effort": ac.review_effort,
                           "requests": r["requests"]}
        elif review:
            apply_review(conc["spans"], {})
            review_meta = {"model": ac.review_model, "effort": ac.review_effort, "requests": []}
        else:
            apply_review(conc["spans"], None)

        analysis = {
            "case_id": case_id, "recording_id": rec["id"], "source_name": rec["source_name"],
            "source_sha256": rec["source_sha256"], "duration_s": rec["duration_s"],
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agents": {ag.name: ag.settings for ag in agents},
            "concordance_settings": {
                "match_threshold": ac.match_threshold, "time_tolerance_s": ac.time_tolerance_s,
                "min_confidence": ac.min_confidence, "merge_gap_s": ac.merge_gap_s},
            "review": review_meta,
            "concordance": {k: conc[k] for k in ("reference", "stats")},
            "spans": conc["spans"],
            "words": conc["words"],
        }
        lines = build_lines(conc["words"], conc["spans"])
        if review:
            sc = scanner([line_text(ln) for ln in lines], model=ac.review_model,
                         effort=ac.review_effort, log=log)
            analysis["doubtful"] = mark_doubtful(lines, sc["suspects"])
            review_meta["requests"] += sc["requests"]
        analysis["transcript"] = [line_text(ln) for ln in lines]
        out_json = out_dir / f"{rec['id']}.analysis.json"
        _write_json(out_json, analysis)
        report = out_dir / f"{rec['id']}.relatorio.md"
        report.write_text(render_report(analysis))
        n_imp = sum(s["status"] == IMPERCEPTIBLE for s in conc["spans"])
        log(f"  {n_imp} imperceptible spans  -> {report}")
        done.append(report)
    return done
