"""Fourth agent: a text specialist (Claude) that re-evaluates the spans the ASR agents
could not settle.

It only ever sees text, never audio. For each span it may pick one agent's version when
that version is coherent in context, or declare the span imperceptible. It cannot write
its own wording: a pick is replaced by that agent's exact words, and anything it does not
answer for is treated as imperceptible.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

IMPERCEPTIBLE = "imperceptivel"

SYSTEM = """\
És um revisor linguístico de transcrições forenses de audiências judiciais em português \
europeu. Vários sistemas automáticos de reconhecimento de fala (agentes) transcreveram a \
mesma gravação. Recebes os trechos onde não houve acordo suficiente entre eles, com o \
contexto que os rodeia e a versão de cada agente.

Para cada trecho decides uma de duas coisas:
- escolher a versão de UM agente, se essa versão for linguística e contextualmente \
coerente e as outras forem claramente erros de reconhecimento (grafia aproximada, \
palavras sem sentido, números por extenso vs. algarismos com o mesmo valor, etc.);
- marcar "imperceptivel", se nenhuma versão for claramente fiável, se duas versões \
plausíveis tiverem significados diferentes, ou se houver qualquer dúvida relevante.

Regras:
- Nunca inventes nem corrijas palavras: só podes escolher uma versão tal como está.
- Isto é prova em processo judicial. Na dúvida, "imperceptivel". Um trecho \
imperceptível bem assinalado é preferível a uma palavra errada dada como certa.
- Nomes próprios, números, datas, quantias, negações e quem disse o quê são \
críticos: exige que a versão escolhida seja inequívoca.
- Se a versão de um agente estiver vazia, esse agente não ouviu fala naquele trecho.
- A justificação é curta (uma frase) e em português europeu.
"""


class SpanDecision(BaseModel):
    id: int
    verdict: str  # agent name or "imperceptivel"
    justification: str


class ReviewResult(BaseModel):
    decisions: list[SpanDecision]


def load_api_key(env_file: Path) -> None:
    """Use ANTHROPIC_API_KEY from the environment, or from a gitignored .env file."""
    if os.environ.get("ANTHROPIC_API_KEY") or not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")


def build_prompt(spans: list[dict], words: list[dict], agent_names: list[str],
                 context_s: float) -> str:
    def context(lo: float, hi: float) -> str:
        # same 0.2 s margin as the span's own words, so context never repeats them
        return " ".join(w["word"] for w in words if lo < (w["start"] + w["end"]) / 2 < hi)

    parts = ["Trechos a rever:\n"]
    for sp in spans:
        parts.append(f'<trecho id="{sp["id"]}" inicio="{sp["start"]:.1f}s" fim="{sp["end"]:.1f}s" '
                     f'motivo="{", ".join(sp["reasons"])}">')
        parts.append(f"  contexto antes: …{context(sp['start'] - context_s, sp['start'] - 0.2)}")
        for name in agent_names:
            parts.append(f'  agente {name}: "{sp["alternatives"].get(name, "")}"')
        parts.append(f"  contexto depois: {context(sp['end'] + 0.2, sp['end'] + context_s)}…")
        parts.append("</trecho>\n")
    names = ", ".join(f'"{n}"' for n in agent_names)
    parts.append(f'Responde com uma decisão por trecho. verdict tem de ser um de: {names} ou '
                 f'"{IMPERCEPTIBLE}".')
    return "\n".join(parts)


def review_spans(spans: list[dict], words: list[dict], agent_names: list[str], model: str,
                 effort: str = "high", batch_size: int = 40, context_s: float = 20.0,
                 client=None, log=print) -> dict:
    """Returns {"decisions": {span id: {...}}, "requests": [...]} ."""
    import anthropic

    client = client or anthropic.Anthropic()
    Verdict = Literal[tuple(agent_names) + (IMPERCEPTIBLE,)]

    class Decision(SpanDecision):
        verdict: Verdict

    class Result(ReviewResult):
        decisions: list[Decision]

    decisions, requests = {}, []
    for i in range(0, len(spans), batch_size):
        batch = spans[i:i + batch_size]
        log(f"  review: spans {batch[0]['id']}-{batch[-1]['id']} of {len(spans)}")
        response = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_format=Result,
            messages=[{"role": "user",
                       "content": build_prompt(batch, words, agent_names, context_s)}],
        )
        requests.append({"request_id": response._request_id, "model": response.model,
                         "stop_reason": response.stop_reason})
        if response.stop_reason != "end_turn" or response.parsed_output is None:
            log(f"  review: batch not answered (stop_reason={response.stop_reason}); "
                f"its spans stay imperceptible")
            continue
        ids = {sp["id"] for sp in batch}
        for d in response.parsed_output.decisions:
            if d.id in ids:
                decisions[d.id] = {"verdict": d.verdict, "justification": d.justification,
                                   "model": response.model}
    return {"decisions": decisions, "requests": requests}


SCAN_SYSTEM = """\
És um revisor linguístico de transcrições forenses de audiências judiciais em português \
europeu, feitas por reconhecimento automático de fala. Lês a transcrição e assinalas as \
expressões que muito provavelmente foram mal reconhecidas: palavras que não existem, nomes \
próprios ou topónimos deformados, frases sem sentido no contexto, números ou datas \
incoerentes.

Regras:
- Assinala só erros prováveis de reconhecimento. Não assinales estilo, pontuação, \
maiúsculas, gralhas óbvias sem impacto no sentido, nem opiniões sobre o conteúdo.
- Os trechos entre [ ] já estão assinalados; ignora-os.
- "phrase" tem de ser copiado exatamente da linha indicada (uma a seis palavras).
- "suggestion" é o que provavelmente foi dito, ou vazio se não souberes. É só uma \
sugestão para o revisor humano; nunca substitui a transcrição.
- "reason" é uma frase curta em português europeu.
- Se não houver nada a assinalar, devolve uma lista vazia.
"""


class Suspect(BaseModel):
    line: int
    phrase: str
    reason: str
    suggestion: str


class ScanResult(BaseModel):
    suspects: list[Suspect]


def scan_transcript(lines: list[str], model: str, effort: str = "high", batch_lines: int = 300,
                    client=None, log=print) -> dict:
    """Full-text pass over the final transcript. Returns {"suspects": [...], "requests": [...]}
    where each suspect carries the 0-based index of its line."""
    import anthropic

    client = client or anthropic.Anthropic()
    suspects, requests = [], []
    for i in range(0, len(lines), batch_lines):
        chunk = lines[i:i + batch_lines]
        log(f"  scan: lines {i + 1}-{i + len(chunk)} of {len(lines)}")
        numbered = "\n".join(f"L{i + k + 1}: {text}" for k, text in enumerate(chunk))
        response = client.beta.messages.parse(
            model=model,
            max_tokens=16000,
            system=SCAN_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_format=ScanResult,
            messages=[{"role": "user", "content":
                       f"Transcrição (uma linha por Lnúmero):\n\n{numbered}\n\n"
                       f"Em \"line\" indica só o número (ex.: 12 para L12)."}],
        )
        requests.append({"request_id": response._request_id, "model": response.model,
                         "stop_reason": response.stop_reason})
        if response.stop_reason != "end_turn" or response.parsed_output is None:
            log(f"  scan: batch not answered (stop_reason={response.stop_reason})")
            continue
        for s in response.parsed_output.suspects:
            if i < s.line <= i + len(chunk):
                suspects.append({"line": s.line - 1, "phrase": s.phrase, "reason": s.reason,
                                 "suggestion": s.suggestion, "model": response.model})
    return {"suspects": suspects, "requests": requests}
