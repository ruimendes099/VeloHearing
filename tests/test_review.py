from velohearing.review import build_prompt


def w(start, word):
    return {"start": start, "end": start + 0.2, "word": word, "prob": 0.9}


def test_prompt_context_excludes_the_span_words():
    words = [w(0, "antes"), w(1.0, "dentro"), w(1.25, "limite"), w(2.0, "depois")]
    span = {"id": 7, "start": 1.0, "end": 1.2, "reasons": ["discordância"],
            "alternatives": {"A": "dentro limite", "B": "dento"}}
    prompt = build_prompt([span], words, ["A", "B"], context_s=5)
    assert "contexto antes: …antes\n" in prompt
    assert "contexto depois: depois…" in prompt
    assert 'agente B: "dento"' in prompt
    assert '"A", "B" ou "imperceptivel"' in prompt
