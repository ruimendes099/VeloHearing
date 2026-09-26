import json

import pytest

from test_transcribe import make_case

from velohearing.analyze import analyze_case
from velohearing.review import IMPERCEPTIBLE, ReviewUnavailable


def w(start, word):
    return {"start": start, "end": start + 0.4, "word": word, "prob": 0.9}


class FakeAgent:
    def __init__(self, name, words):
        self.name, self.words, self.calls = name, words, 0
        self.settings = {"engine": "fake", "model": name}

    def run(self, audio):
        self.calls += 1
        return self.words


def agents():
    # span 1 (0.5 s): A/B/C disagree.  span 2 (3.0 s): same.
    return [
        FakeAgent("A", [w(0, "o"), w(0.5, "réu"), w(1.0, "disse"), w(2.0, "que"), w(3.0, "xx")]),
        FakeAgent("B", [w(0, "o"), w(0.5, "meu"), w(1.0, "disse"), w(2.0, "que"), w(3.0, "yy")]),
        FakeAgent("C", [w(0, "o"), w(0.5, "seu"), w(1.0, "disse"), w(2.0, "que"), w(3.0, "zz")]),
    ]


def fake_reviewer(spans, words, names, **kw):
    # resolves span 1 to B; says nothing about span 2
    return {"decisions": {1: {"verdict": "B", "justification": "coerente", "model": "m"}},
            "requests": [{"request_id": "r1"}]}


def fake_scanner(lines, **kw):
    return {"suspects": [{"line": 0, "phrase": "disse", "reason": "suspeito",
                          "suggestion": "dise", "model": "m"}], "requests": []}


def test_review_resolves_and_unanswered_spans_stay_imperceptible(tmp_path):
    cfg, rec = make_case(tmp_path)
    [report] = analyze_case("c1", cfg, agents=agents(), reviewer=fake_reviewer,
                            scanner=fake_scanner, preflight=lambda m: None, log=lambda *_: None)
    a = json.loads(report.with_name(f"{rec['id']}.analysis.json").read_text())
    s1, s2 = a["spans"]
    assert (s1["status"], s1["verdict"]) == ("resolvido", "B")
    assert s2["status"] == IMPERCEPTIBLE
    assert a["transcript"] == ["[00:00:00] o meu disse [?D1] que [imperceptível 00:00:03–00:00:04]"]
    assert a["doubtful"][0]["text"].startswith("disse")
    text = report.read_text()
    assert "Trechos imperceptíveis: **1**" in text
    assert "| 2 | 00:00:03 | 00:00:04 |" in text


def test_without_review_spans_are_listed_for_review(tmp_path):
    cfg, _ = make_case(tmp_path)
    [report] = analyze_case("c1", cfg, review=False, agents=agents(), log=lambda *_: None)
    text = report.read_text()
    assert "por rever**" in text
    assert "Revisor de texto:** não executado" in text


def test_agent_output_is_cached(tmp_path):
    cfg, _ = make_case(tmp_path)
    ags = agents()
    analyze_case("c1", cfg, review=False, agents=ags, log=lambda *_: None)
    analyze_case("c1", cfg, review=False, agents=ags, log=lambda *_: None)
    assert [a.calls for a in ags] == [1, 1, 1]
    analyze_case("c1", cfg, review=False, force=True, agents=ags, log=lambda *_: None)
    assert [a.calls for a in ags] == [2, 2, 2]


def test_reviewer_access_is_checked_before_agents_run(tmp_path):
    cfg, _ = make_case(tmp_path)
    ags = agents()

    def refuse(model):
        raise ReviewUnavailable("no key")

    with pytest.raises(ReviewUnavailable):
        analyze_case("c1", cfg, agents=ags, preflight=refuse, log=lambda *_: None)
    assert [a.calls for a in ags] == [0, 0, 0]
