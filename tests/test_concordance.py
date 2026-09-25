from velohearing.concordance import DISAGREE, LOW_CONF, MISSED, concord, norm


def w(start, word, prob=0.9):
    return {"start": start, "end": start + 0.4, "word": word, "prob": prob}


def test_norm_ignores_case_accents_and_punctuation():
    assert norm("Olívença,") == norm("olivenca") == "olivenca"


def test_full_agreement_has_no_spans():
    words = [w(0, "bom"), w(0.5, "dia"), w(1.0, "senhores")]
    r = concord({"A": words, "B": list(words), "C": list(words)})
    assert r["spans"] == []
    assert r["stats"]["agreement"] == {"B": 1.0, "C": 1.0}


def test_fuzzy_spelling_counts_as_agreement():
    a = [w(0, "Olivença")]
    r = concord({"A": a, "B": [w(0.1, "Olívença")], "C": [w(0.2, "olivensa")]})
    assert r["spans"] == []


def test_reference_word_without_majority_is_flagged():
    a = [w(0, "o"), w(0.5, "réu"), w(1.0, "disse")]
    b = [w(0, "o"), w(0.5, "meu"), w(1.0, "disse")]
    c = [w(0, "o"), w(0.5, "seu"), w(1.0, "disse")]
    r = concord({"A": a, "B": b, "C": c})
    [sp] = r["spans"]
    assert sp["reasons"] == [DISAGREE]
    assert sp["alternatives"] == {"A": "réu", "B": "meu", "C": "seu"}


def test_low_confidence_agreement_is_flagged():
    a = [w(0, "sim", 0.2)]
    r = concord({"A": a, "B": [w(0, "sim", 0.3)], "C": [w(0, "sim", 0.4)]})
    [sp] = r["spans"]
    assert sp["reasons"] == [LOW_CONF]


def test_word_missed_by_reference_is_flagged():
    a = [w(0, "não"), w(1.0, "vi")]
    b = [w(0, "não"), w(0.5, "o"), w(1.0, "vi")]
    c = [w(0, "não"), w(0.5, "o"), w(1.0, "vi")]
    r = concord({"A": a, "B": b, "C": c}, merge_gap_s=0.0)
    assert [sp["reasons"] for sp in r["spans"]] == [[MISSED]]


def test_nearby_flags_merge_into_one_span():
    a = [w(0, "xx"), w(0.6, "yy")]
    b = [w(0, "aa"), w(0.6, "bb")]
    c = [w(0, "cc"), w(0.6, "dd")]
    r = concord({"A": a, "B": b, "C": c}, merge_gap_s=1.0)
    assert len(r["spans"]) == 1
    assert (r["spans"][0]["start"], r["spans"][0]["end"]) == (0, 1.0)
