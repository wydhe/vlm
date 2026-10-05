"""CPU tests: RRF fusion, MIRAGE answer parsing, GRPO task parsing and rewards."""

from align.task import parse, reward, score
from rag.eval_mirage import parse_letter
from rag.retrievers import rrf_fuse

ANSWERABLE = {"question": "q", "docs": ["a", "b", "c", "d"], "gold": 2, "label": "yes"}
UNANSWERABLE = {"question": "q", "docs": ["a", "b", "c", "d"], "gold": -1, "label": "insufficient"}


def reply(cite: str, answer: str) -> str:
    return f"Reasoning: because.\nCitation: {cite}\nAnswer: {answer}"


def test_rrf_prefers_items_ranked_high_by_both():
    bm25 = [(1, 9.0), (2, 8.0), (3, 7.0)]
    dense = [(2, 0.9), (4, 0.8), (1, 0.7)]
    fused = [row for row, _ in rrf_fuse([bm25, dense], k=4)]
    assert fused[:2] == [2, 1]          # in both lists, near the top
    assert set(fused) == {1, 2, 3, 4}


def test_parse_letter():
    assert parse_letter("C", {"A", "B", "C", "D"}) == "C"
    assert parse_letter("The answer is B.", {"A", "B", "C", "D"}) == "B"
    assert parse_letter("E", {"A", "B", "C", "D"}) is None


def test_full_reward_for_correct_grounded_answer():
    r = reward(reply("[3]", "yes"), ANSWERABLE)
    assert r == {"format": 0.1, "answer": 1.0, "citation": 0.5, "total": 1.6}


def test_wrong_citation_and_over_refusal():
    assert reward(reply("[1]", "yes"), ANSWERABLE)["citation"] == 0.0
    over = reward(reply("none", "insufficient"), ANSWERABLE)
    assert over["answer"] == 0.0 and over["total"] == 0.1


def test_refusal_rewarded_only_when_unanswerable():
    assert reward(reply("none", "insufficient"), UNANSWERABLE)["total"] == 1.6
    assert reward(reply("[2]", "no"), UNANSWERABLE)["total"] == 0.1


def test_malformed_output_gets_nothing():
    assert reward("yes, see doc 3", ANSWERABLE)["total"] == 0.0
    assert not parse("Answer: yes").well_formed


def test_score_metrics():
    texts = [reply("[3]", "yes"), reply("none", "insufficient"), reply("none", "insufficient")]
    m = score(texts, [ANSWERABLE, ANSWERABLE, UNANSWERABLE])
    assert m["answer_acc"] == 0.5 and m["over_refusal"] == 0.5 and m["refusal_recall"] == 1.0
