import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_rrf_scores_match_formula() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 62)


def test_item_in_both_lists_ranks_first() -> None:
    fused = reciprocal_rank_fusion({"a": ["x", "y"], "b": ["y", "z"]})
    assert fused[0][0] == "y"


def test_single_list_preserves_order() -> None:
    fused = reciprocal_rank_fusion({"only": ["c", "a", "b"]})
    assert [i for i, _ in fused] == ["c", "a", "b"]


def test_duplicates_within_a_list_count_once() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x", "x", "y"]}, k=10))
    assert fused["x"] == pytest.approx(1 / 11)
    assert fused["y"] == pytest.approx(1 / 12)  # rank 2, not 3


def test_ties_break_by_id_and_empty_input_is_empty() -> None:
    fused = reciprocal_rank_fusion({"a": ["b"], "b": ["a"]})
    assert [i for i, _ in fused] == ["a", "b"]
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []
