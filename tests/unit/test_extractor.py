import json

import pytest

from ingestion.extractor import ExtractionError, extract_graph, parse_extraction
from ingestion.normalizer import normalize_extraction, normalize_predicate
from llm.json_utils import extract_json_object
from tests.conftest import FakeLLM

VALID = {
    "entities": [{"name": "Aurora Labs", "type": "ORG"}],
    "relations": [
        {"subject": "Aurora Labs", "predicate": "founded by", "object": "Mira"}
    ],
}


def test_parse_plain_json() -> None:
    ex = parse_extraction(json.dumps(VALID))
    assert ex.entities[0].name == "Aurora Labs"
    assert ex.relations[0].object == "Mira"


def test_parse_json_inside_code_fence_and_prose() -> None:
    raw = "Sure! Here you go:\n```json\n" + json.dumps(VALID) + "\n```\nHope it helps."
    assert len(parse_extraction(raw).relations) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        '{"entities": "oops"}',
        '{"entities": [{"name": ""}]}',
        '{"relations": [{"subject": "a", "predicate": "", "object": "b"}]}',
        "[1, 2, 3]",
    ],
)
def test_parse_rejects_malformed_output(raw: str) -> None:
    with pytest.raises(ExtractionError):
        parse_extraction(raw)


def test_extract_json_object_rejects_arrays() -> None:
    with pytest.raises(ValueError):
        extract_json_object("[1]")


async def test_extract_retries_then_succeeds(fake_llm: FakeLLM) -> None:
    replies = iter(["garbage", "still garbage", json.dumps(VALID)])

    async def scripted(system: str, user: str, *, json_mode: bool = False) -> str:
        fake_llm.calls.append("extract")
        return next(replies)

    fake_llm.complete = scripted  # type: ignore[method-assign]
    ex = await extract_graph(fake_llm, "text", max_retries=2)
    assert ex.entities[0].name == "Aurora Labs"
    assert len(fake_llm.calls) == 3


async def test_extract_gives_up_after_max_retries(fake_llm: FakeLLM) -> None:
    fake_llm.extract_raw = "never valid"
    with pytest.raises(ExtractionError):
        await extract_graph(fake_llm, "text", max_retries=1)
    assert fake_llm.calls.count("extract") == 2


def test_normalize_deduplicates_and_adds_missing_endpoints() -> None:
    ex = parse_extraction(
        json.dumps(
            {
                "entities": [
                    {"name": "Aurora Labs", "type": "UNKNOWN"},
                    {"name": "  aurora   labs ", "type": "ORG"},
                ],
                "relations": [
                    {
                        "subject": "Aurora Labs",
                        "predicate": "Founded By",
                        "object": "Mira Chen",
                    },
                    {
                        "subject": "AURORA LABS",
                        "predicate": "founded_by",
                        "object": "mira chen",
                    },
                    {
                        "subject": "Aurora Labs",
                        "predicate": "is",
                        "object": "Aurora Labs",
                    },
                ],
            }
        )
    )
    entities, relations = normalize_extraction(ex)
    by_norm = {e.norm_name: e for e in entities}
    assert set(by_norm) == {"aurora labs", "mira chen"}
    assert by_norm["aurora labs"].type == "ORG"  # concrete type beats UNKNOWN
    assert by_norm["aurora labs"].name == "Aurora Labs"  # first surface form kept
    assert len(relations) == 1  # duplicate and self-loop dropped
    assert relations[0].predicate == "founded_by"


def test_normalize_predicate() -> None:
    assert normalize_predicate("  Headquartered In ") == "headquartered_in"
