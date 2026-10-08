"""LLM-based entity/relation extraction with schema validation and retry."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError, field_validator

from llm.client import LLMClient
from llm.json_utils import extract_json_object

EXTRACT_SYSTEM = (
    "[task:extract]\n"
    "You extract a knowledge graph from text. Reply with ONLY a JSON object:\n"
    '{"entities": [{"name": str, "type": str}], '
    '"relations": [{"subject": str, "predicate": str, "object": str}]}\n'
    "Rules: use entity names exactly as written in the text; predicates are "
    'short lowercase verb phrases such as "founded_by" or "acquired_by"; '
    "only include facts explicitly stated in the text."
)


class ExtractionError(ValueError):
    """Raised when the LLM output cannot be turned into a valid extraction."""


class ExtractedEntity(BaseModel):
    """An entity mention found in a chunk."""

    name: str = Field(min_length=1)
    type: str = "UNKNOWN"

    @field_validator("name", "type")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class ExtractedRelation(BaseModel):
    """A subject-predicate-object triple found in a chunk."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class Extraction(BaseModel):
    """Validated extraction result for one chunk."""

    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []


def parse_extraction(raw: str) -> Extraction:
    """Parse and validate raw LLM output.

    Raises:
        ExtractionError: on malformed JSON or schema violations.
    """
    try:
        return Extraction.model_validate(extract_json_object(raw))
    except (ValueError, ValidationError) as exc:
        raise ExtractionError(str(exc)) from exc


async def extract_graph(llm: LLMClient, text: str, max_retries: int = 2) -> Extraction:
    """Extract entities and relations from ``text``.

    Retries up to ``max_retries`` times, feeding the validation error back to
    the model.

    Raises:
        ExtractionError: if every attempt returned invalid output.
        llm.client.LLMError: if the backend itself fails.
    """
    feedback = ""
    last_error = ""
    for _ in range(max_retries + 1):
        raw = await llm.complete(
            EXTRACT_SYSTEM, f"TEXT:\n{text}{feedback}", json_mode=True
        )
        try:
            return parse_extraction(raw)
        except ExtractionError as exc:
            last_error = str(exc)
            feedback = (
                f"\n\nYour previous reply was invalid ({last_error}). "
                "Reply with ONLY valid JSON in the required format."
            )
    raise ExtractionError(
        f"extraction failed after {max_retries + 1} attempts: {last_error}"
    )
