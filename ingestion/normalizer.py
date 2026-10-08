"""Entity normalization and de-duplication."""

from __future__ import annotations

import re
from dataclasses import dataclass

from graph.naming import entity_id, normalize_name
from ingestion.extractor import Extraction


@dataclass(frozen=True)
class NormEntity:
    """A de-duplicated entity."""

    id: str
    name: str
    norm_name: str
    type: str


@dataclass(frozen=True)
class NormRelation:
    """A de-duplicated relation between two entity IDs."""

    subject_id: str
    predicate: str
    object_id: str


def normalize_predicate(predicate: str) -> str:
    """Lower-case and join words with underscores (``Founded By`` -> ``founded_by``)."""
    return "_".join(re.findall(r"\w+", predicate.lower()))


def normalize_extraction(
    extraction: Extraction,
) -> tuple[list[NormEntity], list[NormRelation]]:
    """De-duplicate entities by normalized name and resolve relation endpoints.

    - The first-seen surface form becomes the display name.
    - A concrete type wins over ``UNKNOWN``.
    - Relation endpoints missing from the entity list are added as entities.
    - Self-loops, empty predicates and duplicate triples are dropped.
    """
    entities: dict[str, NormEntity] = {}

    def register(name: str, type_: str) -> str | None:
        norm = normalize_name(name)
        if len(norm) < 2:
            return None
        eid = entity_id(norm)
        existing = entities.get(eid)
        if existing is None:
            entities[eid] = NormEntity(eid, name.strip(), norm, type_ or "UNKNOWN")
        elif existing.type == "UNKNOWN" and type_ not in ("", "UNKNOWN"):
            entities[eid] = NormEntity(eid, existing.name, norm, type_)
        return eid

    for ent in extraction.entities:
        register(ent.name, ent.type)

    relations: dict[tuple[str, str, str], NormRelation] = {}
    for rel in extraction.relations:
        subj = register(rel.subject, "UNKNOWN")
        obj = register(rel.object, "UNKNOWN")
        pred = normalize_predicate(rel.predicate)
        if subj is None or obj is None or not pred or subj == obj:
            continue
        relations[(subj, pred, obj)] = NormRelation(subj, pred, obj)

    return list(entities.values()), list(relations.values())
