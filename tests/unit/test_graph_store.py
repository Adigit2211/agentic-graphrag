from graph.models import ChunkRecord, EntityRecord
from graph.naming import chunk_id, entity_id, normalize_name
from graph.store import InMemoryGraphStore


def _entity(name: str) -> EntityRecord:
    norm = normalize_name(name)
    return EntityRecord(id=entity_id(norm), name=name, norm_name=norm, type="ORG")


def _chain_store() -> tuple[InMemoryGraphStore, dict[str, str]]:
    """a -- b -- c -- d, one chunk per edge."""
    store = InMemoryGraphStore()
    ids: dict[str, str] = {}
    for name in ("Alpha Corp", "Beta Inc", "Gamma Ltd", "Delta Co"):
        ent = _entity(name)
        store.add_entity(ent)
        ids[name] = ent.id
    pairs = [
        ("Alpha Corp", "Beta Inc"),
        ("Beta Inc", "Gamma Ltd"),
        ("Gamma Ltd", "Delta Co"),
    ]
    for i, (s, o) in enumerate(pairs):
        cid = f"chunk{i}"
        store.add_chunk(ChunkRecord(cid, "doc", i, f"text {i}"))
        for name in (s, o):
            store.add_mention(ids[name], cid)
        store.add_relation(ids[s], "linked_to", ids[o], cid)
    return store, ids


def test_ids_are_deterministic() -> None:
    assert chunk_id("d", 0, "t") == chunk_id("d", 0, "t")
    assert chunk_id("d", 0, "t") != chunk_id("d", 1, "t")
    assert entity_id("aurora labs") == entity_id(normalize_name("Aurora  LABS!"))


def test_writes_are_idempotent() -> None:
    store, ids = _chain_store()
    before = store.stats()
    store.add_entity(_entity("Alpha Corp"))
    store.add_relation(ids["Alpha Corp"], "linked_to", ids["Beta Inc"], "chunk0")
    store.add_mention(ids["Alpha Corp"], "chunk0")
    assert store.stats() == before


def test_expand_respects_hop_limit() -> None:
    store, ids = _chain_store()
    one = store.expand([ids["Alpha Corp"]], hops=1, limit=100)
    two = store.expand([ids["Alpha Corp"]], hops=2, limit=100)
    assert set(one) == {ids["Alpha Corp"], ids["Beta Inc"]}
    assert set(two) == {ids["Alpha Corp"], ids["Beta Inc"], ids["Gamma Ltd"]}
    assert two[ids["Gamma Ltd"]] == 2


def test_expand_respects_entity_limit() -> None:
    store, ids = _chain_store()
    assert len(store.expand([ids["Alpha Corp"]], hops=3, limit=2)) == 2


def test_link_entities_prefers_longest_match() -> None:
    store = InMemoryGraphStore()
    for name in ("Aurora Labs", "Labs", "Mira Chen"):
        store.add_entity(_entity(name))
    linked = store.link_entities(normalize_name("Who founded Aurora Labs?"))
    assert linked == [entity_id("aurora labs")]


def test_chunks_ranked_by_distance() -> None:
    store, ids = _chain_store()
    dist = store.expand([ids["Alpha Corp"]], hops=2, limit=100)
    ranked = store.chunks_for_entities(dist)
    assert ranked[0] == "chunk0"  # touches the seed (distance 0)
    assert set(ranked) == {"chunk0", "chunk1", "chunk2"}


def test_subgraph_contains_only_induced_edges() -> None:
    store, ids = _chain_store()
    ctx = store.subgraph([ids["Alpha Corp"], ids["Beta Inc"]])
    assert len(ctx.nodes) == 2
    assert len(ctx.edges) == 1
    assert ctx.edges[0].chunk_ids == ["chunk0"]
