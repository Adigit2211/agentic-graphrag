# Design decisions

## 1. Protocols first, backends second
`GraphStore` (graph/store.py) and the embedder/LLM interfaces are `typing.Protocol`s.
v0.1 implements the graph and vector stores in memory so the whole system runs and
is testable with no infrastructure. Neo4j and Qdrant are planned as additional
implementations (see README roadmap); the retrieval and agent code does not change.

## 2. Deterministic IDs make ingestion idempotent
- chunk ID = `sha256(doc_id, index, text)[:16]`
- entity ID = `sha256(normalized_name)[:16]`

Every store write is an upsert keyed by these IDs, so re-ingesting the same file
changes nothing (covered by `test_reingesting_does_not_duplicate_data`).
Entities are keyed by name only, not name+type, because small LLMs label types
inconsistently and that would split one real entity into several nodes.

## 3. Graph retrieval is deterministic
Entity linking is a whole-phrase match of normalized entity names inside the
normalized query (longest match wins). Traversal is a bounded breadth-first
expansion. There is no LLM-generated query language anywhere in the retrieval
path, so there is nothing to inject into. When Neo4j is added, only vetted,
parameterized Cypher templates will be used.

Graph retrieval returns *chunks* (those mentioning reached entities, ranked by hop
distance), so graph and vector results share a currency and can be fused by ID.

## 4. Reciprocal Rank Fusion
RRF needs no score calibration between a cosine similarity and a hop distance.
Score = sum over lists of `1 / (k + rank)`, `k = 60`.

## 5. The agent loop is bounded, and the bound is tested
`decide_after_grade` is a pure function. At most `max_iterations` rewrites, then at
most one fallback to the other retriever, then an explicit "insufficient evidence"
answer. `test_grader_loop_always_terminates` simulates an always-rejecting grader
for every route and iteration setting.

## 6. LLM output is never trusted
Router, grader and rewriter degrade to rule-based behaviour on invalid output or
backend errors, and the trace records which path was taken (`source: llm|rules`).
Extraction is schema-validated with error feedback on retry. Citations returned by
the generator are filtered to chunk IDs that were actually retrieved; if none
survive, the response says `grounded: false` rather than hiding it.

## 7. No LangChain
Only `langgraph` (workflow) and `httpx` (OpenAI-compatible HTTP) are used for
orchestration and LLM access. Fewer moving parts and easier to read.
