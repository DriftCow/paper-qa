"""Deterministic paper-level BM25/vector retrieval for local literature screening."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Literal, Sequence

import numpy as np
from lmi import EmbeddingModes

from cjk_search import cjk_bigrams


Channel = Literal["both", "keyword", "phrase", "vector"]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    text: str
    channel: Channel = "both"
    language: str = "unknown"
    family: str = "original"


@dataclass(frozen=True)
class PaperHit:
    dockey: str
    query_id: str
    channel: str
    rank: int
    score: float
    chunk_ids: tuple[str, ...] = ()


@dataclass
class Candidate:
    dockey: str
    rrf_score: float = 0.0
    best_rank: int = 0
    hits: list[PaperHit] = field(default_factory=list)

    @property
    def best_chunk_ids(self) -> list[str]:
        out: dict[str, None] = {}
        for hit in sorted(self.hits, key=lambda h: (h.rank, h.query_id, h.channel)):
            for chunk_id in hit.chunk_ids:
                out.setdefault(chunk_id)
        return list(out)


@dataclass
class HybridResult:
    candidates: list[Candidate]
    errors: list[str] = field(default_factory=list)

    @property
    def dockeys(self) -> list[str]:
        return [candidate.dockey for candidate in self.candidates]


def stable_chunk_id(text: Any) -> str:
    """Return a process-stable ID for a PaperQA Text-like object."""
    raw = "\0".join(
        (str(text.doc.dockey), str(text.name), str(text.text))
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[PaperHit | str]], constant: int = 60
) -> list[Candidate]:
    """Fuse complete paper rankings without applying a post-fusion top-N."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    candidates: dict[str, Candidate] = {}
    for ranking in rankings:
        seen: set[str] = set()
        clean_rank = 0
        for item in ranking:
            dockey = item if isinstance(item, str) else item.dockey
            if dockey in seen:
                continue
            seen.add(dockey)
            clean_rank += 1
            candidate = candidates.setdefault(
                dockey, Candidate(dockey=dockey, best_rank=clean_rank)
            )
            candidate.rrf_score += 1.0 / (constant + clean_rank)
            candidate.best_rank = min(candidate.best_rank, clean_rank)
            if not isinstance(item, str):
                candidate.hits.append(item)
    return sorted(
        candidates.values(),
        key=lambda c: (-c.rrf_score, c.best_rank, c.dockey),
    )


class DensePaperRetriever:
    """Exact cosine retrieval that aggregates all chunks by paper before top-N."""

    def __init__(self, texts: Sequence[Any], best_chunks_per_paper: int = 3):
        if not texts:
            raise ValueError("Cannot build a dense retriever from an empty corpus")
        if best_chunks_per_paper <= 0:
            raise ValueError("best_chunks_per_paper must be positive")
        self.texts = list(texts)
        self.best_chunks_per_paper = best_chunks_per_paper
        try:
            matrix = np.asarray([text.embedding for text in self.texts], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("All chunks must have equally sized numeric embeddings") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(self.texts) or matrix.shape[1] == 0:
            raise ValueError("All chunks must have equally sized numeric embeddings")
        if not np.isfinite(matrix).all():
            raise ValueError("Chunk embeddings contain NaN or infinite values")
        self._norms = np.linalg.norm(matrix, axis=1)
        self._matrix = np.zeros_like(matrix, dtype=np.float32)
        nonzero = self._norms > 0
        self._matrix[nonzero] = matrix[nonzero] / self._norms[nonzero, None]
        self.chunk_ids = [stable_chunk_id(text) for text in self.texts]
        self.chunk_by_id = dict(zip(self.chunk_ids, self.texts, strict=True))

    @property
    def dimensions(self) -> int:
        return int(self._matrix.shape[1])

    def rank_vector(
        self, query_vector: Sequence[float], query_id: str, depth: int
    ) -> list[PaperHit]:
        if depth <= 0:
            raise ValueError("Retrieval depth must be positive")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.dimensions,) or not np.isfinite(query).all():
            raise ValueError("Query embedding has an incompatible shape or invalid values")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("Query embedding must not be a zero vector")
        scores = self._matrix @ (query / norm)
        scores[self._norms == 0] = -np.inf

        best: dict[str, tuple[float, str, list[tuple[float, str]]]] = {}
        for index, text in enumerate(self.texts):
            dockey = str(text.doc.dockey)
            score = float(scores[index])
            chunk_id = self.chunk_ids[index]
            if dockey not in best:
                best[dockey] = (score, chunk_id, [(score, chunk_id)])
                continue
            best_score, best_chunk, chunks = best[dockey]
            chunks.append((score, chunk_id))
            if score > best_score or (score == best_score and chunk_id < best_chunk):
                best[dockey] = (score, chunk_id, chunks)
        if not best:
            return []
        ordered = sorted(
            best.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
        hits: list[PaperHit] = []
        for rank, (dockey, (score, _, chunks)) in enumerate(ordered[:depth], start=1):
            if not np.isfinite(score):
                continue
            top_chunks = sorted(chunks, key=lambda item: (-item[0], item[1]))[
                : self.best_chunks_per_paper
            ]
            hits.append(
                PaperHit(
                    dockey=dockey,
                    query_id=query_id,
                    channel="vector",
                    rank=rank,
                    score=score,
                    chunk_ids=tuple(chunk_id for _, chunk_id in top_chunks),
                )
            )
        return hits

    async def rank(
        self, query: str, embedding_model: Any, query_id: str, depth: int
    ) -> list[PaperHit]:
        embedding_model.set_mode(EmbeddingModes.QUERY)
        try:
            vector = (await embedding_model.embed_documents([query]))[0]
        finally:
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)
        return self.rank_vector(vector, query_id, depth)


class TantivyDocRetriever:
    """Read only Tantivy hits without deserializing each saved Docs object."""

    def __init__(self, index: Any):
        self.index = index
        self.fields = [field for field in index.fields if field != "year"]

    async def search(self, spec: QuerySpec, depth: int) -> list[PaperHit]:
        if spec.channel not in {"keyword", "phrase"}:
            raise ValueError("Tantivy search requires a keyword or phrase QuerySpec")
        if depth <= 0:
            raise ValueError("Retrieval depth must be positive")
        from paperqa.agents.search import SearchIndex

        query = cjk_bigrams(spec.text)
        if spec.channel == "keyword":
            query = SearchIndex.CLEAN_QUERY_REGEX.sub("", query)
        searcher = await self.index.searcher
        tantivy_index = await self.index.index
        parsed = tantivy_index.parse_query(query, self.fields)
        hits = searcher.search(parsed, depth).hits
        out: list[PaperHit] = []
        for score, address in hits:
            if score <= 0:
                continue
            document = searcher.doc(address)
            dockey = document["file_location"][0]
            out.append(
                PaperHit(
                    dockey=dockey,
                    query_id=spec.query_id,
                    channel=spec.channel,
                    rank=len(out) + 1,
                    score=float(score),
                )
            )
        return out


class HybridRetriever:
    """Execute a fixed query plan; one failed channel never erases other hits."""

    def __init__(self, dense: DensePaperRetriever, tantivy: TantivyDocRetriever):
        self.dense = dense
        self.tantivy = tantivy

    async def search(
        self, queries: Sequence[QuerySpec], embedding_model: Any, depth: int = 80
    ) -> HybridResult:
        rankings: list[list[PaperHit]] = []
        errors: list[str] = []
        for spec in queries:
            channels = ("keyword", "vector") if spec.channel == "both" else (spec.channel,)
            for channel in channels:
                channel_spec = QuerySpec(
                    query_id=spec.query_id,
                    text=spec.text,
                    channel=channel,
                    language=spec.language,
                    family=spec.family,
                )
                try:
                    if channel == "vector":
                        ranking = await self.dense.rank(
                            spec.text, embedding_model, spec.query_id, depth
                        )
                    else:
                        ranking = await self.tantivy.search(channel_spec, depth)
                    rankings.append(ranking)
                except Exception as exc:  # Keep partial results and make incompleteness visible.
                    errors.append(f"{spec.query_id}/{channel}: {type(exc).__name__}: {exc}")
        return HybridResult(reciprocal_rank_fusion(rankings), errors)
