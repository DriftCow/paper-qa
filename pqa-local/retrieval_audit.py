r"""Retrieval audit trail + multi-query pre-seeding for FIND questions.

Root cause this module addresses (verified against the live index): with
1100+ papers, BM25 keyword queries under OR semantics never surface a short
4-page target into the top-N (not even top-40), so a directly relevant paper
could vanish from every candidate list with no trace.  Three fixes live here:

1. **Audit hook** -- ``RetrievalAudit.install_hook()`` patches
   ``paperqa.agents.tools.get_directory_index`` (the module-level reference
   used by the ``paper_search`` tool) so every agent search records its
   query, top_n, offset and the ranked candidates (dockey/docname/title).
   Pre-seeded searches (below) are recorded with source ``find_preseed``.
2. **Multi-query pre-seeding** -- ``run_retrieval_plans()`` executes a plan
   from ``task_router.find_retrieval_plans`` (keyword + phrase angles) against
   the same Tantivy index and merges the hits into the session's ``Docs``
   pool *before* the agent starts.  Phrase queries keep their quotes by
   calling ``Index.parse_query`` directly (``SearchIndex.query`` strips
   them), which is what makes exact formulations like ``"the claim's
   verbatim wording"`` hit rank 1 while keyword queries miss entirely.  Stored
   texts already carry embeddings, so keyword/phrase pre-seeding costs zero
   API calls.
3. **Vector pre-seeding** -- ``run_vector_query()`` adds the original
   natural-language question's dense retrieval ranking to the same candidate
   pool. It costs one query embedding and never generates an answer.

``finalize()`` closes the loop after a run: candidates, papers that entered
evidence, papers cited in the final answer, and every other candidate get an
exclusion reason taken from what actually happened ("未说明" when the pipeline
does not expose one -- reasons are never invented).  ``save()`` writes JSON
under ``audit/``; no API keys and no full paper texts are ever stored (only
docname/dockey/title, ranks, and 200-char evidence excerpts).

Invariant relied upon: evidence papers ⊆ candidates, because ``Docs``
is only filled via ``aadd_texts`` from the audited ``paper_search`` and the
audited pre-seed.  Any violation is reported as a warning instead of being
silently dropped.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SearchRecord:
    """One executed search: query + ranked candidates."""

    source: str  # "find_preseed" | "agent_paper_search"
    mode: str  # "keyword" | "phrase" | "vector"
    query: str
    top_n: int
    offset: int
    candidates: list[dict[str, Any]] = field(default_factory=list)
    n_added: int = 0  # papers newly merged into the Docs pool (preseed only)
    error: str | None = None


def _result_candidates(results: list, offset: int = 0) -> list[dict[str, Any]]:
    """Rank/dockey/docname/title for a ``list[Docs]`` search result."""
    out: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        doc = next(iter(r.docs.values()), None)
        if doc is None:
            continue
        out.append(
            {
                "rank": offset + i + 1,
                "dockey": getattr(doc, "dockey", None),
                "docname": getattr(doc, "docname", None),
                "title": getattr(doc, "title", None),
            }
        )
    return out


class _AuditedSearchIndex:
    """Proxy forwarding to the real index while recording every query."""

    def __init__(self, real_index: Any, audit: "RetrievalAudit") -> None:
        self._real = real_index
        self._audit = audit

    async def query(
        self,
        query: str,
        top_n: int = 10,
        offset: int = 0,
        min_score: float = 0.0,
        keep_filenames: bool = False,
        field_subset: list[str] | None = None,
    ) -> list[Any]:
        results = await self._real.query(
            query,
            top_n=top_n,
            offset=offset,
            min_score=min_score,
            keep_filenames=keep_filenames,
            field_subset=field_subset,
        )
        self._audit.record_search(
            source="agent_paper_search",
            mode="keyword",
            query=query,
            top_n=top_n,
            offset=offset,
            candidates=_result_candidates(results, offset=offset),
        )
        return results

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


async def phrase_query(
    index: Any,
    query: str,
    top_n: int,
    offset: int = 0,
    field_subset: list[str] | None = None,
) -> list[Any]:
    """Phrase query that survives ``SearchIndex.query``'s quote stripping.

    ``SearchIndex.CLEAN_QUERY_REGEX`` removes ``"`` before parsing, so a
    phrase query routed through ``query()`` degrades to an OR keyword query.
    Here the quotes are kept and ``Index.parse_query`` builds a real phrase
    clause.  CJK runs are bigram-segmented first (mirroring the class-level
    patch installed by ``cjk_search.install_cjk_query_patch``).
    """
    from cjk_search import cjk_bigrams

    query_fields = list(field_subset or index.fields)
    searcher = await index.searcher
    tantivy_index = await index.index
    try:
        parsed = tantivy_index.parse_query(cjk_bigrams(query), query_fields)
    except ValueError:
        return []  # rejected by tantivy (e.g. unmatched quote)
    hits = searcher.search(parsed, top_n, offset=offset).hits
    out = []
    for score, address in hits:
        if score <= 0.0:
            continue
        doc = searcher.doc(address)
        obj = await index.get_saved_object(doc["file_location"][0])
        if obj is not None:
            out.append(obj)
    return out


class RetrievalAudit:
    """Collects search records and derives the per-paper audit summary."""

    def __init__(self) -> None:
        self.searches: list[SearchRecord] = []
        self._restore_fn: Any = None

    # -- recording -------------------------------------------------------------
    def record_search(
        self,
        source: str,
        mode: str,
        query: str,
        top_n: int,
        offset: int,
        candidates: list[dict[str, Any]],
        n_added: int = 0,
        error: str | None = None,
    ) -> None:
        self.searches.append(
            SearchRecord(
                source=source,
                mode=mode,
                query=query,
                top_n=top_n,
                offset=offset,
                candidates=candidates,
                n_added=n_added,
                error=error,
            )
        )

    # -- paper_search hook -------------------------------------------------------
    def install_hook(self) -> None:
        """Audit every agent ``paper_search`` call.

        Patches the module-level ``get_directory_index`` reference inside
        ``paperqa.agents.tools`` (the one ``PaperSearch.paper_search`` uses).
        ``paperqa.agents.main``'s independent reference is untouched.
        """
        if self._restore_fn is not None:
            return
        import paperqa.agents.tools as tools_module

        original = tools_module.get_directory_index
        audit = self

        async def audited_get_directory_index(settings=None, build=True):
            index = await original(settings, build=build)
            return _AuditedSearchIndex(index, audit)

        tools_module.get_directory_index = audited_get_directory_index

        def _restore() -> None:
            tools_module.get_directory_index = original
            audit._restore_fn = None

        self._restore_fn = _restore

    def uninstall_hook(self) -> None:
        if self._restore_fn is not None:
            self._restore_fn()

    # -- multi-query pre-seeding -------------------------------------------------
    async def run_retrieval_plans(
        self,
        docs: Any,
        plans: list[tuple[str, str]],
        settings: Any,
        top_n: int | None = None,
    ) -> int:
        """Execute (mode, query) plans and pre-seed candidates into ``docs``.

        Returns the number of newly added papers.  Papers already in the pool
        are skipped by ``Docs.aadd_texts`` (returns False).  Stored texts
        carry embeddings, so no embedding API calls are made.
        """
        if not plans:
            return 0
        from paperqa.agents.search import SearchIndex

        index = SearchIndex(
            fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
            index_name=settings.agent.index.name or settings.get_index_name(),
            index_directory=settings.agent.index.index_directory,
        )
        n = top_n if top_n is not None else settings.agent.search_count
        field_subset = [f for f in index.fields if f != "year"]
        added = 0
        for mode, query in plans:
            error = None
            results: list[Any] = []
            try:
                if mode == "phrase":
                    results = await phrase_query(
                        index, query, top_n=n, field_subset=field_subset
                    )
                else:
                    results = await index.query(
                        query, top_n=n, offset=0, field_subset=field_subset
                    )
            except Exception as exc:  # noqa: BLE001 - a failed angle is recorded, not fatal
                error = f"{type(exc).__name__}: {exc}"
            n_added = 0
            for r in results:
                doc_details = next(iter(r.docs.values()), None)
                if doc_details is None:
                    continue
                if await docs.aadd_texts(
                    texts=r.texts, doc=doc_details, settings=settings
                ):
                    n_added += 1
            added += n_added
            self.record_search(
                source="find_preseed",
                mode=mode,
                query=query,
                top_n=n,
                offset=0,
                candidates=_result_candidates(results),
                n_added=n_added,
                error=error,
            )
        return added

    async def run_vector_query(
        self,
        docs: Any,
        corpus_docs: Any,
        query: str,
        settings: Any,
        top_n: int | None = None,
    ) -> int:
        """Add a natural-language vector ranking to the candidate pool."""
        n = top_n if top_n is not None else settings.agent.search_count
        candidates: list[dict[str, Any]] = []
        n_added = 0
        error = None
        try:
            texts = await corpus_docs.retrieve_texts(
                query, k=min(len(corpus_docs.texts), n * 4), settings=settings
            )
            dockeys = list(dict.fromkeys(text.doc.dockey for text in texts))[:n]
            for rank, dockey in enumerate(dockeys, start=1):
                doc = corpus_docs.docs[dockey]
                candidates.append(
                    {
                        "rank": rank,
                        "dockey": dockey,
                        "docname": doc.docname,
                        "title": getattr(doc, "title", None),
                    }
                )
                doc_texts = [
                    text for text in corpus_docs.texts if text.doc.dockey == dockey
                ]
                if await docs.aadd_texts(
                    texts=doc_texts, doc=doc, settings=settings
                ):
                    n_added += 1
        except Exception as exc:  # noqa: BLE001 - keyword retrieval can still proceed
            error = f"{type(exc).__name__}: {exc}"
        self.record_search(
            source="find_preseed",
            mode="vector",
            query=query,
            top_n=n,
            offset=0,
            candidates=candidates,
            n_added=n_added,
            error=error,
        )
        return n_added

    # -- post-run summary ---------------------------------------------------------
    def finalize(
        self,
        session: Any,
        cited_docnames: list[str],
        coverage_reasons: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Derive the per-paper audit summary from what actually happened.

        Stage values (mutually exclusive, union = all candidates):
          cited              -- cited in the final answer body
          evidence_not_cited -- entered gather_evidence, not cited in the answer
          candidate_only     -- in the pool, gather_evidence never surfaced it
        Exclusion reasons describe the observed pipeline outcome; anything the
        pipeline does not expose is marked 未说明, never invented.
        """
        name_to_key: dict[str, str] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for ctx in getattr(session, "contexts", []) or []:
            text = getattr(ctx, "text", None)
            doc = getattr(text, "doc", None)
            if doc is None:
                continue
            name_to_key[doc.docname] = doc.dockey
            e = evidence.setdefault(
                doc.dockey,
                {
                    "docname": doc.docname,
                    "n_contexts": 0,
                    "max_score": 0,
                    "chunks": [],
                },
            )
            e["n_contexts"] += 1
            e["max_score"] = max(e["max_score"], getattr(ctx, "score", 0) or 0)
            e["chunks"].append(
                {
                    "chunk": text.name,
                    "score": getattr(ctx, "score", 0) or 0,
                    "excerpt": (getattr(ctx, "context", "") or "")[:200],
                }
            )

        candidates: dict[str, dict[str, Any]] = {}
        for s in self.searches:
            if s.error or not s.candidates:
                continue
            for c in s.candidates:
                key = c["dockey"]
                if key is None:
                    continue
                d = candidates.setdefault(
                    key,
                    {
                        "docname": c["docname"],
                        "title": c.get("title"),
                        "best_rank": None,
                        "hit_by": [],
                    },
                )
                if d["best_rank"] is None or c["rank"] < d["best_rank"]:
                    d["best_rank"] = c["rank"]
                d["hit_by"].append(
                    {
                        "query": s.query,
                        "mode": s.mode,
                        "source": s.source,
                        "rank": c["rank"],
                    }
                )

        cited_keys = {name_to_key[n] for n in cited_docnames if n in name_to_key}

        papers: list[dict[str, Any]] = []
        for key, d in sorted(candidates.items(), key=lambda kv: kv[1]["docname"] or ""):
            ev = evidence.get(key)
            if key in cited_keys:
                stage, reason = "cited", "已进入最终答案(正文引用)"
            elif ev:
                stage = "evidence_not_cited"
                # Coverage-check verdicts (dockey-keyed) are evidence-grounded
                # LLM judgements; without them the pipeline exposes no reason,
                # which is stated honestly instead of being invented.
                reason = (coverage_reasons or {}).get(key) or (
                    f"进入证据({ev['n_contexts']}个片段,最高相关分{ev['max_score']})"
                    "但未被最终答案正文引用;原因未说明(由答案生成阶段决定)"
                )
            else:
                stage = "candidate_only"
                reason = (
                    "检索候选已进入语料池,但gather_evidence未取到其文本块"
                    "(向量检索未命中);具体原因未说明"
                )
            papers.append(
                {
                    "dockey": key,
                    "docname": d["docname"],
                    "title": d["title"],
                    "best_rank": d["best_rank"],
                    "hit_by": d["hit_by"],
                    "stage": stage,
                    "exclusion_reason": reason,
                    "n_evidence_chunks": ev["n_contexts"] if ev else 0,
                }
            )

        warnings = []
        for key in evidence:
            if key not in candidates:
                warnings.append(
                    f"证据中出现 dockey={key} 但不在任何检索候选记录中(审计不完整)"
                )
        for docname in cited_docnames:
            if docname not in name_to_key:
                warnings.append(
                    f"答案引用了 {docname} 但该文献不在本轮证据上下文中(引用来源未知)"
                )

        return {
            "papers": papers,
            "evidence": sorted(evidence.values(), key=lambda e: e["docname"] or ""),
            "cited_docnames": list(cited_docnames),
            "counts": {
                "n_searches": len(self.searches),
                "n_candidate_papers": len(candidates),
                "n_evidence_papers": len(evidence),
                "n_cited_papers": len(cited_keys),
                "n_evidence_not_cited": sum(
                    1 for p in papers if p["stage"] == "evidence_not_cited"
                ),
                "n_candidate_only": sum(
                    1 for p in papers if p["stage"] == "candidate_only"
                ),
            },
            "warnings": warnings,
        }

    def to_dict(
        self,
        question: str,
        intent: str,
        session: Any,
        cited_docnames: list[str],
        coverage_reasons: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        summary = self.finalize(session, cited_docnames, coverage_reasons=coverage_reasons)
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": question,
            "intent": intent,
            "searches": [asdict(s) for s in self.searches],
            **summary,
        }

    def save(self, payload: dict[str, Any], path: Path) -> Path:
        """Write the audit JSON (no API keys, no full paper texts)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path
