"""Stage-wise evaluation and stratified human review for find_papers results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from hybrid_retrieval import Candidate, PaperHit, QuerySpec
from paper_review import FindRequest, FindResult, PaperVerdict


CENTRALITIES = ("core", "background", "irrelevant", "uncertain")
STANCES = ("support", "oppose", "mixed", "neutral", "uncertain", "not_applicable")


def load_result(path: Path) -> FindResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for raw in data["candidates"]:
        hits = [PaperHit(**hit) for hit in raw.get("hits", [])]
        candidates.append(
            Candidate(
                dockey=raw["dockey"],
                rrf_score=raw.get("rrf_score", 0.0),
                best_rank=raw.get("best_rank", 0),
                hits=hits,
            )
        )
    return FindResult(
        request=FindRequest(**data["request"]),
        queries=[QuerySpec(**query) for query in data["queries"]],
        candidates=candidates,
        verdicts={
            dockey: PaperVerdict(**verdict)
            for dockey, verdict in data["verdicts"].items()
        },
        coverage=data.get("coverage", {}),
        errors=data.get("errors", []),
    )


def load_labels(path: Path, question: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    labels = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if "dockey" not in row:
            raise ValueError(f"Label line {line_number} has no dockey")
        if question is None or row.get("question") == question:
            labels.append(row)
    return labels


def final_label(row: dict[str, Any]) -> str | None:
    """Only user/final adjudicated labels are evaluation truth; AI labels are silver."""
    value = row.get("final_label") or row.get("user_label")
    if isinstance(value, dict):
        value = value.get("centrality")
    return str(value) if value else None


def final_stance(row: dict[str, Any]) -> str | None:
    value = row.get("final_stance") or row.get("user_stance")
    if isinstance(value, dict):
        value = value.get("stance")
    return str(value) if value else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def recall_at_k(ranking: Sequence[str], gold: set[str], k: int) -> float | None:
    if not gold:
        return None
    return len(gold.intersection(ranking[:k])) / len(gold)


def compute_metrics(
    result: FindResult, labels: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compute confirmed-label metrics and attribute every confirmed core miss."""
    result.validate_accounting()
    truth = {row["dockey"]: final_label(row) for row in labels if final_label(row)}
    rows = {row["dockey"]: row for row in labels if final_label(row)}
    gold_core = {dockey for dockey, label in truth.items() if label == "core"}
    indexed_gold_core = {
        dockey for dockey in gold_core if rows[dockey].get("indexed", True)
    }
    candidates = [candidate.dockey for candidate in result.candidates]
    candidate_set = set(candidates)
    main = {
        dockey
        for dockey, verdict in result.verdicts.items()
        if verdict.review_status == "completed" and verdict.centrality == "core"
    }
    labeled_main = main.intersection(truth)
    true_positive = len(labeled_main.intersection(gold_core))
    candidate_hits = len(indexed_gold_core.intersection(candidate_set))
    final_hits = len(gold_core.intersection(main))

    confusion: dict[str, dict[str, int]] = {
        actual: {predicted: 0 for predicted in (*CENTRALITIES, "not_candidate")}
        for actual in CENTRALITIES
    }
    for dockey, actual in truth.items():
        if actual not in confusion:
            continue
        predicted = (
            result.verdicts[dockey].centrality
            if dockey in candidate_set
            else "not_candidate"
        )
        confusion[actual][predicted] += 1

    gold_background = {dockey for dockey, label in truth.items() if label == "background"}
    background_to_core = len(gold_background.intersection(main))
    unjudged = sum(
        verdict.review_status != "completed" for verdict in result.verdicts.values()
    )
    misses = []
    for dockey in sorted(gold_core - main):
        if not rows[dockey].get("indexed", True):
            stage = "not_indexed"
        elif dockey not in candidate_set:
            stage = "not_candidate"
        elif result.verdicts[dockey].review_status != "completed":
            stage = "pending_or_api_error"
        else:
            stage = "review_false_negative"
        misses.append({"dockey": dockey, "stage": stage})

    channel_sets: dict[str, set[str]] = defaultdict(set)
    for candidate in result.candidates:
        for hit in candidate.hits:
            channel_sets[hit.channel].add(candidate.dockey)
    channel_recall = {
        channel: _ratio(len(indexed_gold_core.intersection(dockeys)), len(indexed_gold_core))
        for channel, dockeys in sorted(channel_sets.items())
    }

    stance_pairs = []
    for dockey in sorted(gold_core.intersection(main)):
        expected = final_stance(rows[dockey])
        if expected:
            stance_pairs.append((expected, result.verdicts[dockey].stance))
    stance_f1 = {}
    for stance in STANCES:
        tp = sum(expected == stance and predicted == stance for expected, predicted in stance_pairs)
        fp = sum(expected != stance and predicted == stance for expected, predicted in stance_pairs)
        fn = sum(expected == stance and predicted != stance for expected, predicted in stance_pairs)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        stance_f1[stance] = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
    present_stances = {expected for expected, _ in stance_pairs}
    macro_f1 = (
        sum(stance_f1[stance] for stance in present_stances) / len(present_stances)
        if present_stances
        else None
    )
    return {
        "question": result.request.raw_question,
        "label_scope": "user_or_final_labels_only",
        "n_confirmed_labels": len(truth),
        "n_gold_core": len(gold_core),
        "index_coverage": _ratio(len(indexed_gold_core), len(gold_core)),
        "candidate_core_recall_indexed": _ratio(candidate_hits, len(indexed_gold_core)),
        "candidate_core_recall_end_to_end": _ratio(candidate_hits, len(gold_core)),
        "final_core_recall_indexed": _ratio(
            len(indexed_gold_core.intersection(main)), len(indexed_gold_core)
        ),
        "final_core_recall_end_to_end": _ratio(final_hits, len(gold_core)),
        "main_core_precision": _ratio(true_positive, len(labeled_main)),
        "main_core_precision_wilson95": wilson_interval(true_positive, len(labeled_main)),
        "background_to_core_count": background_to_core,
        "background_to_core_rate": _ratio(background_to_core, len(gold_background)),
        "unjudged": unjudged,
        "render_drop": sum(
            verdict.review_status == "completed" and verdict.centrality == "core"
            for verdict in result.verdicts.values()
        ) - len(main),
        "recall_at_k": {
            str(k): recall_at_k(candidates, indexed_gold_core, k) for k in (50, 100, 200)
        },
        "channel_recall": channel_recall,
        "centrality_confusion": confusion,
        "stance_n": len(stance_pairs),
        "stance_accuracy": _ratio(
            sum(expected == predicted for expected, predicted in stance_pairs),
            len(stance_pairs),
        ),
        "stance_macro_f1": macro_f1,
        "core_misses": misses,
        "run_complete": result.is_complete,
        "run_errors": result.errors,
    }


def build_stratified_records(
    result: FindResult,
    all_dockeys: Iterable[str] | None = None,
    labels: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Create review strata, including non-candidates when the corpus keys are supplied."""
    label_by_key = {row["dockey"]: row for row in labels}
    records = []
    candidate_keys = {candidate.dockey for candidate in result.candidates}
    for candidate in result.candidates:
        verdict = result.verdicts[candidate.dockey]
        channels = {hit.channel for hit in candidate.hits}
        if verdict.review_status != "completed" or verdict.centrality == "uncertain":
            stratum = "pending_or_uncertain"
        elif dockey_in_labels(label_by_key, candidate.dockey):
            expected = final_label(label_by_key[candidate.dockey])
            stratum = (
                "ai_system_disagreement"
                if expected and expected != verdict.centrality
                else verdict.centrality
            )
        elif channels == {"vector"}:
            stratum = "vector_only"
        elif channels and channels <= {"keyword", "phrase"}:
            stratum = "bm25_only"
        else:
            stratum = verdict.centrality
        records.append(
            {
                "question": result.request.raw_question,
                "dockey": candidate.dockey,
                "docname": verdict.docname,
                "sample_stratum": stratum,
                "predicted_centrality": verdict.centrality,
                "predicted_stance": verdict.stance,
                "confidence": verdict.confidence,
                "is_candidate": True,
            }
        )
    for dockey in sorted(set(all_dockeys or ()) - candidate_keys):
        records.append(
            {
                "question": result.request.raw_question,
                "dockey": dockey,
                "docname": dockey,
                "sample_stratum": "non_candidate",
                "predicted_centrality": "not_candidate",
                "predicted_stance": "not_applicable",
                "confidence": 0.0,
                "is_candidate": False,
            }
        )
    return records


def dockey_in_labels(labels: dict[str, Any], dockey: str) -> bool:
    return dockey in labels and bool(final_label(labels[dockey]))


def stratified_sample(
    records: Sequence[dict[str, Any]], total: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Sample every stratum deterministically and retain inclusion probabilities."""
    if total <= 0:
        raise ValueError("Sample size must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["sample_stratum"]].append(record)
    if not groups:
        return []
    rng = random.Random(seed)
    quota = max(1, total // len(groups))
    selected: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        population = sorted(groups[stratum], key=lambda row: row["dockey"])
        # Low-confidence and disagreement cases are always included before random sampling.
        priority = [
            row for row in population
            if row["sample_stratum"] == "ai_system_disagreement" or row["confidence"] < 0.6
        ]
        remainder = [row for row in population if row not in priority]
        chosen = priority[:quota]
        if len(chosen) < quota:
            chosen.extend(rng.sample(remainder, min(quota - len(chosen), len(remainder))))
        for row in chosen:
            item = dict(row)
            item["sampling_probability"] = len(chosen) / len(population)
            item["sample_weight"] = len(population) / len(chosen)
            item["ai_label"] = ""
            item["user_label"] = ""
            item["final_label"] = ""
            selected.append(item)
    return selected


def export_review_cards(
    sample: Sequence[dict[str, Any]], evidence_path: Path, output_path: Path
) -> None:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    cards = []
    for row in sample:
        pack = evidence.get(row["dockey"], {})
        cards.append(
            {
                **row,
                "neutral_evidence": [
                    {
                        "chunk_name": span["chunk_name"],
                        "pages": span["pages"],
                        "section": span["section"],
                        "text": span["text"],
                    }
                    for span in pack.get("spans", [])
                ],
                "instruction": "先仅根据原文证据判断core/background/irrelevant/uncertain，再查看AI理由",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def merge_ai_prelabels(
    existing: Sequence[dict[str, Any]], result: FindResult
) -> list[dict[str, Any]]:
    """Import strict-corpus verdicts as silver labels without overwriting human truth."""
    if not result.coverage.get("strict_corpus"):
        raise ValueError("AI full-corpus prelabels require a --strict-corpus result")
    merged = {
        (row.get("question"), row["dockey"]): dict(row) for row in existing
    }
    for candidate in result.candidates:
        verdict = result.verdicts[candidate.dockey]
        key = result.request.raw_question, candidate.dockey
        row = merged.get(key, {"question": key[0], "dockey": candidate.dockey})
        row.update(
            {
                "ai_label": verdict.centrality,
                "ai_stance": verdict.stance,
                "ai_confidence": verdict.confidence,
                "ai_reason": verdict.reason,
                "ai_evidence_ids": verdict.evidence_ids,
                "ai_review_status": verdict.review_status,
                "indexed": True,
            }
        )
        merged[key] = row
    return [merged[key] for key in sorted(merged, key=lambda item: (str(item[0]), item[1]))]


def write_labels(path: Path, labels: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="find_papers 运行目录中的 result.json")
    parser.add_argument("--labels", type=Path, default=Path("eval/find_papers_labels.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-dockeys", type=Path, help="JSON列表或每行一个dockey，用于非候选抽查")
    parser.add_argument("--sample-out", type=Path)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--import-ai-labels",
        action="store_true",
        help="将strict全库结果导入为AI silver标签，保留用户/final标签",
    )
    args = parser.parse_args()
    result = load_result(args.result)
    if args.import_ai_labels:
        merged = merge_ai_prelabels(load_labels(args.labels), result)
        write_labels(args.labels, merged)
        print(f"AI silver标签已写入：{args.labels}（{len(merged)}条）")
        return
    labels = load_labels(args.labels, result.request.raw_question)
    metrics = compute_metrics(result, labels)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.sample_out:
        all_dockeys = None
        dockeys_path = args.all_dockeys or args.result.with_name("corpus_dockeys.json")
        if dockeys_path.exists():
            raw = dockeys_path.read_text(encoding="utf-8")
            try:
                all_dockeys = json.loads(raw)
            except json.JSONDecodeError:
                all_dockeys = raw.splitlines()
        records = build_stratified_records(result, all_dockeys, labels)
        sample = stratified_sample(records, args.sample_size, args.seed)
        export_review_cards(sample, args.result.with_name("evidence.json"), args.sample_out)
        print(f"抽查证据卡：{args.sample_out}（{len(sample)}篇）")


if __name__ == "__main__":
    main()
