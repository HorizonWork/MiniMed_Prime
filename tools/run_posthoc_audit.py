#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from judges.posthoc_auditor import PostHocAuditor, render_posthoc_audit_markdown  # noqa: E402
from src.layers.layer1_retrieval import AgenticRetriever  # noqa: E402
from src.schemas import EvidenceBundle  # noqa: E402
from src.training.medreason_adapter import (  # noqa: E402
    DEFAULT_MEDREASON_SOURCE,
    load_medreason_records,
    normalize_medreason_record,
)
from src.utils.kaggle_env import KaggleEnv, T4Hardening  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run post-hoc answer/reasoning audit against retrieved evidence.")
    parser.add_argument("--trace-file", type=Path, default=None, help="Existing trace JSON file.")
    parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE, help="HF dataset name or local MedReason file/dir.")
    parser.add_argument("--split", default="train", help="Dataset split; use 'none' for local unsplit files.")
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--random-one", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-answer", default=None)
    parser.add_argument("--current-reasoning-output", default=None)
    parser.add_argument("--mode", choices=("answer_only", "answer_plus_cot"), default="answer_only")
    parser.add_argument("--auditor-model", default="heuristic")
    parser.add_argument("--auditor-device", default="cpu")
    parser.add_argument("--primekg-path", type=Path, default=Path("data/kg/primekg"))
    parser.add_argument("--pubmed-cache-path", type=Path, default=Path("data/pubmed_cache.jsonl"))
    parser.add_argument("--pubmed-api-key", default=None)
    parser.add_argument("--hard-constraints-json", default=None, help="JSON object string.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/audits"))
    parser.add_argument("--embed-in-trace", action="store_true", help="When --trace-file is used, write audit result back into trace JSON.")
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args(argv)

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    hard_constraints = _parse_hard_constraints(args.hard_constraints_json)
    output_dir = KaggleEnv.ensure_writeable(args.output_dir if args.output_dir.is_absolute() else KaggleEnv.path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trace_file is not None:
        trace_payload = _load_trace_payload(args.trace_file)
        sample_id, question, evidence_bundle, candidate_answer, reasoning_output, gold_cot_steps = _resolve_from_trace_payload(
            trace_payload=trace_payload,
            candidate_answer_override=args.candidate_answer,
            reasoning_override=args.current_reasoning_output,
        )
        if not hard_constraints:
            hard_constraints = _default_hard_constraints(question_type=_trace_question_type(trace_payload), question=question)
    else:
        split = None if str(args.split).lower() in {"", "none", "null", "false"} else args.split
        records = load_medreason_records(args.source, split=split)
        index, record = _select_record(
            records,
            sample_index=args.sample_index,
            question_id=args.question_id,
            random_one=args.random_one,
            seed=args.seed,
        )
        example = normalize_medreason_record(record, index=index)
        retriever = AgenticRetriever(
            primekg_path=KaggleEnv.path(args.primekg_path),
            pubmed_api_key=args.pubmed_api_key,
            pubmed_cache_path=KaggleEnv.ensure_writeable(KaggleEnv.path(args.pubmed_cache_path)),
        )
        evidence_bundle = retriever.retrieve(example.question)
        sample_id = example.group_id
        question = example.question
        candidate_answer = args.candidate_answer or (example.answer if example.answer.strip() else None)
        reasoning_output = args.current_reasoning_output or _flatten_to_text(example.reasoning)
        gold_cot_steps = _extract_steps(example.reasoning)
        if not hard_constraints:
            hard_constraints = _default_hard_constraints(question_type=evidence_bundle.question_type, question=question)

    auditor = PostHocAuditor(model_name=args.auditor_model, device=args.auditor_device)
    result = auditor.evaluate_with_diagnostics(
        question=question,
        evidence_bundle=evidence_bundle,
        candidate_answer=candidate_answer,
        current_reasoning_output=reasoning_output,
        gold_cot_steps=gold_cot_steps,
        hard_constraints=hard_constraints,
        mode=args.mode,
    )

    safe_id = _safe_sample_id(sample_id)
    audit_json_path = output_dir / f"{safe_id}.audit.json"
    audit_md_path = output_dir / f"{safe_id}.audit.md"
    audit_payload = result.to_json_dict()
    audit_payload.update(
        {
            "sample_id": sample_id,
            "question": question,
            "mode": args.mode,
            "candidate_answer": candidate_answer,
            "current_reasoning_output": reasoning_output,
            "hard_constraints": hard_constraints,
        }
    )
    audit_json_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8")
    audit_md_path.write_text(
        render_posthoc_audit_markdown(sample_id=sample_id, question=question, output=result.output),
        encoding="utf-8",
    )

    if args.trace_file is not None and args.embed_in_trace:
        trace_payload = _load_trace_payload(args.trace_file)
        trace_payload["posthoc_audit"] = {
            "status": "ok",
            "result": result.output.model_dump(mode="json"),
            "latency_ms": result.latency_ms,
            "parse_success": result.parse_success,
            "json_retry_count": result.json_retry_count,
            "auditor_model": result.auditor_model,
        }
        args.trace_file.write_text(json.dumps(trace_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "sample_id": sample_id,
                "overall_recommendation": result.output.overall_recommendation,
                "sample_label": result.output.sample_label,
                "faithfulness_score": result.output.faithfulness_score,
                "h5_present": result.output.h5_present,
                "parse_success": result.parse_success,
                "json_retry_count": result.json_retry_count,
                "audit_json": str(audit_json_path),
                "audit_md": str(audit_md_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_trace_payload(trace_file: Path) -> dict[str, Any]:
    resolved = trace_file if trace_file.is_absolute() else KaggleEnv.path(trace_file)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Trace file must contain a JSON object.")
    return payload


def _resolve_from_trace_payload(
    *,
    trace_payload: Mapping[str, Any],
    candidate_answer_override: str | None,
    reasoning_override: str | None,
) -> tuple[str, str, EvidenceBundle, str | None, str | None, list[str]]:
    sample_id = str(trace_payload.get("sample_id") or "trace_sample")
    question = str(trace_payload.get("question") or "")
    question_type = _trace_question_type(trace_payload)
    entity_rows = _mapping_get(trace_payload, "entity_extraction", "entities") or []
    edge_rows = _mapping_get(trace_payload, "primekg_retrieval", "top_edges") or []
    passage_rows = _mapping_get(trace_payload, "pubmed_retrieval", "passages") or []
    evidence_bundle = EvidenceBundle.model_validate(
        {
            "question_text": question,
            "question_type": question_type,
            "question_entities": [_trace_entity_row(entity_row) for entity_row in _list_or_empty(entity_rows)],
            "subgraph_edges": _list_or_empty(edge_rows),
            "pubmed_passages": _trace_passages(passage_rows),
            "metadata": {
                "source": "trace_file",
                "sample_id": sample_id,
            },
        }
    )
    candidate_answer = candidate_answer_override or _optional_text(trace_payload.get("candidate_answer")) or _optional_text(
        trace_payload.get("gold_answer")
    )
    reasoning_output = reasoning_override or _flatten_to_text(trace_payload.get("current_reasoning_output")) or _flatten_to_text(
        trace_payload.get("gold_reasoning")
    )
    gold_cot_steps = _extract_steps(trace_payload.get("gold_reasoning"))
    if not candidate_answer and not reasoning_output:
        reasoning_output = _path_summary_from_trace(trace_payload)
    return sample_id, question, evidence_bundle, candidate_answer, reasoning_output, gold_cot_steps


def _trace_entity_row(entity_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "surface": entity_row.get("surface"),
        "cui": entity_row.get("cui"),
        "primekg_node_id": entity_row.get("primekg_node_id"),
        "entity_type": entity_row.get("entity_type", "other"),
    }


def _trace_passages(passage_rows: Any) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for row in _list_or_empty(passage_rows):
        if not isinstance(row, Mapping):
            continue
        parsed.append(
            {
                "pmid": row.get("pmid"),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "relevance_score": float(row.get("relevance_score") or row.get("combined_score") or 0.0),
            }
        )
    return parsed


def _trace_question_type(trace_payload: Mapping[str, Any]) -> str:
    predicted = _mapping_get(trace_payload, "question_type", "predicted")
    if predicted:
        return str(predicted)
    return "other"


def _path_summary_from_trace(trace_payload: Mapping[str, Any]) -> str:
    edges = _list_or_empty(_mapping_get(trace_payload, "primekg_retrieval", "top_edges"))
    if not edges:
        return ""
    snippets: list[str] = []
    for edge in edges[:5]:
        if not isinstance(edge, Mapping):
            continue
        snippets.append(
            f"{edge.get('head', '')} {edge.get('display_relation', edge.get('relation', 'related_to'))} {edge.get('tail', '')} [edge:{edge.get('edge_id')}]"
        )
    return ". ".join(snippets)


def _select_record(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_index: int | None,
    question_id: str | None,
    random_one: bool,
    seed: int,
) -> tuple[int, Mapping[str, Any]]:
    if not records:
        raise ValueError("No records were loaded from MedReason source.")
    selector_count = int(sample_index is not None) + int(bool(question_id)) + int(random_one)
    if selector_count > 1:
        raise ValueError("Use only one selector: --sample-index, --question-id, or --random-one.")
    if question_id:
        for index, record in enumerate(records):
            if _record_id(record, index) == question_id:
                return index, record
        raise ValueError(f"question_id not found: {question_id}")
    if random_one:
        selected = random.Random(seed).randrange(len(records))
        return selected, records[selected]
    selected = sample_index if sample_index is not None else 0
    if selected < 0 or selected >= len(records):
        raise IndexError(f"sample-index {selected} is out of range.")
    return selected, records[selected]


def _record_id(record: Mapping[str, Any], index: int) -> str:
    return str(
        record.get("id")
        or record.get("question_id")
        or record.get("uid")
        or record.get("sample_id")
        or index
    )


def _parse_hard_constraints(raw_json: str | None) -> dict[str, Any]:
    if raw_json is None or not raw_json.strip():
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("--hard-constraints-json must decode to an object.")
    return parsed


def _default_hard_constraints(*, question_type: str, question: str) -> dict[str, Any]:
    lowered_question = question.lower()
    return {
        "question_type": question_type,
        "dosage_sensitive": bool(question_type == "dosage" or any(term in lowered_question for term in ("dose", "dosage", "mg", "mcg"))),
        "ddi_sensitive": bool(question_type == "drug_interaction" or any(term in lowered_question for term in ("interaction", "interact", "contraindication", "contraindicated", "ddi"))),
    }


def _mapping_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _flatten_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return " ".join(_flatten_to_text(item) for item in value.values() if _flatten_to_text(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_to_text(item) for item in value if _flatten_to_text(item))
    return str(value).strip()


def _extract_steps(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _flatten_to_text(value)
    if not text:
        return []
    return [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+|\n+", text) if fragment.strip()]


def _optional_text(value: Any) -> str | None:
    text = _flatten_to_text(value)
    return text if text else None


def _safe_sample_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())[:120] or "sample"


if __name__ == "__main__":
    raise SystemExit(main())
