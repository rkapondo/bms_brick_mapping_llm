#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {e}") from e
    return records


def infer_vendor_from_path(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def build_gold_index(gold_dir: Path) -> Dict[str, Dict[str, Any]]:
    gold_index: Dict[str, Dict[str, Any]] = {}
    for path in gold_dir.rglob("*.json"):
        record = load_json(path)
        image_name = record.get("image", path.with_suffix("").name)
        rel = path.relative_to(gold_dir)
        vendor = infer_vendor_from_path(str(rel))
        gold_index[image_name] = {
            "record": record,
            "vendor": vendor,
            "path": str(rel),
        }
    return gold_index


def build_prediction_index(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pred_index: Dict[str, Dict[str, Any]] = {}
    for record in predictions:
        image_name = record.get("image")
        if not image_name:
            continue
        pred_index[image_name] = {
            "record": record,
            "vendor": "unknown",
            "path": image_name,
        }
    return pred_index


def relaxed_item_key(item: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """
    Relaxed OCR matching key:
    - ignores id
    - ignores pipe_color
    - ignores is_ui_label
    - ignores measurement
    - scores primarily on base_id, value, unit
    """
    return (
        item.get("base_id"),
        item.get("value"),
        item.get("unit"),
    )


def safe_items(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = record.get("items", [])
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]
    return []


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OCR predictions against gold annotations using a relaxed metric: (base_id, value, unit)."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--gold-dir", required=True, help="Directory containing gold JSON files")
    parser.add_argument("--output", required=True, help="Path to save comparison JSON")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    gold_dir = Path(args.gold_dir)
    output_path = Path(args.output)

    predictions = load_jsonl(predictions_path)
    pred_index = build_prediction_index(predictions)
    gold_index = build_gold_index(gold_dir)

    all_images = sorted(set(pred_index.keys()) | set(gold_index.keys()))

    per_image: Dict[str, Any] = {}
    by_vendor_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"images": 0, "exact_matches": 0, "tp": 0, "fp": 0, "fn": 0}
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_exact = 0

    for image_name in all_images:
        pred_info = pred_index.get(image_name)
        gold_info = gold_index.get(image_name)

        pred_record = pred_info["record"] if pred_info else {"image": image_name, "items": []}
        gold_record = gold_info["record"] if gold_info else {"image": image_name, "items": []}

        vendor = "unknown"
        if gold_info:
            vendor = gold_info["vendor"]
        elif pred_info:
            vendor = pred_info["vendor"]

        pred_items = safe_items(pred_record)
        gold_items = safe_items(gold_record)

        pred_keys: Set[Tuple[Any, Any, Any]] = {relaxed_item_key(item) for item in pred_items}
        gold_keys: Set[Tuple[Any, Any, Any]] = {relaxed_item_key(item) for item in gold_items}

        tp = len(pred_keys & gold_keys)
        fp = len(pred_keys - gold_keys)
        fn = len(gold_keys - pred_keys)

        precision, recall, f1 = compute_prf(tp, fp, fn)
        exact_match = pred_keys == gold_keys

        only_in_prediction = sorted([list(x) for x in (pred_keys - gold_keys)], key=str)
        only_in_gold = sorted([list(x) for x in (gold_keys - pred_keys)], key=str)

        per_image[image_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "only_in_prediction": only_in_prediction,
            "only_in_gold": only_in_gold,
            "vendor": vendor,
            "pred_count": len(pred_keys),
            "gold_count": len(gold_keys),
        }

        by_vendor_counts[vendor]["images"] += 1
        by_vendor_counts[vendor]["tp"] += tp
        by_vendor_counts[vendor]["fp"] += fp
        by_vendor_counts[vendor]["fn"] += fn
        if exact_match:
            by_vendor_counts[vendor]["exact_matches"] += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn
        if exact_match:
            total_exact += 1

    summary_precision, summary_recall, summary_f1 = compute_prf(total_tp, total_fp, total_fn)

    by_vendor: Dict[str, Any] = {}
    for vendor, counts in sorted(by_vendor_counts.items()):
        v_precision, v_recall, v_f1 = compute_prf(counts["tp"], counts["fp"], counts["fn"])
        by_vendor[vendor] = {
            "images": counts["images"],
            "exact_match_rate": counts["exact_matches"] / counts["images"] if counts["images"] else 0.0,
            "micro_precision": v_precision,
            "micro_recall": v_recall,
            "micro_f1": v_f1,
        }

    result = {
        "metric": "relaxed_base_id_value_unit",
        "summary": {
            "num_images": len(all_images),
            "exact_match_rate": total_exact / len(all_images) if all_images else 0.0,
            "micro_precision": summary_precision,
            "micro_recall": summary_recall,
            "micro_f1": summary_f1,
        },
        "by_vendor": by_vendor,
        "per_image": per_image,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()