import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_predictions_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[obj["image"]] = obj
    return rows


def load_gold_dir(gold_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for p in sorted(gold_dir.rglob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows[obj["image"]] = obj
    return rows


def mapping_key(item: Dict[str, Any]) -> Tuple[Any, Any]:
    return (item.get("base_id"), item.get("brick_class"))


def infer_vendor_from_gold_path(image_name: str, gold_dir: Path) -> str:
    for p in gold_dir.rglob("*.json"):
        obj = json.loads(p.read_text(encoding="utf-8"))
        if obj.get("image") == image_name:
            rel = p.relative_to(gold_dir)
            if len(rel.parts) >= 2:
                return rel.parts[0]
    return "unknown"


def compute_scores(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage 2 Brick mappings against gold.")
    parser.add_argument("--predictions", required=True, help="JSONL predictions from benchmark_stage2_reasoning.py")
    parser.add_argument("--gold-dir", required=True, help="Directory of gold Stage 2 JSON files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pred_rows = load_predictions_jsonl(Path(args.predictions))
    gold_rows = load_gold_dir(Path(args.gold_dir))

    per_image: Dict[str, Any] = {}
    by_vendor_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "images": 0, "exact": 0})
    total_tp = total_fp = total_fn = 0
    exact_count = 0

    for image_name, gold_obj in gold_rows.items():
        pred_obj = pred_rows.get(image_name, {"image": image_name, "mappings": []})

        pred_items = pred_obj.get("mappings", [])
        gold_items = gold_obj.get("mappings", [])

        pred_set: Set[Tuple[Any, Any]] = {mapping_key(x) for x in pred_items if isinstance(x, dict)}
        gold_set: Set[Tuple[Any, Any]] = {mapping_key(x) for x in gold_items if isinstance(x, dict)}

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        exact = pred_set == gold_set

        scores = compute_scores(tp, fp, fn)
        vendor = infer_vendor_from_gold_path(image_name, Path(args.gold_dir))

        per_image[image_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **scores,
            "exact_match": exact,
            "only_in_prediction": sorted(list(pred_set - gold_set)),
            "only_in_gold": sorted(list(gold_set - pred_set)),
            "vendor": vendor,
            "pred_count": len(pred_set),
            "gold_count": len(gold_set),
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        exact_count += int(exact)

        by_vendor_counts[vendor]["tp"] += tp
        by_vendor_counts[vendor]["fp"] += fp
        by_vendor_counts[vendor]["fn"] += fn
        by_vendor_counts[vendor]["images"] += 1
        by_vendor_counts[vendor]["exact"] += int(exact)

    summary_scores = compute_scores(total_tp, total_fp, total_fn)
    by_vendor = {}
    for vendor, counts in by_vendor_counts.items():
        vendor_scores = compute_scores(counts["tp"], counts["fp"], counts["fn"])
        by_vendor[vendor] = {
            "images": counts["images"],
            "exact_match_rate": counts["exact"] / counts["images"] if counts["images"] else 0.0,
            **vendor_scores,
        }

    result = {
        "summary": {
            "num_images": len(gold_rows),
            "exact_match_rate": exact_count / len(gold_rows) if gold_rows else 0.0,
            **summary_scores,
        },
        "by_vendor": by_vendor,
        "per_image": per_image,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()