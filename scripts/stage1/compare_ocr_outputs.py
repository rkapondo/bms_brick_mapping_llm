import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows[obj["image"]] = obj
    return rows


def load_gold_dir(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in sorted(path.rglob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        image = obj.get("image", p.stem + ".png")
        out[image] = obj
    return out


def canon_item(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
    item.get("base_id"),
    item.get("measurement"),
    item.get("pipe_color"),
    item.get("is_ui_label"),
    item.get("value"),
    item.get("unit"),
)


def score(pred_items: List[Dict[str, Any]], gold_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_counter = Counter(canon_item(x) for x in pred_items)
    gold_counter = Counter(canon_item(x) for x in gold_items)
    tp = sum((pred_counter & gold_counter).values())
    fp = sum((pred_counter - gold_counter).values())
    fn = sum((gold_counter - pred_counter).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = pred_counter == gold_counter
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
        "only_in_prediction": [list(x) for x in list((pred_counter - gold_counter).elements())[:50]],
        "only_in_gold": [list(x) for x in list((gold_counter - pred_counter).elements())[:50]],
    }


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["image"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR predictions against gold JSON files.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    preds = load_jsonl(Path(args.predictions))
    gold = load_gold_dir(Path(args.gold_dir))
    manifest = load_manifest(Path(args.manifest)) if args.manifest else load_manifest(Path(args.predictions).with_suffix(".manifest.json"))

    all_images = sorted(set(preds) | set(gold))
    per_image = {}
    total_tp = total_fp = total_fn = exact_count = 0
    vendor_buckets = defaultdict(lambda: {"images": 0, "tp": 0, "fp": 0, "fn": 0, "exact": 0})

    for image in all_images:
        pred_obj = preds.get(image, {"image": image, "items": []})
        gold_obj = gold.get(image, {"image": image, "items": []})
        result = score(pred_obj.get("items", []), gold_obj.get("items", []))
        vendor = manifest.get(image, {}).get("vendor", "unknown")
        result["vendor"] = vendor
        result["pred_count"] = len(pred_obj.get("items", []))
        result["gold_count"] = len(gold_obj.get("items", []))
        per_image[image] = result

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        exact_count += 1 if result["exact_match"] else 0

        vb = vendor_buckets[vendor]
        vb["images"] += 1
        vb["tp"] += result["tp"]
        vb["fp"] += result["fp"]
        vb["fn"] += result["fn"]
        vb["exact"] += 1 if result["exact_match"] else 0

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    vendor_summary = {}
    for vendor, vb in vendor_buckets.items():
        vp = vb["tp"] / (vb["tp"] + vb["fp"]) if vb["tp"] + vb["fp"] else 0.0
        vr = vb["tp"] / (vb["tp"] + vb["fn"]) if vb["tp"] + vb["fn"] else 0.0
        vf1 = 2 * vp * vr / (vp + vr) if vp + vr else 0.0
        vendor_summary[vendor] = {
            "images": vb["images"],
            "exact_match_rate": vb["exact"] / vb["images"] if vb["images"] else 0.0,
            "micro_precision": vp,
            "micro_recall": vr,
            "micro_f1": vf1,
        }

    output = {
        "summary": {
            "num_images": len(all_images),
            "exact_match_rate": exact_count / len(all_images) if all_images else 0.0,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        },
        "by_vendor": vendor_summary,
        "per_image": per_image,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
