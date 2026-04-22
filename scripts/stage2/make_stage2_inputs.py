import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_dir(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(path.rglob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def infer_vendor(image_name: str, source_dir: Path) -> str:
    for p in source_dir.rglob("*"):
        if p.is_file() and p.name == image_name:
            rel = p.relative_to(source_dir)
            if len(rel.parts) >= 2:
                return rel.parts[0]
    return "unknown"


def normalize_stage2_input(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image": record["image"],
        "items": [
            {
                "id": item.get("id"),
                "base_id": item.get("base_id"),
                "pipe_color": item.get("pipe_color"),
                "is_ui_label": item.get("is_ui_label"),
                "value": item.get("value"),
                "unit": item.get("unit"),
            }
            for item in record.get("items", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-image Stage 2 inputs from Stage 1 outputs.")
    parser.add_argument("--source", required=True, help="Stage 1 JSONL file or directory of JSON files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default=None, help="Optional root directory to infer vendor subfolders")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root) if args.image_root else None
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        if source.suffix.lower() != ".jsonl":
            raise ValueError("If --source is a file, it must be a .jsonl file")
        records = load_jsonl(source)
    elif source.is_dir():
        records = load_json_dir(source)
    else:
        raise FileNotFoundError(source)

    for record in records:
        image_name = record["image"]
        vendor = infer_vendor(image_name, image_root) if image_root else "unknown"
        vendor_dir = output_dir / vendor
        vendor_dir.mkdir(parents=True, exist_ok=True)

        out_path = vendor_dir / f"{Path(image_name).stem}.json"
        out_obj = normalize_stage2_input(record)
        out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created {out_path}")


if __name__ == "__main__":
    main()