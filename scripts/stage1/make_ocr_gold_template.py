import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create empty gold JSON templates for OCR annotation.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--gold-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    gold_dir = Path(args.gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        rel = image_path.relative_to(input_dir)
        target = gold_dir / rel.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        obj = {"image": image_path.name, "items": []}
        target.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created {target}")


if __name__ == "__main__":
    main()
