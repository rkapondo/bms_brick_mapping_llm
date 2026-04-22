import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PROMPT = r'''You are an expert Building Management System (BMS) OCR extraction system.

Your task is to read the BMS screenshot image I provide and extract every visible printed item without missing any tags, labels, devices, or readings.

Return output in this format.

JSONL schema

For each image, write one JSON object in this exact structure:

{
  "image": "filename.png",
  "items": [
    {
      "id": "1",
      "base_id": "...",
      "measurement": "...",
      "pipe_color": "purple|orange|red|green|blue|yellow|brown|turquoise|null",
      "is_ui_label": true,
      "value": "...",
      "unit": "°C|bar|%|kWh|MWh|Wh|m3|m³|m3/h|m³/h|kW|l/tim|null"
    }
  ]
}

Core objective
	•	Extract all visible printed text items relevant to the BMS diagram.
	•	Do not skip any reading just because it shares the same tag as another reading.
	•	Do not collapse multiple visible readings into one item.
	•	Do not omit valves, sensors, equipment tags, section headers, descriptive labels, or meter rows.
	•	The output must be exhaustive.

Exhaustiveness rule

Before finalizing, check the image systematically and ensure all of the following visible item types have been considered:
	•	Section headers
	•	Device tags
	•	Sensor tags
	•	Valve/controller tags
	•	Equipment tags
	•	Meter/device summary headers
	•	Every visible measurement row under a summary block
	•	Every visible numeric reading beside a point
	•	Every descriptive label paired with a point
	•	UI labels and descriptive text without anchors

If a printed label or reading is visible, it must appear in the output as an item unless it is clearly part of another item under the rules below.

Preserve text exactly
	•	Preserve case, spaces, punctuation, hyphens, and dots exactly.
	•	Preserve O vs 0 exactly as printed.
	•	Preserve VV vs W exactly as printed.
	•	Preserve uppercase/lowercase exactly, for example Vvc must remain Vvc.
	•	Do not invent text.
	•	Do not normalize text.
	•	Do not rewrite prefixes.
	•	Do not merge duplicates.
	•	If the same printed base_id appears multiple times, keep multiple entries as needed.
	•	If the same printed base_id has multiple visible values, create multiple separate entries.

Multi-line identifier rule
	•	A single printed identifier may be split across multiple lines due to layout.
	•	If stacked text fragments clearly belong to one identifier, combine them into one base_id in normal reading order.
	•	Do not insert spaces unless a space is visibly printed.
	•	Preserve printed punctuation exactly when combining fragments.
	•	Only combine fragments when they are visually one label/tag split by a line break.
	•	Do not combine nearby text just because it is vertically close.

Examples:
	•	VP01
-GT401
→ VP01-GT401
	•	VP01-
GT401
→ VP01-GT401
	•	VP01
GT401
→ combine only if clearly one identifier, then VP01GT401

Reading completeness rule
	•	Every visible reading must be extracted.
	•	If one printed point tag has two or more visible values, create one separate item for each visible value.
	•	Do not keep only one value when multiple values are printed.
	•	This rule applies even when the repeated items share the same base_id, measurement, and unit.

Examples:
	•	If VS01-GT101 has two printed values, output two separate items with base_id = "VS01-GT101".
	•	If VV01-GT101 has two printed values, output two separate items with base_id = "VV01-GT101".
	•	If VS01-GT301 appears once but has two labeled rows Utetemp. and Dämpad., output two separate items.

Meter summary block rule
	•	If a device or meter ID appears as a header above or beside a list of named readings, treat that device or meter ID as the shared base_id for all readings in that block.
	•	Each reading row becomes a separate item.
	•	Store the row label in the measurement field exactly as printed.
	•	Do not concatenate the row label into base_id.
	•	Do not merge all readings into one item.
	•	Do not omit any row in the block.

Example:
	•	If the image shows VP01-VMM801 with:
	•	Energi: 953.683 MWh
	•	Volym: 19922.0 m³
	•	Effekt: 38.0 kW
	•	Flöde: 840 l/tim
	•	Tilloppstemp.: 74.4 °C
	•	Returtemp.: 35.2 °C
	•	Diff.temp.: 39.3 °C
then output seven separate items, all with:
	•	base_id = "VP01-VMM801"

Single-point descriptive label rule
	•	If a point identifier is shown together with one descriptive label and one value/unit, treat it as one real physical point.
	•	Use the point tag as base_id.
	•	Use the descriptive label as measurement.
	•	Do not treat this as a meter summary block unless there are multiple separate reading rows under the same parent device or meter.
	•	If a sensor or device anchor is visible, set is_ui_label=false.

Example:
	•	VS01-GT302
	•	Rumstemp.
	•	21.3 °C

becomes one item with:
	•	base_id = "VS01-GT302"
	•	measurement = "Rumstemp."
	•	value = "21.3"
	•	unit = "°C"

Multi-row descriptive reading rule
	•	If one point tag has multiple descriptive reading rows, output one item per row.
	•	Use the same base_id for all rows.
	•	Use each row label as its own measurement.
	•	Do not collapse them into one item.

Example:
	•	VS01-GT301
	•	Utetemp.: -5.2 °C
	•	Dämpad: -2.1 °C

becomes two items:
	•	base_id = "VS01-GT301", measurement = "Utetemp.", value = "-5.2", unit = "°C"
	•	base_id = "VS01-GT301", measurement = "Dämpad", value = "-2.1", unit = "°C"

Field rules

id
	•	Set id as a string starting at "1" and increment by 1 for each extracted item in reading order.

base_id
	•	Use the printed identifier exactly as shown.
	•	If the identifier is split across lines, reconstruct it using the multi-line identifier rule.
	•	For meter summary blocks, use the parent device or meter ID as the base_id for every row in that block.
	•	For single-point and multi-row point readings, use the point tag as base_id.
	•	For UI labels that are not attached to a physical point, use the visible printed text exactly as base_id.

measurement
	•	Use measurement for the descriptive reading label associated with a value, exactly as printed.
	•	Examples:
	•	Energi
	•	Volym
	•	Effekt
	•	Flöde
	•	Tilloppstemp.
	•	Returtemp.
	•	Diff.temp.
	•	Rumstemp.
	•	Utetemp.
	•	Dämpad
	•	For point readings with no descriptive label, set measurement to null.
	•	For UI labels, set measurement to null.

value
	•	Extract the visible value if present.
	•	Keep minus signs if printed.
	•	Keep decimals exactly as printed.
	•	Otherwise set value to null.

unit
	•	Extract the visible unit if present.
	•	Otherwise set unit to null.
	•	Allowed units only:
	•	°C
	•	bar
	•	%
	•	kWh
	•	MWh
	•	Wh
	•	m3
	•	m³
	•	m3/h
	•	m³/h
	•	kW
	•	l/tim

is_ui_label

Set is_ui_label=true only for:
	•	Text inside grey rounded rectangles
	•	Section headers such as VP01, VS01, VV01 that are not attached to an instrument or valve anchor
	•	Descriptive labels such as Rad., Kv. utg., Kv. ink., Vv., Vvc that are not attached to an anchor

Set is_ui_label=false for real physical points, including when value=null and unit=null:
	•	Sensors (GT, GP, PT, etc.) with a probe/stem anchor
	•	Valves/controllers (STV, PV, PCV, etc.) with a valve symbol inserted into a pipe
	•	Equipment tags such as EXP201
	•	Meter summary block rows that belong to a real physical device or meter
	•	Single-point descriptive label cases
	•	Multi-row descriptive reading cases

Critical anchor rule
	•	value=null and unit=null does not mean UI label.
	•	Use anchor presence to decide:
	•	If there is a device, valve, sensor, or equipment anchor, then is_ui_label=false
	•	If there is no anchor and the text is a header, pill, or descriptive UI text, then is_ui_label=true
	•	For meter summary blocks, rows inherit the physical status of the parent meter/device and should be is_ui_label=false.

pipe_color rules

When is_ui_label=true
	•	pipe_color must be null

When is_ui_label=false
	•	Determine pipe_color only from the thick colored pipe segment that the anchor physically touches.

Allowed values:
	•	purple
	•	orange
	•	red
	•	blue
	•	yellow
	•	brown
	•	turquoise
	•	green
	•	null

Important color constraints
	•	Use anchor contact, not text proximity.
	•	Valid anchors include:
	•	sensor probe plus stem
	•	valve/controller symbol inserted in pipe
	•	equipment symbol connection point
	•	Do not assign pipe color from nearby text alone.
	•	Do not use a heat exchanger coil or zigzag unless the anchor touches that exact colored segment.
	•	For meter summary blocks, use the pipe color of the parent device/meter anchor if visible; otherwise null.
	•	For physical valves and controllers, use the color of the pipe segment they are inserted into.
	•	For equipment connecting between two colors, use the color of the segment the visible anchor/connection belongs to. If not unambiguous, use null.

Special rule for valves/controllers and equipment

Do not miss items that have:
	•	a tag
	•	a physical valve/controller/equipment symbol
	•	but no visible numeric value

These are still valid extracted items and must be included with:
	•	value = null
	•	unit = null

Examples:
	•	VS01-PV601
	•	VS01-STV201
	•	VV01-PCV201
	•	VV01-STV201
	•	VS01-EXP201

Reading order
	•	Assign IDs in reading order from top to bottom, left to right.
	•	For grouped blocks, process the header first, then each row in visible order.
	•	For repeated values under one tag, keep the same top-to-bottom order as printed.

Final completeness checklist

Before producing the JSONL, verify all of the following:
	1.	Every visible section header is included if it is printed separately.
	2.	Every visible sensor/device/valve/equipment tag is included.
	3.	Every visible numeric reading is included.
	4.	Every visible descriptive row label tied to a reading is included in measurement.
	5.	Every summary-block row is included.
	6.	Every repeated value for the same base_id is included as a separate item.
	7.	Every anchored item without a value is still included.
	8.	Every UI label such as Vv., Vvc, Kv. utg., Kv. ink. is included if visible.
	9.	All text casing matches exactly as printed.
	10.	UI labels have pipe_color = null.

Summary of decision logic
	1.	Extract every visible label, tag, and reading exactly as printed.
	2.	Join split identifiers only when they are clearly one printed identifier.
	3.	If a meter/device header has multiple reading rows, create one item per row with shared base_id.
	4.	If one point has multiple visible values or multiple labeled rows, create one item per visible reading.
	5.	If a physical tag is anchored but has no value, still create an item.
	6.	Decide UI label vs real point using anchor presence.
	7.	Use only anchor-touching segments for pipe_color.
	8.	Never normalize text.
	9.	Never merge repeated printed items.
	10.	Never omit a visible reading.

Output requirements
	•	One JSON object per image.
	•	No commentary in the file.
	•	No extra explanations in the file.
	•	In chat, paste the JSONL only.'''

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_images(input_dir: Path) -> List[Path]:
    images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def repair_common_json_issues(text: str) -> str:
    text = re.sub(
        r'("pipe_color"\s*:\s*)(red|blue|purple|orange|yellow|brown|turquoise|green)(\s*[,}])',
        r'\1"\2"\3',
        text,
    )
    return text


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = strip_code_fences(text)
    cleaned = repair_common_json_issues(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        candidate = repair_common_json_issues(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def call_lmstudio(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
    request_timeout: int = 300,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=(30, 900),
    )
    resp.raise_for_status()
    return resp.json()


def infer_vendor(image_path: Path, input_dir: Path) -> str:
    rel = image_path.relative_to(input_dir)
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch OCR benchmark for Gemma via LM Studio.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_MODEL", "google/gemma-3-12b"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-dir", default=None, help="Optional directory to save raw model text per image")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    images = collect_images(input_dir)
    if args.limit is not None:
        images = images[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, image_path in enumerate(images, start=1):
            print(f"[{idx}/{len(images)}] Processing {image_path}")
            raw_text = ""
            parsed = None
            error = None

            try:
                response = call_lmstudio(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    image_path=image_path,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw_text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(raw_text)
                if parsed is None:
                    error = "invalid_json"
            except Exception as exc:  # noqa: BLE001
                error = f"request_failed: {exc}"

            if raw_dir:
                raw_file = raw_dir / f"{image_path.stem}.txt"
                raw_file.write_text(raw_text, encoding="utf-8")

            if parsed is None:
                parsed = {"image": image_path.name, "items": []}

            parsed["image"] = image_path.name

            out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            manifest_rows.append(
                {
                    "image": image_path.name,
                    "relative_path": str(image_path.relative_to(input_dir)),
                    "vendor": infer_vendor(image_path, input_dir),
                    "valid_json": error is None,
                    "error": error,
                    "num_items": len(parsed.get("items", [])) if isinstance(parsed.get("items", []), list) else None,
                }
            )

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved predictions to {output_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()