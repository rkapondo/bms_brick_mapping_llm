import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_PROMPT = r'''You are Ministral (reasoning stage) in a two-stage pipeline.

You will receive OCR Input as JSON.

Task:
Map ONLY real points (items with is_ui_label=false) to the most appropriate Brick class.

Allowed output fields for each mapping:
- id
- base_id
- brick_class

Output requirements:
- Output STRICT JSON ONLY.
- Do NOT output markdown.
- Do NOT output code fences.
- Do NOT restate the input.
- Do NOT output retrieval text.
- Do NOT output reasoning.
- Do NOT output notes, headings, or explanations.
- Do NOT output any text before or after the JSON.
- Always return exactly one top-level JSON object with key "mappings".

Classification rules:
- Ignore all items where is_ui_label=true.
- For each remaining item, output exactly one mapping.
- brick_class must be a valid Brick class in CURIE form, e.g. brick:Water_Temperature_Sensor.
- Choose the most specific valid class supported by the retrieved Brick knowledge.
- If the point reads a physical quantity, prefer a Sensor class.
- If the point is a target value, prefer a Setpoint class.
- If the point sends a control signal, prefer a Command class.
- If the point reports operating state, prefer a Status class.
- If the selected class is a temperature sensor and pipe_color is purple, orange, red, or blue, prefer brick:Water_Temperature_Sensor when supported.
- If no exact match is available, fall back to one of:
  - brick:Sensor
  - brick:Setpoint
  - brick:Command
  - brick:Status
- If the role cannot be inferred, use brick:Sensor.
- Use the id exactly as provided.
- Use the base_id exactly as provided.

Output format:
{
  "mappings": [
    { "id": "1", "base_id": "VP01-GT401", "brick_class": "brick:Water_Temperature_Sensor" }
  ]
}
'''

JSON_REPAIR_CLASSES = [
    "brick:Sensor",
    "brick:Setpoint",
    "brick:Command",
    "brick:Status",
    "brick:Temperature_Sensor",
    "brick:Water_Temperature_Sensor",
    "brick:Flow_Sensor",
    "brick:Pressure_Sensor",
    "brick:Energy_Sensor",
    "brick:Power_Sensor",
    "brick:Percentage_Sensor",
    "brick:Volume_Sensor",
    "brick:Valve_Position_Sensor",
    "brick:Temperature_Setpoint",
    "brick:Flow_Setpoint",
    "brick:Pressure_Setpoint",
    "brick:Enable_Command",
    "brick:Open_Command",
    "brick:Start_Stop_Command",
    "brick:Valve_Status",
    "brick:Alarm_Status",
    "brick:Mode_Status",
]


def collect_inputs(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.json") if p.is_file()])


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    while text.endswith("```"):
        text = text[:-3].strip()
    return text


def repair_common_json_issues(text: str) -> str:
    classes_pat = "|".join(re.escape(x) for x in JSON_REPAIR_CLASSES)
    text = re.sub(
        rf'("brick_class"\s*:\s*)({classes_pat})(\s*[,}}])',
        r'\1"\2"\3',
        text,
    )
    return text


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = repair_common_json_issues(strip_code_fences(text))

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "mappings" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    candidates: List[str] = []
    stack: List[str] = []
    start_idx: Optional[int] = None

    for i, ch in enumerate(cleaned):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(cleaned[start_idx:i + 1])
                    start_idx = None

    for candidate in reversed(candidates):
        candidate = repair_common_json_issues(candidate)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "mappings" in obj:
                return obj
        except json.JSONDecodeError:
            continue

    return None


def build_full_prompt(prompt: str, input_json: Dict[str, Any]) -> str:
    return (
        f"{prompt}\n\n"
        f"BEGIN INPUT JSON\n\n"
        f"{json.dumps(input_json, ensure_ascii=False, indent=2)}\n\n"
        f"END INPUT JSON"
    )


def post_chat_completion(
    url: str,
    headers: Dict[str, str],
    model: str,
    temperature: float,
    max_tokens: int,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": 40,
        "repeat_penalty": 1,
        "messages": messages,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def extract_message_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})

    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")

    parts: List[str] = []

    if isinstance(reasoning_content, str) and reasoning_content.strip():
        parts.append(reasoning_content.strip())
    elif isinstance(reasoning_content, list):
        for block in reasoning_content:
            if isinstance(block, dict):
                text_part = block.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())

    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text_part = block.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
                elif block.get("type") == "output_text" and isinstance(block.get("text"), str) and block.get("text").strip():
                    parts.append(block["text"].strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())

    return "\n".join(parts).strip()


def call_lmstudio_with_continue(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    input_json: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    full_prompt = build_full_prompt(prompt, input_json)
    messages: List[Dict[str, str]] = [{"role": "user", "content": full_prompt}]

    first_response = post_chat_completion(
        url=url,
        headers=headers,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )

    first_text = extract_message_text(first_response)
    if first_text.strip():
        return first_response

    messages.append({"role": "assistant", "content": ""})
    messages.append({"role": "user", "content": "Continue. Output the final JSON now."})

    second_response = post_chat_completion(
        url=url,
        headers=headers,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return second_response


def normalize_output(parsed: Dict[str, Any], image_name: str) -> Dict[str, Any]:
    mappings = parsed.get("mappings", [])
    if not isinstance(mappings, list):
        mappings = []

    clean = []
    for item in mappings:
        if not isinstance(item, dict):
            continue

        item_id = item.get("id")
        base_id = item.get("base_id")
        brick_class = item.get("brick_class")

        if not isinstance(item_id, str):
            item_id = None
        if not isinstance(base_id, str):
            base_id = None
        if not isinstance(brick_class, str):
            brick_class = None

        if item_id is None or base_id is None or brick_class is None:
            continue

        clean.append(
            {
                "id": item_id,
                "base_id": base_id,
                "brick_class": brick_class,
            }
        )

    return {"image": image_name, "mappings": clean}


def infer_vendor(input_path: Path, input_root: Path) -> str:
    rel = input_path.relative_to(input_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Stage 2 Brick reasoning benchmark via LM Studio.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_MODEL", "mistralai/ministral-3-14b-reasoning"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    inputs = collect_inputs(input_dir)
    if args.limit is not None:
        inputs = inputs[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, input_path in enumerate(inputs, start=1):
            print(f"[{idx}/{len(inputs)}] Processing {input_path}")
            raw_text = ""
            parsed = None
            error = None

            try:
                input_json = json.loads(input_path.read_text(encoding="utf-8"))
                response = call_lmstudio_with_continue(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt,
                    input_json=input_json,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if raw_dir:
                    debug_file = raw_dir / f"{input_path.stem}.response.json"
                    debug_file.write_text(
                        json.dumps(response, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                raw_text = extract_message_text(response)
                parsed = parse_json_object(raw_text)
                if parsed is None:
                    error = "invalid_json"
            except Exception as exc:
                error = f"request_failed: {exc}"

            if raw_dir:
                raw_file = raw_dir / f"{input_path.stem}.txt"
                raw_file.write_text(raw_text, encoding="utf-8")

            image_name = input_path.with_suffix(".png").name
            if parsed is None:
                parsed = {"mappings": []}

            normalized = normalize_output(parsed, image_name)
            out_f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

            manifest_rows.append(
                {
                    "image": image_name,
                    "relative_path": str(input_path.relative_to(input_dir)),
                    "vendor": infer_vendor(input_path, input_dir),
                    "valid_json": error is None,
                    "error": error,
                    "num_mappings": len(normalized.get("mappings", [])),
                }
            )

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved predictions to {output_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()