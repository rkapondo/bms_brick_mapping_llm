import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_PROMPT = r'''You are Ministral in Stage 2 of a two-stage pipeline.

You receive OCR JSON for BMS points.

Task:
Map each real point (`is_ui_label=false`) to the most appropriate Brick class.

Use retrieved context only as optional support.
If retrieval is generic, weak, or only partly relevant, rely on the OCR JSON evidence.
Do not force a class just because retrieval mentions it.

Return STRICT JSON ONLY in exactly this form:
{
  "mappings": [
    { "id": "1", "base_id": "VP01-GT401", "brick_class": "brick:Water_Temperature_Sensor" }
  ]
}

Rules:
- Ignore items where `is_ui_label=true`
- Output exactly one mapping for each remaining item
- Use only valid Brick CURIE classes from this task:
  - `brick:Sensor`
  - `brick:Status`
  - `brick:Temperature_Sensor`
  - `brick:Water_Temperature_Sensor`
  - `brick:Pressure_Sensor`
  - `brick:Flow_Sensor`
  - `brick:Energy_Sensor`
  - `brick:Power_Sensor`
  - `brick:Volume_Sensor`
  - `brick:Valve_Position_Sensor`
- Prefer the most specific justified class
- Use OCR evidence such as `base_id`, `measurement`, `unit`, `value`, and `pipe_color`
- For repeated meter-like rows, classify each row independently
- Do not collapse repeated meter rows to a single generic class
- Use `brick:Water_Temperature_Sensor` only when water-loop evidence is present
- Use `brick:Temperature_Sensor` when temperature is evident but water-loop evidence is insufficient
- Use `brick:Valve_Position_Sensor` only when valve-position evidence is present
- Do not use `brick:Sensor` when the unit strongly identifies a specific quantity:
  - `kWh`, `MWh`, `Wh` -> `brick:Energy_Sensor`
  - `kW`, `W`, `MW` -> `brick:Power_Sensor`
  - `m3`, `m³` -> `brick:Volume_Sensor`
  - `m3/h`, `m³/h`, `l/tim`, `l/s` -> `brick:Flow_Sensor`
  - `bar` -> `brick:Pressure_Sensor`
  - `%` with valve evidence -> `brick:Valve_Position_Sensor`
- If a specific class is not justified, fall back to:
  - `brick:Sensor`
  - `brick:Status`

Return only the JSON object itself.
Do not write BEGIN/END markers.
Do not write explanations.
Do not write notes.
Do not write reasoning.
Do not restate the input.
'''

ALLOWED_OUTPUT_CLASSES = [
    "brick:Sensor",
    "brick:Status",
    "brick:Temperature_Sensor",
    "brick:Water_Temperature_Sensor",
    "brick:Pressure_Sensor",
    "brick:Flow_Sensor",
    "brick:Energy_Sensor",
    "brick:Power_Sensor",
    "brick:Volume_Sensor",
    "brick:Valve_Position_Sensor",
]

JSON_REPAIR_CLASSES = ALLOWED_OUTPUT_CLASSES.copy()

DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
DEFAULT_BRICK_COLLECTION = os.getenv("BRICK_COLLECTION_NAME", "brick")
DEFAULT_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


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


def remove_common_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*BEGIN OUTPUT JSON\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*END OUTPUT JSON\s*$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


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

    cleaned = repair_common_json_issues(remove_common_wrappers(strip_code_fences(text)))

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "mappings" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*"mappings"[\s\S]*\}', cleaned)
    if match:
        candidate = repair_common_json_issues(match.group(0))
        try:
            obj = json.loads(candidate)
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


def build_retriever(chroma_dir: str, collection_name: str, embed_model_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    embed_model = SentenceTransformer(embed_model_name)
    return collection, embed_model


def normalize_unit(unit: Any) -> str:
    if unit is None:
        return ""
    return str(unit).strip().lower()


def hydronic_cue_text(base_id: Any, measurement: Any, pipe_color: Any) -> str:
    text_parts = [
        str(base_id or ""),
        str(measurement or ""),
        str(pipe_color or ""),
    ]
    return " ".join(text_parts).lower()


def has_hydronic_evidence(base_id: Any, measurement: Any, pipe_color: Any) -> bool:
    text = hydronic_cue_text(base_id, measurement, pipe_color)
    hydronic_keywords = [
        "vv", "vvc", "vs", "vp", "fjv",
        "tillopp", "framledning", "retur", "returledning",
        "gt", "vmm", "em", "värme", "radiator",
    ]
    pipe_colors = {"purple", "orange", "red", "blue"}
    has_pipe = str(pipe_color).strip().lower() in pipe_colors if pipe_color is not None else False
    return has_pipe or any(k in text for k in hydronic_keywords)


def build_retrieval_queries(input_json: Dict[str, Any]) -> List[str]:
    queries: List[str] = []

    for item in input_json.get("items", []):
        if not isinstance(item, dict):
            continue
        if item.get("is_ui_label") is True:
            continue

        base_id = item.get("base_id")
        measurement = item.get("measurement")
        pipe_color = item.get("pipe_color")
        value = item.get("value")
        unit = item.get("unit")

        parts: List[str] = []
        if isinstance(base_id, str) and base_id.strip():
            parts.append(f"base_id {base_id}")
        if isinstance(measurement, str) and measurement.strip():
            parts.append(f"measurement {measurement}")
        if isinstance(unit, str) and unit.strip():
            parts.append(f"unit {unit}")
        if isinstance(pipe_color, str) and pipe_color.strip():
            parts.append(f"pipe_color {pipe_color}")
        if value is not None:
            parts.append(f"value {value}")

        base_upper = str(base_id).upper() if base_id is not None else ""
        measurement_lower = str(measurement).lower() if measurement is not None else ""
        unit_norm = normalize_unit(unit)

        semantic_hints: List[str] = []

        if unit_norm == "bar":
            semantic_hints.append("pressure point")
            semantic_hints.append("pressure sensor candidate")

        if unit_norm in {"%", "percent"}:
            semantic_hints.append("percentage point")
            semantic_hints.append("valve position candidate")

        if unit_norm in {"kwh", "mwh", "wh"}:
            semantic_hints.append("energy meter row")
            semantic_hints.append("energy sensor candidate")

        if unit_norm in {"kw", "w", "mw"}:
            semantic_hints.append("power meter row")
            semantic_hints.append("power sensor candidate")

        if unit_norm in {"m3", "m³"}:
            semantic_hints.append("volume meter row")
            semantic_hints.append("volume sensor candidate")

        if unit_norm in {"m3/h", "m³/h", "l/tim", "l/s"}:
            semantic_hints.append("flow meter row")
            semantic_hints.append("flow sensor candidate")

        if unit_norm == "°c":
            if has_hydronic_evidence(base_id, measurement, pipe_color):
                semantic_hints.append("water loop temperature candidate")
            else:
                semantic_hints.append("generic temperature candidate")

        if value is None and unit is None:
            semantic_hints.append("generic equipment-or-point object")
            semantic_hints.append("prefer sensor fallback unless specific point evidence exists")

        if "GT" in base_upper:
            semantic_hints.append("temperature-like point")
        if "GP" in base_upper:
            semantic_hints.append("pressure-like point")
        if "SV" in base_upper or "STV" in base_upper:
            semantic_hints.append("valve-related point")
        if "EM" in base_upper or "VMM" in base_upper:
            semantic_hints.append("meter-like point")
        if "VVX" in base_upper:
            semantic_hints.append("equipment-like object point")
        if "P1" in base_upper or "PU" in base_upper:
            semantic_hints.append("pump-like object point")

        if "energi" in measurement_lower:
            semantic_hints.append("energy row")
        if "effekt" in measurement_lower:
            semantic_hints.append("power row")
        if "flöde" in measurement_lower or "volymflöde" in measurement_lower:
            semantic_hints.append("flow row")
        if "volym" in measurement_lower:
            semantic_hints.append("volume row")
        if any(x in measurement_lower for x in ["tillopp", "framledning", "retur", "returledning"]):
            semantic_hints.append("water temperature row")
        if any(x in measurement_lower for x in ["rumstemp", "utetemp", "dämpad"]):
            semantic_hints.append("generic temperature row")

        if semantic_hints:
            parts.append("; ".join(dict.fromkeys(semantic_hints)))

        parts.append("Brick class for BMS point")
        query = ", ".join(parts).strip()
        if query:
            queries.append(query)

    seen = set()
    deduped: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def classify_retrieved_doc(doc: str) -> str:
    lower = doc.lower()

    if "valve_position_sensor" in lower or "valve position" in lower:
        return "valve"
    if "pressure_sensor" in lower or "pressure" in lower or "bar" in lower:
        return "pressure"
    if "flow_sensor" in lower or "flow rate" in lower or "m3/h" in lower or "m³/h" in lower or "l/tim" in lower:
        return "flow"
    if "energy_sensor" in lower or "energy consumption" in lower or "kwh" in lower or "mwh" in lower:
        return "energy"
    if "power_sensor" in lower or "instantaneous power" in lower or "kw" in lower:
        return "power"
    if "volume_sensor" in lower or "accumulated volume" in lower or "m3" in lower or "m³" in lower:
        return "volume"
    if "water_temperature_sensor" in lower or "water loop temperature" in lower:
        return "water_temp"
    if "temperature_sensor" in lower or "generic temperature" in lower:
        return "temp"
    if "meter" in lower or "repeated rows" in lower:
        return "meter"
    return "other"


def retrieve_brick_context(
    collection,
    embed_model,
    input_json: Dict[str, Any],
    top_k: int,
    max_chars: int,
) -> str:
    queries = build_retrieval_queries(input_json)
    if not queries:
        return ""

    scored_chunks: List[Dict[str, Any]] = []

    for q in queries:
        embedding = embed_model.encode([q], normalize_embeddings=True).tolist()[0]
        res = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            source = meta.get("source") if isinstance(meta, dict) else None
            scored_chunks.append(
                {
                    "query": q,
                    "document": str(doc).strip(),
                    "source": source,
                    "distance": dist,
                    "kind": classify_retrieved_doc(str(doc)),
                }
            )

    scored_chunks.sort(key=lambda x: x["distance"] if x["distance"] is not None else 999999.0)

    selected: List[Dict[str, Any]] = []
    selected_docs = set()
    selected_kinds = set()
    total_chars = 0

    preferred_order = [
        "meter",
        "valve",
        "pressure",
        "flow",
        "energy",
        "power",
        "volume",
        "water_temp",
        "temp",
        "other",
    ]

    for preferred_kind in preferred_order:
        for chunk in scored_chunks:
            doc = chunk["document"]
            kind = chunk["kind"]

            if not doc or kind != preferred_kind:
                continue
            if doc in selected_docs:
                continue
            if kind in selected_kinds and kind != "other":
                continue
            if total_chars + len(doc) > max_chars:
                continue

            selected.append(chunk)
            selected_docs.add(doc)
            selected_kinds.add(kind)
            total_chars += len(doc)

    if not selected:
        for chunk in scored_chunks:
            doc = chunk["document"]
            if not doc or doc in selected_docs:
                continue
            if total_chars + len(doc) > max_chars:
                continue
            selected.append(chunk)
            selected_docs.add(doc)
            total_chars += len(doc)
            if len(selected) >= top_k:
                break

    if not selected:
        return ""

    lines = ["BEGIN RAG RETRIEVAL", ""]
    for i, chunk in enumerate(selected, start=1):
        lines.append(f"{i}. Source: {chunk['source'] or 'unknown'}")
        lines.append(chunk["document"])
        lines.append("")
    lines.append("END RAG RETRIEVAL")
    return "\n".join(lines).strip()


def build_full_prompt(prompt: str, input_json: Dict[str, Any], retrieval_text: str) -> str:
    ocr_json = json.dumps(input_json, ensure_ascii=False, indent=2)

    if retrieval_text:
        return (
            f"{prompt}\n\n"
            f"BEGIN INPUT JSON\n\n"
            f"{ocr_json}\n\n"
            f"END INPUT JSON\n\n"
            f"{retrieval_text}"
        )

    return (
        f"{prompt}\n\n"
        f"BEGIN INPUT JSON\n\n"
        f"{ocr_json}\n\n"
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
                elif (
                    block.get("type") == "output_text"
                    and isinstance(block.get("text"), str)
                    and block.get("text").strip()
                ):
                    parts.append(block["text"].strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())

    return "\n".join(parts).strip()


def call_lmstudio_with_continue(
    base_url: str,
    api_key: str,
    model: str,
    full_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

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
    parsed = parse_json_object(first_text)
    if parsed is not None:
        return first_response

    messages.append({"role": "assistant", "content": first_text if first_text else ""})
    messages.append(
        {
            "role": "user",
            "content": (
                'Return only the final JSON object now. '
                'No headings. No BEGIN/END markers. No explanation. '
                'Use exactly {"mappings":[...]} format.'
            ),
        }
    )

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
    seen = set()

    for item in mappings:
        if not isinstance(item, dict):
            continue

        item_id = item.get("id")
        base_id = item.get("base_id")
        brick_class = item.get("brick_class")

        if not isinstance(item_id, str):
            continue
        if not isinstance(base_id, str):
            continue
        if not isinstance(brick_class, str):
            continue
        if brick_class not in ALLOWED_OUTPUT_CLASSES:
            continue

        key = (item_id, base_id, brick_class)
        if key in seen:
            continue
        seen.add(key)

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
    parser = argparse.ArgumentParser(description="Batch Stage 2 Brick reasoning benchmark via LM Studio with optional Brick RAG.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--api-key", default=os.getenv("LMSTUDIO_API_KEY", "lm-studio"))
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_MODEL", "mistralai/ministral-3-14b-reasoning"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--raw-dir", default=None)

    parser.add_argument("--use-rag", action="store_true", help="Enable Brick retrieval from Chroma and append it to the prompt.")
    parser.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--brick-collection", default=DEFAULT_BRICK_COLLECTION)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-retrieved-chars", type=int, default=900)
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

    collection = None
    embed_model = None
    if args.use_rag:
        print(f"Initializing Brick RAG from {args.chroma_dir} / collection={args.brick_collection}")
        collection, embed_model = build_retriever(
            chroma_dir=args.chroma_dir,
            collection_name=args.brick_collection,
            embed_model_name=args.embed_model,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, input_path in enumerate(inputs, start=1):
            print(f"[{idx}/{len(inputs)}] Processing {input_path}")
            raw_text = ""
            parsed = None
            error = None
            retrieval_text = ""

            try:
                input_json = json.loads(input_path.read_text(encoding="utf-8"))

                if args.use_rag:
                    retrieval_text = retrieve_brick_context(
                        collection=collection,
                        embed_model=embed_model,
                        input_json=input_json,
                        top_k=args.top_k,
                        max_chars=args.max_retrieved_chars,
                    )

                full_prompt = build_full_prompt(
                    prompt=prompt,
                    input_json=input_json,
                    retrieval_text=retrieval_text,
                )

                response = call_lmstudio_with_continue(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    full_prompt=full_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                if raw_dir:
                    debug_file = raw_dir / f"{input_path.stem}.response.json"
                    debug_file.write_text(
                        json.dumps(response, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    prompt_file = raw_dir / f"{input_path.stem}.prompt.txt"
                    prompt_file.write_text(full_prompt, encoding="utf-8")

                    if args.use_rag:
                        rag_file = raw_dir / f"{input_path.stem}.retrieval.txt"
                        rag_file.write_text(retrieval_text, encoding="utf-8")

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