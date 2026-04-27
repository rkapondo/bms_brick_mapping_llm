import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ALLOWED_CLASSES = {
    "brick:Sensor",
    "brick:Status",
    "brick:Temperature_Sensor",
    "brick:Water_Temperature_Sensor",
    "brick:Outdoor_Temperature_Sensor",
    "brick:Flow_Sensor",
    "brick:Pressure_Sensor",
    "brick:Energy_Sensor",
    "brick:Power_Sensor",
    "brick:Volume_Sensor",
    "brick:Valve_Position_Sensor",
}

CLASS_GUIDANCE: Dict[str, str] = {
    "brick:Sensor": (
        "Class guidance for brick:Sensor. Use as the generic fallback for a real point when the point is clearly "
        "a measured or reported point, but the OCR evidence is not strong enough to justify a more specific class. "
        "Do not use brick:Sensor when a more specific valid sensor class is clearly supported by unit, Swedish cue, "
        "or point identifier."
    ),
    "brick:Status": (
        "Class guidance for brick:Status. Use for points that report operating state or condition rather than a "
        "measured physical quantity. Typical evidence includes explicit state-like language such as status, drift, "
        "on/off, enabled, disabled, open/closed, active/inactive, or similar state reporting. Do not use Status for "
        "a physical reading such as temperature, pressure, flow, energy, power, volume, or valve opening percentage."
    ),
    "brick:Temperature_Sensor": (
        "Class guidance for brick:Temperature_Sensor. Use when a point measures temperature but there is not enough "
        "evidence that the medium is water. Typical evidence includes generic temperature cues, room temperature, "
        "outdoor temperature, damped outdoor temperature, or temperature-like values without a clear hydronic or "
        "water-loop context."
    ),
    "brick:Water_Temperature_Sensor": (
        "Class guidance for brick:Water_Temperature_Sensor. Use when a point measures the temperature of water in a "
        "hydronic, district-heating, domestic hot water, radiator, or circulation loop. Strong evidence includes °C "
        "together with loop cues such as tillopp, framledning, retur, returledning, VV, VVC, VS, VP, GT, colored "
        "pipe runs, or a meter row that clearly refers to water supply or return temperatures. Do not assign this "
        "class from °C alone."
    ),
    "brick:Flow_Sensor": (
        "Class guidance for brick:Flow_Sensor. Use for a point that measures flow rate rather than accumulated "
        "volume. Strong evidence includes m3/h, m³/h, l/tim, l/s, flöde, or volymflöde. Do not confuse flow with "
        "accumulated volume."
    ),
    "brick:Pressure_Sensor": (
        "Class guidance for brick:Pressure_Sensor. Use for a point that measures pressure. Strong evidence includes "
        "unit bar or pressure-related cues such as tryck or GP."
    ),
    "brick:Energy_Sensor": (
        "Class guidance for brick:Energy_Sensor. Use for a point that measures accumulated energy consumption or "
        "production. Strong evidence includes kWh, MWh, Wh, or the Swedish cue energi. In meter-style groups, the "
        "row labeled Energi maps to Energy_Sensor."
    ),
    "brick:Power_Sensor": (
        "Class guidance for brick:Power_Sensor. Use for a point that measures instantaneous power. Strong evidence "
        "includes kW, W, MW, or the Swedish cue effekt. In meter-style groups, the row labeled Effekt maps to "
        "Power_Sensor rather than Energy_Sensor."
    ),
    "brick:Volume_Sensor": (
        "Class guidance for brick:Volume_Sensor. Use for a point that measures accumulated volume, usually with unit "
        "m3 or m³. The Swedish cue volym supports this class. Do not confuse accumulated volume with flow rate."
    ),
    "brick:Valve_Position_Sensor": (
        "Class guidance for brick:Valve_Position_Sensor. Use for a sensor that reports valve opening or valve "
        "position, often as a percentage. Strong evidence includes a valve-related tag such as SV, STV, valve, "
        "ventil, läge, or position together with unit percent. Percent alone is not enough."
    ),
}

GENERAL_RULES: List[Tuple[str, str]] = [
    (
        "general_rule_units",
        "General unit rule. Use units as supporting evidence rather than as the only signal. kWh and MWh support "
        "Energy_Sensor. kW supports Power_Sensor. m3 and m³ support Volume_Sensor. m3/h, m³/h, and l/tim support "
        "Flow_Sensor. bar supports Pressure_Sensor. °C supports a temperature sensor, but OCR context is still "
        "needed to distinguish Water_Temperature_Sensor from Temperature_Sensor."
    ),
    (
        "general_rule_meter_rows",
        "General meter-row rule. Repeated rows under the same meter identifier can map to different Brick classes. "
        "Classify each row by its own measurement label and unit. Typical mappings are Energi to Energy_Sensor, "
        "Effekt to Power_Sensor, Flöde or Volymflöde to Flow_Sensor, Volym to Volume_Sensor, and supply or return "
        "temperatures to the appropriate temperature class."
    ),
    (
        "general_rule_swedish_cues",
        "General Swedish cue rule. temperatur supports a temperature sensor. tillopp and framledning suggest supply "
        "water context. retur and returledning suggest return water context. flöde and volymflöde support "
        "Flow_Sensor. energi supports Energy_Sensor. effekt supports Power_Sensor. volym supports Volume_Sensor. "
        "ventil, läge, and position support Valve_Position_Sensor when paired with percent or valve-like identifiers."
    ),
    (
        "general_rule_avoid_overspecialization",
        "General fallback rule. Prefer the most specific allowed class only when the OCR evidence supports it. If the "
        "point is clearly a real point but there is not enough support for a specific class, use brick:Sensor. Do not "
        "invent classes outside the allowed Stage 2 benchmark output space."
    ),
]

KEEP_DEFINITION_PATTERNS = [
    r"temperature sensor",
    r"water temperature sensor",
    r"flow sensor",
    r"pressure sensor",
    r"energy sensor",
    r"power sensor",
    r"volume sensor",
    r"valve position sensor",
    r"\bsensor\b",
    r"\bstatus\b",
]

SKIP_DEFINITION_PATTERNS = [
    r"setpoint",
    r"command",
    r"alarm",
    r"heat exchanger",
    r"expansion tank",
    r"pump",
    r"equipment",
    r"mode status",
    r"alarm status",
    r"valve status",
]

FAMILY_FILES = [
    "brick_sensor.txt",
    "brick_status.txt",
    "brick_temperature.txt",
    "brick_flow.txt",
]

MODULE_FILES = [
    "sensor.py",
    "status.py",
    "meters.py",
    "quantities.py",
]

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def safe_id(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "doc"

def should_keep_definition(label: str, definition: str) -> bool:
    t = f"{label} {definition}".lower()
    if any(re.search(p, t) for p in SKIP_DEFINITION_PATTERNS):
        return False
    return any(re.search(p, t) for p in KEEP_DEFINITION_PATTERNS)

def extract_relevant_definitions(definitions_csv: Path) -> List[Dict]:
    docs: List[Dict] = []
    if not definitions_csv.exists():
        return docs

    with definitions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            uri = normalize_space(row[0])
            definition = normalize_space(row[1])
            if not uri:
                continue

            label = uri.split("#")[-1] if "#" in uri else uri.rsplit("/", 1)[-1]
            label_spaced = label.replace("_", " ")
            if not should_keep_definition(label_spaced, definition):
                continue

            doc_text = f"Local definition for {label}. {definition}".strip()
            docs.append(
                {
                    "id": f"local_definition_{safe_id(label)}",
                    "text": doc_text,
                    "metadata": {
                        "source": str(definitions_csv),
                        "doc_type": "local_definition",
                        "label": label,
                    },
                }
            )
    return docs

def load_family_guidance(data_brick_dir: Path) -> List[Dict]:
    docs: List[Dict] = []
    for name in FAMILY_FILES:
        path = data_brick_dir / name
        if not path.exists():
            continue
        text = normalize_space(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue

        short = text[:1200]
        docs.append(
            {
                "id": f"family_guidance_{safe_id(name)}",
                "text": f"Family guidance from {name}. {short}",
                "metadata": {
                    "source": str(path),
                    "doc_type": "family_guidance",
                    "label": name,
                },
            }
        )
    return docs

def summarize_python_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    class_lines = [ln.strip() for ln in lines if re.match(r"^\s*class\s+[A-Za-z0-9_]+", ln)]
    func_lines = [ln.strip() for ln in lines if re.match(r"^\s*def\s+[A-Za-z0-9_]+", ln)]

    pieces: List[str] = []
    if class_lines:
        pieces.append("Classes: " + "; ".join(class_lines[:10]))
    if func_lines:
        pieces.append("Functions: " + "; ".join(func_lines[:10]))

    if not pieces:
        snippet = normalize_space("\n".join(lines[:40]))
        pieces.append(snippet[:800])

    return normalize_space(" ".join(pieces))[:1200]

def load_module_summaries(bricksrc_dir: Path) -> List[Dict]:
    docs: List[Dict] = []
    for name in MODULE_FILES:
        path = bricksrc_dir / name
        if not path.exists():
            continue

        summary = summarize_python_file(path)
        if not summary:
            continue

        docs.append(
            {
                "id": f"module_summary_{safe_id(name)}",
                "text": f"Module summary from {name}. {summary}",
                "metadata": {
                    "source": str(path),
                    "doc_type": "module_summary",
                    "label": name,
                },
            }
        )
    return docs

def build_builtin_docs() -> List[Dict]:
    docs: List[Dict] = []

    for cls, text in CLASS_GUIDANCE.items():
        docs.append(
            {
                "id": f"class_guidance_{safe_id(cls)}",
                "text": text,
                "metadata": {
                    "source": "builtin",
                    "doc_type": "class_guidance",
                    "label": cls,
                },
            }
        )

    for doc_id, text in GENERAL_RULES:
        docs.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": {
                    "source": "builtin",
                    "doc_type": "general_rule",
                    "label": doc_id,
                },
            }
        )

    return docs

def dedupe_docs(docs: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for doc in docs:
        key = normalize_space(doc["text"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.tolist()

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest compact Stage 2 task-specific Brick docs into Chroma.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--chroma-dir", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    chroma_dir = Path(args.chroma_dir).resolve()

    bricksrc_dir = project_root / "Brick" / "bricksrc"
    data_brick_dir = project_root / "data" / "brick"

    docs: List[Dict] = []
    docs.extend(build_builtin_docs())
    docs.extend(extract_relevant_definitions(bricksrc_dir / "definitions.csv"))
    docs.extend(load_family_guidance(data_brick_dir))
    docs.extend(load_module_summaries(bricksrc_dir))
    docs = dedupe_docs(docs)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(args.collection)
    except Exception:
        pass

    collection = client.create_collection(args.collection)

    model = SentenceTransformer(args.embed_model)

    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]
    embeds = embed_texts(model, texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeds,
    )

    manifest = {
        "project_root": str(project_root),
        "chroma_dir": str(chroma_dir),
        "collection": args.collection,
        "docs_indexed": len(docs),
        "doc_types": {},
        "allowed_classes": sorted(ALLOWED_CLASSES),
    }

    for d in docs:
        t = d["metadata"]["doc_type"]
        manifest["doc_types"][t] = manifest["doc_types"].get(t, 0) + 1

    if args.manifest_out:
        Path(args.manifest_out).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("✅ Stage 2 Brick task-doc ingestion complete.")
    print(f"Project root: {project_root}")
    print(f"Chroma path: {chroma_dir}")
    print(f"Collection: {args.collection}")
    print(f"Docs indexed: {len(docs)}")
    print("Doc types:")
    for k, v in sorted(manifest["doc_types"].items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
