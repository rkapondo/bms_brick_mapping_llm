import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse

import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

# Collections
COLLECTION_NAME = "rag_week1"          # your local ./data documents
BRICK_COLLECTION_NAME = "brick"        # created by ingest_brick.py
BRICK_DOCS_WEB_COLLECTION = "brick_docs_web"  # NEW: docs.brickschema.org pages

# LM Studio / OpenAI-compatible endpoint
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")  # any string works
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "mistralai/ministral-3-14b-reasoning")

# Chunking (whitespace token chunks)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 4

# Context limits passed into the LLM
MAX_TOTAL_CONTEXT_CHARS = 9000   # total budget across all sources
MAX_CHARS_PER_CHUNK = 3000       # cap each chunk

# Web docs ingest
USER_AGENT = "Mozilla/5.0 (RAG Brick Docs Indexer)"
WEB_TIMEOUT = 30

# Curated starter set
BRICK_DOCS_URLS = [
    # Intro
    "https://docs.brickschema.org/intro.html",

    # Core Brick
    "https://docs.brickschema.org/brick/overview.html",
    "https://docs.brickschema.org/brick/concepts.html",
    "https://docs.brickschema.org/brick/relationships.html",
    "https://docs.brickschema.org/brick/timeseries.html",
    "https://docs.brickschema.org/brick/aliases.html",

    # Lifecycle
    "https://docs.brickschema.org/lifecycle/creation.html",
    "https://docs.brickschema.org/lifecycle/inference.html",
    "https://docs.brickschema.org/lifecycle/versioning.html",

    # Modeling guides
    "https://docs.brickschema.org/modeling/collections.html",
    "https://docs.brickschema.org/modeling/meters.html",

    # Extra
    "https://docs.brickschema.org/extra/brick-rec.html",

    # Software
    "https://docs.brickschema.org/software/interfaces.html",
]

# ----------------------------
# Helpers
# ----------------------------
def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple whitespace-based chunking with overlap.
    chunk_size + overlap are in "word tokens" (split() tokens), not model tokens.
    """
    text = re.sub(r"\r\n", "\n", text)
    tokens = text.split()
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def load_documents(data_dir: Path) -> List[Dict]:
    docs = []
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()

        if path.suffix.lower() == ".ttl" and ("brick" in path.name.lower()):
            print("[INGEST]", path)

        # For Week 1: handle common text formats
        if suffix in [".txt", ".md", ".ttl", ".rdf", ".json", ".yaml", ".yml", ".csv"]:
            text = read_text_file(path)
            if text.strip():
                docs.append({"source": str(path), "text": text})

        # Optional PDF support (disabled by default)
        # elif suffix == ".pdf":
        #     from pypdf import PdfReader
        #     reader = PdfReader(str(path))
        #     pages = [page.extract_text() or "" for page in reader.pages]
        #     docs.append({"source": str(path), "text": "\n".join(pages)})

    return docs

def extract_ttl_subject_block(text: str, subject: str, max_block_chars: int = 6000) -> str | None:
    """
    Extract a Turtle 'block' for a given subject by capturing from the first
    occurrence of a subject definition up to the terminating '.' of that statement.

    Tries patterns like:
      - brick:feeds a ...
      - brick:feeds rdf:type ...
      - :feeds a ...
    """
    subj_local = subject.split(":")[-1]
    patterns = [
        rf"(^|\n)\s*{re.escape(subject)}\s+(a|rdf:type)\s+",
        rf"(^|\n)\s*:{re.escape(subj_local)}\s+(a|rdf:type)\s+",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue

        start = m.start()
        # Find the end of the Turtle statement (first "\n.\n" after start)
        end = text.find("\n.\n", start)
        if end == -1:
            # fallback: find "\n.\r\n" etc.
            end = re.search(r"\n\.\s*(\n|$)", text[start:], flags=re.IGNORECASE)
            if end:
                end = start + end.start()
            else:
                end = min(len(text), start + max_block_chars)

        block = text[start:end + 3]  # include "\n.\n"
        return block[:max_block_chars]

    return None


def lmstudio_chat(messages: List[Dict], temperature: float = 0.1) -> str:
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if not r.ok:
        print("LM Studio error status:", r.status_code)
        print("LM Studio error body:", r.text)
        r.raise_for_status()

    data = r.json()
    return data["choices"][0]["message"]["content"]


def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def html_to_text_basic(html: str) -> Tuple[str, Optional[str]]:
    """
    Minimal HTML -> text without extra deps.
    Extract <title> and remove scripts/styles/nav-ish content roughly.
    This is not perfect, but it's good enough for starter RAG.
    """
    title = None
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", re.sub(r"<.*?>", "", m.group(1))).strip()

    # drop script/style
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.IGNORECASE | re.DOTALL)

    # remove common boilerplate-ish tags (rough)
    html = re.sub(r"</?(nav|footer|header|aside)\b[^>]*>", "\n", html, flags=re.IGNORECASE)

    # convert <br> and block-ish tags to newlines
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</(p|div|h1|h2|h3|h4|h5|h6|li|ul|ol|section|article|table|tr|pre)>", "\n", html, flags=re.IGNORECASE)

    # strip remaining tags
    text = re.sub(r"<[^>]+>", "", html)

    # unescape a few common entities
    text = (
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )

    return normalize_whitespace(text), title


def fetch_url_text(url: str) -> Tuple[str, Optional[str]]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=WEB_TIMEOUT)
    r.raise_for_status()
    text, title = html_to_text_basic(r.text)
    return text, title


def safe_id(prefix: str, s: str) -> str:
    # stable, filesystem/Chroma-friendly
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", s)
    s = s.strip("-")
    return f"{prefix}-{s}"[:160]

def lookup_in_file(path: Path, needle: str, window_chars: int = 2500) -> str | None:
    text = read_text_file(path)
    hay = text.lower()

    # allow matching local name if prefixes differ
    local = needle.split(":")[-1].lower()
    candidates = [needle.lower(), local]

    best_idx = None
    best_score = -1

    for cand in candidates:
        start = 0
        while True:
            idx = hay.find(cand, start)
            if idx == -1:
                break

            # score the neighborhood: prefer OWL property definitions over SHACL shapes
            neighborhood = hay[max(0, idx - 500): idx + 500]
            score = 0
            if "owl:objectproperty" in neighborhood:
                score += 3
            if "owl:datatypeproperty" in neighborhood:
                score += 3
            if "rdf:property" in neighborhood:
                score += 2
            if "rdfs:domain" in neighborhood:
                score += 2
            if "rdfs:range" in neighborhood:
                score += 2
            if "sh:propertyshape" in neighborhood:
                score -= 2  # de-prefer shapes if we can

            if score > best_score:
                best_score = score
                best_idx = idx

            start = idx + 1

    if best_idx is None:
        return None

    start = max(0, best_idx - window_chars // 2)
    end = min(len(text), best_idx + window_chars // 2)
    return text[start:end]

def lookup_property_in_brick(col, property_name: str) -> List[Tuple[str, Dict]]:
    """
    Deterministic lookup for a specific Brick property inside TTL chunks.
    """
    results = col.get(include=["documents", "metadatas"])
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    # ---- DEBUG PRINTS ----
    print("\n[DEBUG] PROPERTY SEARCHING FOR:", property_name)
    print("[DEBUG] TOTAL BRICK CHUNKS:", len(docs))
    # ----------------------

    hits = []

    # IMPORTANT: search only local name (after colon)
    local_name = property_name.split(":")[-1].lower()

    for text, meta in zip(docs, metas):
        if not text:
            continue

        if local_name in text.lower():
            hits.append((text, meta))

    print("[DEBUG] MATCHES FOUND:", len(hits))

    return hits[:5]  # limit to first few matches


# ----------------------------
# Indexing + Retrieval
# ----------------------------
def build_or_load_collections():
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    st_model = SentenceTransformer(embed_model_name)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Main (your data/) collection
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.create_collection(COLLECTION_NAME)

    # Brick collection (created by ingest_brick.py)
    try:
        brick_col = client.get_collection(BRICK_COLLECTION_NAME)
    except Exception:
        brick_col = None

    # Brick docs web collection (NEW)
    try:
        brick_docs_col = client.get_collection(BRICK_DOCS_WEB_COLLECTION)
    except Exception:
        brick_docs_col = client.create_collection(BRICK_DOCS_WEB_COLLECTION)

    return client, col, brick_col, brick_docs_col, st_model


def index_documents_if_needed(col, st_model):
    """
    Index ./data into the rag_week1 collection.
    For convenience, we only build it if the collection is currently empty.
    """
    try:
        if col.count() > 0:
            return
    except Exception:
        pass

    docs = load_documents(DATA_DIR)
    if not docs:
        raise RuntimeError(
            "No documents found under ./data. Add files to data/brick, data/manuals, or data/examples."
        )

    ids, texts, metas = [], [], []
    idx = 0

    for doc in docs:
        chunks = split_into_chunks(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chunks:
            if not c.strip():
                continue
            ids.append(f"data-{idx}")
            texts.append(c)
            metas.append({"source": doc["source"], "collection": "data"})
            idx += 1

    print(f"[INFO] Embedding {len(texts)} chunks from ./data ...")
    embeddings = st_model.encode(texts, normalize_embeddings=True).tolist()
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)


def index_brick_docs_web_if_needed(brick_docs_col, st_model, urls: List[str]):
    """
    Index curated docs.brickschema.org pages into brick_docs_web.
    Skips if collection already has something in it.
    """
    try:
        if brick_docs_col.count() > 0:
            return
    except Exception:
        pass

    texts, ids, metas = [], [], []
    for url in urls:
        try:
            page_text, title = fetch_url_text(url)
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            continue

        if not page_text.strip():
            print(f"[WARN] Empty text for {url}")
            continue

        chunks = split_into_chunks(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        base = safe_id("brickdocs", urlparse(url).path.replace("/", "_") or "index")
        for i, c in enumerate(chunks):
            if not c.strip():
                continue
            ids.append(f"{base}-{i}")
            texts.append(c)
            metas.append({
                "source": url,
                "url": url,
                "title": title or "",
                "collection": "brick_docs_web",
            })

    if not texts:
        print("[WARN] No Brick docs web pages indexed.")
        return

    print(f"[INFO] Embedding {len(texts)} chunks from Brick docs web ...")
    embeddings = st_model.encode(texts, normalize_embeddings=True).tolist()
    brick_docs_col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)


def retrieve(col, st_model, query: str, top_k: int) -> List[Tuple[str, Dict]]:
    q_emb = st_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return list(zip(docs, metas))


def build_context(retrieved: List[Tuple[str, Dict]]) -> str:
    """
    Build a bounded context string with [Source i] blocks.
    """
    context_blocks = []
    total = 0

    for i, (text, meta) in enumerate(retrieved, start=1):
        meta = meta or {}
        src = meta.get("source", "unknown")
        coll = meta.get("collection", "unknown")
        title = meta.get("title", "")

        trimmed = text[:MAX_CHARS_PER_CHUNK]
        head = f"[Source {i}] ({coll}) {src}"
        if title:
            head += f" — {title}"
        block = f"{head}\n{trimmed}\n"

        if total + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break

        context_blocks.append(block)
        total += len(block)

    return "\n".join(context_blocks)


def merge_retrievals(
    q: str,
    data_hits: List[Tuple[str, Dict]],
    brick_hits: List[Tuple[str, Dict]],
    web_hits: List[Tuple[str, Dict]],
) -> List[Tuple[str, Dict]]:
    """
    Heuristic ordering:
    - If query looks ontology/relationship-ish, prefer Brick TTL first (then web docs, then local data).
      This remains true even if the query includes words like "definition" or "meaning".
    - If query looks like a plain definition/overview question, prefer official web docs first
      (then local data, then Brick).
    - Otherwise, prefer local data first, then web docs, then Brick.
    """

    ONTOLOGY_HINTS = [
        "feeds", "haspoint", "relationship", "property", "ontology",
        "rdf", "rdfs", "owl", "domain", "range", "subclass", "subproperty",
        "inverse", "axiom",
        # NOTE: intentionally NOT including "definition"/"meaning" here
        # because those words appear in both ontology and non-ontology questions.
    ]

    DEFINITION_HINTS = [
        "what is", "define", "definition", "overview", "explain",
        "what problem", "purpose",
    ]

    q_low = q.lower()

    prefer_ontology = any(h in q_low for h in ONTOLOGY_HINTS)
    is_definition = any(h in q_low for h in DEFINITION_HINTS)

    # ensure collection tags exist for clarity
    def tag(hits: List[Tuple[str, Dict]], name: str) -> List[Tuple[str, Dict]]:
        fixed = []
        for t, m in hits:
            m = dict(m or {})
            m.setdefault("collection", name)
            fixed.append((t, m))
        return fixed

    data_hits = tag(data_hits, "data")
    brick_hits = tag(brick_hits, "brick")
    web_hits = tag(web_hits, "brick_docs_web")

    # Ontology/property questions should go to Brick TTL first (even if phrased as "definition/meaning")
    if prefer_ontology:
        return brick_hits + web_hits + data_hits

    # Plain definition/overview questions should trust official docs first
    if is_definition:
        return web_hits + data_hits + brick_hits

    # Default ordering
    return data_hits + web_hits + brick_hits


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    client, col, brick_col, brick_docs_col, st_model = build_or_load_collections()

    # Index ./data if needed
    index_documents_if_needed(col, st_model)

    # Index Brick docs web if needed
    index_brick_docs_web_if_needed(brick_docs_col, st_model, BRICK_DOCS_URLS)

    if brick_col is not None:
        print("Brick ontology collection loaded")
    else:
        print("Brick ontology collection not found (run ingest_brick.py)")

    print("Brick docs web collection loaded")
    print("\nRAG ready. Ask a question (Ctrl+C to exit).\n")

    while True:
        q = input("Q: ").strip()

        # Treat shell commands as commands, not questions
        if q.strip().startswith("grep "):
            try:
                out = subprocess.check_output(q, shell=True, text=True, stderr=subprocess.STDOUT)
                print("\n" + out + "\n")
            except subprocess.CalledProcessError as e:
                print("\n" + e.output + "\n")
            continue

        # --- Symbolic lookup for Brick terms in Brick.ttl ---
        prop_match = re.search(r"\bbrick:[A-Za-z_][A-Za-z0-9_]*\b", q.lower())
        if prop_match:
            prop = prop_match.group(0)
            brick_path = DATA_DIR / "Brick.ttl"

            if brick_path.exists():
                full = read_text_file(brick_path)
                block = extract_ttl_subject_block(full, prop)

                if block:
                    retrieved = [(
                        block,
                        {"source": str(brick_path), "collection": "data"}
                    )]

                    # 1. Increase web retrieval for richer explanation
                    retrieved += retrieve(brick_docs_col, st_model, q, top_k=6)

                    # 2. Add one targeted relationship example query
                    local_name = prop.split(":")[-1]
                    retrieved += retrieve(
                        brick_docs_col,
                        st_model,
                        f"{local_name} relationship examples",
                        top_k=4
                    )

                    context = build_context(retrieved)
            
                    # ... then continue with system/user prompt + lmstudio_chat
                    # (IMPORTANT: skip the normal retrieval branch)
                    system = (
                        "You are a helpful assistant. Answer the user's question using ONLY the provided context.\n"
                        "If the context does not contain the answer, say: \"I don't know based on the provided documents.\" \n"
                        "Cite sources by writing [Source 1], [Source 2], etc.\n"
                    )

                    user = f"Question: {q}\n\nContext:\n{context}"

                    answer = lmstudio_chat(
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.1,
                    )

                    print("\nA:", answer, "\n")
                    print("Retrieved sources:")
                    for i, (_, meta) in enumerate(retrieved, start=1):
                        meta = meta or {}
                        src = meta.get("source", "unknown")
                        coll = meta.get("collection", "unknown")
                        print(f"  [Source {i}] ({coll}) {src}")
                    print()
                    continue


        if not q:
            continue

        # Retrieve from ./data
        retrieved_data = retrieve(col, st_model, q, top_k=TOP_K)

        # Retrieve from Brick ontology (if available)
        retrieved_brick = []
        if brick_col is not None:
            retrieved_brick = retrieve(brick_col, st_model, q, top_k=TOP_K * 2)

        # Retrieve from Brick docs web (curated)
        retrieved_web = retrieve(brick_docs_col, st_model, q, top_k=TOP_K)

        retrieved = merge_retrievals(q, retrieved_data, retrieved_brick, retrieved_web)
        context = build_context(retrieved)

        system = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain the answer, say: \"I don't know based on the provided documents.\" \n"
            "Cite sources by writing [Source 1], [Source 2], etc.\n"
        )

        user = f"Question: {q}\n\nContext:\n{context}"

        answer = lmstudio_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )

        print("\nA:", answer, "\n")
        print("Retrieved sources:")
        for i, (_, meta) in enumerate(retrieved, start=1):
            meta = meta or {}
            src = meta.get("source", "unknown")
            coll = meta.get("collection", "unknown")
            print(f"  [Source {i}] ({coll}) {src}")
        print()
