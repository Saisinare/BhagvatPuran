import json
import io
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load your model
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Paths
input_path = "RAG/data/processed/gita_docs.jsonl"
index_path = "RAG/data/indices/gita_faiss.index"
meta_path = "RAG/data/indices/gita_metas.json"

# Prepare lists
texts, metas = [], []

def iter_json_objects_multiline(file_path: str):
    """Yield JSON objects from a file where each object may span multiple lines.

    This parser accumulates lines until a full JSON object is formed based on
    brace/bracket balance and then attempts to json.loads() the accumulated
    chunk. It tolerates whitespace and blank lines between objects.
    """
    buffer_lines = []
    brace_balance = 0
    in_object = False

    with io.open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not in_object:
                # Skip leading whitespace/blank lines until an object starts
                if line.strip().startswith("{"):
                    in_object = True
                else:
                    continue

            buffer_lines.append(line)

            # Update balance counters roughly; this is a heuristic but
            # sufficient for well-formed JSON without stray braces in strings.
            # To avoid counting braces inside strings, we do a simple state
            # machine for quotes.
            in_string = False
            escaped = False
            for ch in line:
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == '{' or ch == '[':
                        brace_balance += 1
                    elif ch == '}' or ch == ']':
                        brace_balance -= 1

            if in_object and brace_balance <= 0:
                chunk = "\n".join(buffer_lines).strip()
                buffer_lines = []
                brace_balance = 0
                in_object = False
                if not chunk:
                    continue
                try:
                    yield json.loads(chunk)
                except json.JSONDecodeError:
                    # As a fallback, try to recover by removing trailing commas
                    # or other minor formatting issues.
                    cleaned = chunk.rstrip(", ")
                    yield json.loads(cleaned)


# Read all docs (supports multi-line JSON objects)
for obj in iter_json_objects_multiline(input_path):
    text = " ".join([
        obj.get("sanskrit", ""),
        obj.get("transliteration", ""),
        obj.get("english", ""),
        obj.get("hindi", "")
    ]).strip()
    if len(text) < 10:
        continue
    texts.append(text)
    metas.append({
        "id": obj["id"],
        "chapter": obj["chapter"],
        "verse": obj["verse"],
        "english": obj.get("english", "")[:120]
    })

print(f"Encoding {len(texts)} verses...")
embeddings = model.encode(texts, batch_size=16,
                          show_progress_bar=True, convert_to_numpy=True)
embeddings = embeddings.astype("float32")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save files
os.makedirs("RAG/data/indices", exist_ok=True)
faiss.write_index(index, index_path)
with open(meta_path, "w", encoding="utf8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)

print(f"Saved index to {index_path} with {len(metas)} entries.")
