# backend/scripts/verify_index.py
"""
Verify FAISS index + embeddings for the Gita dataset.

Checks performed:
- index file exists & loads
- metadata loads & count matches index.ntotal
- embedding model encodes queries with expected dimension
- basic semantic searches return sensible results
- sanity test: identical text -> near-zero distance
- prints top-k results for a few example queries for manual inspection

Run:
    cd backend/scripts
    python verify_index.py
"""

import os
import json
import sys
from pathlib import Path

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: pip install faiss-cpu sentence-transformers numpy")
    sys.exit(1)

# --- Config: adjust paths if needed ---
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
INDEX_PATH = BASE_DIR  / "data" / "indices" / "gita_faiss.index"
METAS_PATH = BASE_DIR  / "data" / "indices" / "gita_metas.json"

EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 5

def load_index(index_path):
    assert index_path.exists(), f"Index file not found: {index_path}"
    idx = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index from {index_path}")
    return idx

def load_metas(metas_path):
    assert metas_path.exists(), f"Metas file not found: {metas_path}"
    with open(metas_path, "r", encoding="utf8") as f:
        metas = json.load(f)
    print(f"Loaded {len(metas)} meta entries from {metas_path}")
    return metas

def sanity_checks(index, metas):
    # 1) index size vs metadata
    ntotal = index.ntotal
    assert ntotal == len(metas), f"Index size ({ntotal}) != number of meta entries ({len(metas)})"
    print(f"Index contains {ntotal} vectors — matches metadata count.")

def load_model(name):
    print("Loading embedding model:", name)
    model = SentenceTransformer(name)
    print("Model loaded.")
    return model

def embed_texts(model, texts):
    embs = model.encode(texts, convert_to_numpy=True)
    embs = embs.astype("float32")
    return embs

def search_index(index, query_vec, k=TOP_K):
    # if index expects normalized vectors (IP) ensure you encode similarly
    D, I = index.search(query_vec, k)
    return D, I

def pretty_print_results(D, I, metas):
    for i, (d_row, i_row) in enumerate(zip(D, I)):
        print(f"\n=== Query {i+1} results ===")
        for rank, (idx, dist) in enumerate(zip(i_row, d_row), start=1):
            if idx < 0:
                continue
            meta = metas[idx]
            print(f"Rank {rank} — idx {idx} — dist {dist:.4f} — verse {meta.get('chapter')}:{meta.get('verse')}")
            # print short preview
            print("   preview:", meta.get("english","")[:150].replace("\n"," "))

def identical_text_test(model, index):
    t = "धृतराष्ट्र उवाच धर्मक्षेत्रे कुरुक्षेत्रे samavetā yuyutsavaḥ"
    v = model.encode([t], convert_to_numpy=True).astype("float32")
    D, I = index.search(v, 2)
    print("\nIdentical-text sanity test (top 2):")
    print("Distances:", D[0])
    # Expect the top match distance to be very small relative to other distances
    return D, I

def run_example_queries(model, index, metas):
    queries = [
        "duty without attachment",        # semantic English query (expect 2:47-ish verses)
        "कर्तव्य बिना फल की चिंता मत करो",  # Hindi paraphrase
        "what did dhritarashtra ask sanjaya", # literal question (should retrieve 1:1)
    ]
    q_embs = embed_texts(model, queries)
    D, I = search_index(index, q_embs, k=TOP_K)
    pretty_print_results(D, I, metas)

def main():
    print("=== Gita index verification script ===")
    index = load_index(INDEX_PATH)
    metas = load_metas(METAS_PATH)
    sanity_checks(index, metas)

    model = load_model(EMBED_MODEL_NAME)

    # Check embedding dimension matches index dimension
    sample_emb = embed_texts(model, ["test"])[0]
    emb_dim = sample_emb.shape[0]
    print("Embedding dim:", emb_dim)
    if hasattr(index, "d"):
        idx_dim = index.d
    else:
        # Some FAISS indexes expose ntotal but not d; try reading from stored vectors (risky)
        try:
            idx_dim = index.reconstruct(0).shape[0]
        except Exception:
            idx_dim = None
    print("Index dim:", idx_dim)
    if idx_dim is not None:
        assert idx_dim == emb_dim, f"Embedding dim ({emb_dim}) != index dim ({idx_dim})"

    # Test: identical text yields very small distance to itself
    D_id, I_id = identical_text_test(model, index)

    # Example queries
    run_example_queries(model, index, metas)

    print("\nAll checks ran. Please inspect the printed results for correctness.\nIf top-k results for the example queries look relevant, your index is working.")

if __name__ == "__main__":
    main()
