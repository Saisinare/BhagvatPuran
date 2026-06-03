# RAG/data/ingest_hf_gita.py
from datasets import load_dataset
import jsonlines
import datetime
import os

os.makedirs("RAG/data/processed", exist_ok=True)
ds = load_dataset("JDhruv14/Bhagavad-Gita_Dataset", split="train")

out_path = "RAG/data/processed/gita_docs.jsonl"
with jsonlines.open(out_path, mode="w") as writer:
    for row in ds:
        # canonicalize fields; default to empty strings if missing
        chap = int(row.get("chapter") )
        verse = int(row.get("verse") )
        doc = {
            "id": f"gita_{chap}_{verse}",
            "chapter": chap,
            "verse": verse,
            "sanskrit": row.get("sanskrit", "").strip(),
            "transliteration": row.get("transliteration", "").strip() if row.get("transliteration") else "",
            "english": row.get("english", "").strip(),
            "hindi": row.get("hindi", "").strip(),
            "source": "JDhruv14/Bhagavad-Gita_Dataset",
            "entities": [],
            "themes": [],
            "commentaries": [],
            "tags": ["verse", "translation"],
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
        writer.write(doc)

print("Wrote:", out_path)
