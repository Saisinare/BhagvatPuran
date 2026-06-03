import json
import random
p = "RAG/data/processed/gita_docs.jsonl"
with open(p, 'r', encoding='utf8') as f:
    lines = f.readlines()
print("Docs:", len(lines))
for l in random.sample(lines, 3):
    obj = json.loads(l)
    print(obj["id"], obj["chapter"], obj["verse"],
          obj["sanskrit"][:60], obj["english"][:80])
