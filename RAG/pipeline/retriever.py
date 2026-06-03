# RAG/pipeline/retriever.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
import io

# adjust if folder depth differs
# REPO_ROOT = Path(__file__).resolve().parents[3]
# INDICES_DIR = REPO_ROOT / "data" / "indices"
# Resolve paths relative to the repository regardless of CWD
RAG_DIR = Path(__file__).resolve().parents[1]  # .../RAG
DATA_DIR = RAG_DIR / "data"
INDICES_DIR = DATA_DIR / "indices"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSED_DIR = DATA_DIR / "pre_processed_data"

INDEX_PATH = str(INDICES_DIR / "gita_faiss.index")
METAS_PATH = str(INDICES_DIR / "gita_metas.json")
DOCS_PATH_DEFAULTS = [
    str(PROCESSED_DIR / "gita_docs.jsonl"),
    str(PREPROCESSED_DIR / "gita_docs.jsonl"),
]

EMBED_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Better for Q&A retrieval


class Retriever:
    def __init__(self, model_name=EMBED_MODEL_NAME, index_path=INDEX_PATH, metas_path=METAS_PATH):
        self.model = SentenceTransformer(model_name)
        self.docs_path = self._resolve_docs_path()
        self._ensure_index(index_path, metas_path)
        self.index = faiss.read_index(str(index_path))
        with open(metas_path, "r", encoding="utf8") as f:
            self.metas = json.load(f)

    def _resolve_docs_path(self) -> str:
        for p in DOCS_PATH_DEFAULTS:
            if os.path.exists(p):
                return p
        # None found
        raise RuntimeError(
            "Could not find a processed Gita JSONL file. Looked for: "
            + ", ".join(str(Path(p).resolve()) for p in DOCS_PATH_DEFAULTS)
            + ". Please run your ingest step or point Retriever to the correct path."
        )

    def _ensure_index(self, index_path: str, metas_path: str):
        if os.path.exists(index_path) and os.path.exists(metas_path):
            return
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        texts, metas = self._load_docs()
        if not texts:
            raise RuntimeError(
                f"No texts loaded from {Path(self.docs_path).resolve()}. Cannot build index.")

        embeddings = self.model.encode(
            texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, index_path)
        with open(metas_path, "w", encoding="utf8") as f:
            json.dump(metas, f, ensure_ascii=False, indent=2)

    def _load_docs(self):
        def iter_json_objects_multiline(file_path: str):
            buffer_lines = []
            brace_balance = 0
            in_object = False
            with io.open(file_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.rstrip("\n")
                    if not in_object:
                        if line.strip().startswith("{"):
                            in_object = True
                        else:
                            continue
                    buffer_lines.append(line)
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
                            cleaned = chunk.rstrip(", ")
                            yield json.loads(cleaned)

        texts, metas = [], []
        if not os.path.exists(self.docs_path):
            return texts, metas
        for obj in iter_json_objects_multiline(self.docs_path):
            # Enhanced text representation for better semantic retrieval
            english = obj.get("english", "")
            transliteration = obj.get("transliteration", "")
            
            # Create richer text representation with emphasis on English content
            text_parts = []
            
            # Primary content (weighted more heavily)
            if english:
                text_parts.append(english)
                text_parts.append(english)  # Duplicate for emphasis
            
            # Secondary content for broader matching
            if transliteration:
                text_parts.append(transliteration)
            
            # Add conceptual keywords based on content analysis
            keywords = self._extract_conceptual_keywords(english)
            if keywords:
                text_parts.append(" ".join(keywords))
            
            text = " ".join(text_parts).strip()
            
            if len(text) < 10:
                continue
                
            texts.append(text)
            metas.append({
                "id": obj.get("id"),
                "chapter": obj.get("chapter"),
                "verse": obj.get("verse"),
                "english": english,
                "full_text": text  # Store full text for debugging
            })
        return texts, metas

    def embed(self, texts):
        emb = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        return emb

    def semantic_search(self, query, top_k=5):
        """Enhanced semantic search with query expansion and multi-strategy retrieval"""
        
        # Strategy 1: Direct semantic search
        q_emb = self.embed([query])
        D, I = self.index.search(q_emb, top_k * 2)  # Get more candidates for reranking
        
        # Strategy 2: Query expansion for better retrieval
        expanded_query = self._expand_query(query)
        if expanded_query != query:
            q_exp_emb = self.embed([expanded_query])
            D_exp, I_exp = self.index.search(q_exp_emb, top_k)
            
            # Combine results from both searches
            combined_indices = list(I[0]) + list(I_exp[0])
            combined_distances = list(D[0]) + [d + 0.1 for d in D_exp[0]]  # Slight penalty for expanded
        else:
            combined_indices = list(I[0])
            combined_distances = list(D[0])
        
        # Remove duplicates and sort by distance
        seen_indices = set()
        unique_results = []
        for idx, dist in zip(combined_indices, combined_distances):
            if idx < 0 or idx in seen_indices:
                continue
            seen_indices.add(idx)
            unique_results.append((idx, dist))
        
        # Sort by distance and take top_k
        unique_results.sort(key=lambda x: x[1])
        unique_results = unique_results[:top_k]
        
        # Convert to result format
        results = []
        for idx, score in unique_results:
            meta = self.metas[idx]
            results.append({
                "id": meta.get("id"),
                "chapter": meta.get("chapter"),
                "verse": meta.get("verse"),
                "english": meta.get("english"),
                "score": float(score)
            })
        
        return results
    
    def _expand_query(self, query):
        """Expand queries with domain-specific terms for better retrieval"""
        query_lower = query.lower()
        
        # Query expansion mapping for Gita-specific terms
        expansions = {
            # Concept mappings
            "soul": "soul atman self eternal spirit consciousness",
            "duty": "duty dharma righteousness prescribed action",
            "action": "action karma deed work activity",
            "knowledge": "knowledge jnana wisdom understanding truth",
            "devotion": "devotion bhakti love surrender worship",
            "meditation": "meditation dhyana yoga concentration focus",
            "peace": "peace shanti tranquility calm serenity",
            "mind": "mind manas thought consciousness intellect",
            "wisdom": "wisdom jnana knowledge understanding truth",
            "yoga": "yoga union discipline practice path",
            "lord": "lord krishna divine supreme god",
            "arjuna": "arjuna pandava warrior disciple",
            "krishna": "krishna lord divine supreme bhagavan",
            "gita": "gita bhagavad dialogue teaching scripture",
            "righteousness": "righteousness dharma duty moral law",
            "liberation": "liberation moksha freedom release salvation",
            "attachment": "attachment desire clinging bondage",
            "detachment": "detachment vairagya dispassion equanimity",
            "sacrifice": "sacrifice yajna offering dedication",
            "surrender": "surrender sharanagati devotion submission",
            "reality": "reality truth brahman absolute existence",
            "illusion": "illusion maya delusion ignorance",
            "three gunas": "three gunas sattva rajas tamas qualities nature",
            "evil": "evil sin adharma wrong negative",
            "suffering": "suffering pain sorrow grief distress",
            "fear": "fear anxiety worry dread concern",
            "anger": "anger rage wrath fury irritation",
            "desire": "desire kama lust craving want",
            "ego": "ego ahamkara pride self identity",
            "battlefield": "battlefield kurukshetra war conflict struggle",
            "verses": "verses shlokas lines stanzas chapters"
        }
        
        # Apply expansions
        expanded_terms = []
        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_terms.append(expansion)
        
        if expanded_terms:
            return query + " " + " ".join(expanded_terms)
        
        return query
    
    def _extract_conceptual_keywords(self, text):
        """Extract conceptual keywords from verse text for better retrieval"""
        if not text:
            return []
            
        text_lower = text.lower()
        keywords = []
        
        # Concept detection patterns
        concept_patterns = {
            "soul_concepts": ["soul", "atman", "self", "embodied", "depart", "dwell", "eternal", "indestructible"],
            "action_concepts": ["action", "duty", "work", "perform", "deed", "karma", "activity", "engage"],
            "knowledge_concepts": ["knowledge", "wisdom", "understand", "realize", "know", "learn", "truth"],
            "devotion_concepts": ["devotion", "worship", "surrender", "love", "faith", "devotee", "dedicated"],
            "meditation_concepts": ["meditation", "mind", "concentration", "focus", "yoga", "discipline"],
            "divine_concepts": ["divine", "god", "lord", "supreme", "brahman", "krishna", "sacred"],
            "ethical_concepts": ["righteousness", "dharma", "right", "wrong", "moral", "virtue", "sin"],
            "emotional_concepts": ["peace", "joy", "fear", "anger", "desire", "attachment", "detachment"],
            "philosophical_concepts": ["reality", "illusion", "maya", "existence", "being", "consciousness"],
            "practical_concepts": ["path", "way", "practice", "discipline", "control", "achieve"]
        }
        
        for concept_type, terms in concept_patterns.items():
            if any(term in text_lower for term in terms):
                # Add the concept type as a keyword
                concept_name = concept_type.replace("_concepts", "").replace("_", " ")
                keywords.append(concept_name)
                
                # Add matching terms
                matching_terms = [term for term in terms if term in text_lower]
                keywords.extend(matching_terms[:2])  # Limit to avoid over-weighting
        
        return keywords[:5]  # Limit total keywords

    def embed(self, texts):
        """Embed texts using the sentence transformer model"""
        emb = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        return emb

    def batch_embed(self, texts, batch_size=16):
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True).astype("float32")
