# RAG/pipeline/reranker.py
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=5):
        """Enhanced reranking with multiple signals"""
        if not candidates:
            return []
            
        # candidates: list of dicts with "english" or "text" field
        texts = [c.get("english", "") for c in candidates]
        pairs = [[query, t] for t in texts]
        
        # Get cross-encoder scores
        cross_encoder_scores = self.model.predict(pairs)
        
        scored = []
        for i, (c, ce_score) in enumerate(zip(candidates, cross_encoder_scores)):
            c2 = c.copy()
            
            # Primary score from cross-encoder
            c2["cross_encoder_score"] = float(ce_score)
            
            # Additional ranking signals
            text = c.get("english", "")
            
            # Length bonus (moderate length preferred)
            length_score = self._calculate_length_score(text)
            
            # Keyword overlap score
            keyword_score = self._calculate_keyword_overlap(query, text)
            
            # Chapter priority (some chapters more fundamental)
            chapter_score = self._calculate_chapter_priority(c.get("chapter", 1))
            
            # Semantic search score (from retrieval)
            retrieval_score = 1.0 / (1.0 + c.get("score", 0))  # Lower distance = higher score
            
            # Composite score with weights
            composite_score = (
                0.5 * ce_score +           # Primary signal
                0.2 * keyword_score +      # Keyword relevance
                0.15 * length_score +      # Content length appropriateness  
                0.1 * chapter_score +      # Chapter importance
                0.05 * retrieval_score     # Original retrieval signal
            )
            
            c2["rerank_score"] = float(composite_score)
            c2["length_score"] = length_score
            c2["keyword_score"] = keyword_score
            c2["chapter_score"] = chapter_score
            c2["retrieval_score"] = retrieval_score
            
            scored.append(c2)
        
        scored_sorted = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
        return scored_sorted[:top_k]
    
    def _calculate_length_score(self, text):
        """Score based on text length - moderate length preferred"""
        if not text:
            return 0.0
        
        length = len(text.split())
        
        # Optimal range: 15-50 words
        if 15 <= length <= 50:
            return 1.0
        elif 10 <= length <= 70:
            return 0.8
        elif 5 <= length <= 100:
            return 0.6
        else:
            return 0.3
    
    def _calculate_keyword_overlap(self, query, text):
        """Calculate keyword overlap between query and text"""
        if not query or not text:
            return 0.0
            
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "shall"}
        
        query_words = query_words - stop_words
        text_words = text_words - stop_words
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words & text_words)
        return overlap / len(query_words)
    
    def _calculate_chapter_priority(self, chapter):
        """Priority scoring based on chapter importance for foundational concepts"""
        if not chapter:
            return 0.5
            
        # Chapter priority mapping based on fundamental teachings
        chapter_priorities = {
            2: 1.0,   # Fundamental teachings about soul, duty, yoga
            3: 0.9,   # Karma yoga
            4: 0.9,   # Knowledge and action
            6: 0.8,   # Meditation yoga  
            7: 0.8,   # Knowledge of the absolute
            9: 0.8,   # Royal knowledge
            12: 0.8,  # Devotion
            15: 0.7,  # Supreme person
            18: 0.9,  # Conclusion with key teachings
        }
        
        return chapter_priorities.get(chapter, 0.6)
