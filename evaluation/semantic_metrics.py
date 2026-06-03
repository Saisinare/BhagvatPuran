"""
Advanced semantic metrics for RAGita evaluation.
Includes BERTScore and other transformer-based metrics.
"""
import re
from typing import List, Tuple, Optional
import numpy as np

class SemanticMetrics:
    """
    Semantic similarity metrics using pre-trained models.
    More sophisticated than BLEU/ROUGE for evaluating semantic quality.
    """
    
    def __init__(self):
        self._sentence_transformer = None
        self._bert_score_model = None
    
    def _get_sentence_transformer(self):
        """Lazy loading of sentence transformer for semantic similarity."""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")
                return None
        return self._sentence_transformer
    
    def semantic_similarity(self, reference: str, candidate: str) -> float:
        """
        Compute semantic similarity using sentence transformers.
        Returns cosine similarity between sentence embeddings.
        """
        model = self._get_sentence_transformer()
        if model is None:
            return 0.0
        
        try:
            embeddings = model.encode([reference, candidate])
            # Cosine similarity
            cos_sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(cos_sim)
        except Exception as e:
            print(f"Warning: Semantic similarity computation failed: {e}")
            return 0.0
    
    def bert_score_simple(self, reference: str, candidate: str) -> Tuple[float, float, float]:
        """
        Simplified BERTScore implementation using sentence similarity.
        Returns (precision, recall, f1) approximation.
        
        Note: This is a simplified version. For full BERTScore, use the bert-score library.
        """
        # Split into sentences for more granular comparison
        ref_sentences = [s.strip() for s in re.split(r'[.!?]+', reference) if s.strip()]
        cand_sentences = [s.strip() for s in re.split(r'[.!?]+', candidate) if s.strip()]
        
        if not ref_sentences or not cand_sentences:
            return 0.0, 0.0, 0.0
        
        model = self._get_sentence_transformer()
        if model is None:
            # Fallback to simple word overlap
            return self._word_overlap_score(reference, candidate)
        
        try:
            # Compute pairwise similarities
            ref_embeddings = model.encode(ref_sentences)
            cand_embeddings = model.encode(cand_sentences)
            
            # Compute similarity matrix
            similarities = np.dot(cand_embeddings, ref_embeddings.T)
            
            # BERTScore-style precision: for each candidate sentence, max similarity to reference
            precision_scores = np.max(similarities, axis=1)
            precision = np.mean(precision_scores)
            
            # BERTScore-style recall: for each reference sentence, max similarity to candidate  
            recall_scores = np.max(similarities, axis=0)
            recall = np.mean(recall_scores)
            
            # F1 score
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            return float(precision), float(recall), float(f1)
            
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return self._word_overlap_score(reference, candidate)
    
    def _word_overlap_score(self, reference: str, candidate: str) -> Tuple[float, float, float]:
        """Fallback word overlap score when transformers not available."""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        if not ref_words and not cand_words:
            return 1.0, 1.0, 1.0
        if not ref_words or not cand_words:
            return 0.0, 0.0, 0.0
        
        intersection = ref_words & cand_words
        precision = len(intersection) / len(cand_words)
        recall = len(intersection) / len(ref_words)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def contextual_relevance(self, query: str, answer: str) -> float:
        """
        Measure how well the answer addresses the query contextually.
        Uses semantic similarity between query and answer.
        """
        return self.semantic_similarity(query, answer)
    
    def factual_consistency(self, context_passages: List[str], generated_answer: str) -> float:
        """
        Measure factual consistency between generated answer and source passages.
        Returns average semantic similarity to most relevant passages.
        """
        if not context_passages:
            return 0.0
        
        model = self._get_sentence_transformer()
        if model is None:
            return 0.0
        
        try:
            # Find most similar passages
            similarities = [self.semantic_similarity(passage, generated_answer) 
                          for passage in context_passages]
            
            # Return average of top-3 similarities (or all if less than 3)
            top_similarities = sorted(similarities, reverse=True)[:3]
            return sum(top_similarities) / len(top_similarities)
            
        except Exception as e:
            print(f"Warning: Factual consistency computation failed: {e}")
            return 0.0