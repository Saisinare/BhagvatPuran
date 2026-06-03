"""
Core evaluation metrics for RAGita system.
Implemented in ascending order of significance for academic papers.
"""
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import math

class AnswerQualityMetrics:
    """
    Basic answer quality metrics (Significance Level: 1-2)
    BLEU, ROUGE-L - traditional but less important for modern RAG evaluation
    """
    
    @staticmethod
    def bleu_score(reference: str, candidate: str, n: int = 4) -> float:
        """
        Compute BLEU score between reference and candidate text.
        Simple implementation for RAG evaluation.
        """
        def get_ngrams(text: str, n: int) -> Dict[tuple, int]:
            words = text.lower().split()
            ngrams = defaultdict(int)
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                ngrams[ngram] += 1
            return ngrams
        
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(cand_words) == 0:
            return 0.0
            
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_words) / len(cand_words)))
        
        # Compute precision for each n-gram order
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = get_ngrams(reference, i)
            cand_ngrams = get_ngrams(candidate, i)
            
            overlap = 0
            total = 0
            for ngram, count in cand_ngrams.items():
                overlap += min(count, ref_ngrams.get(ngram, 0))
                total += count
            
            if total == 0:
                precisions.append(0)
            else:
                precisions.append(overlap / total)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            return bp * geo_mean
        return 0.0
    
    @staticmethod
    def rouge_l(reference: str, candidate: str) -> float:
        """
        Compute ROUGE-L score (Longest Common Subsequence based).
        """
        def lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1].lower() == y[j-1].lower():
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        ref_words = reference.split()
        cand_words = candidate.split()
        
        if len(ref_words) == 0 and len(cand_words) == 0:
            return 1.0
        if len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
        
        lcs_len = lcs_length(ref_words, cand_words)
        precision = lcs_len / len(cand_words) if len(cand_words) > 0 else 0
        recall = lcs_len / len(ref_words) if len(ref_words) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class RetrievalMetrics:
    """
    Retrieval effectiveness metrics (Significance Level: 3-4)
    Critical for RAG systems - measures how well relevant passages are found
    """
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Recall@k: fraction of relevant documents retrieved in top-k results
        """
        if not relevant_docs:
            return 1.0 if not retrieved_docs[:k] else 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        Precision@k: fraction of retrieved documents that are relevant in top-k
        """
        if not retrieved_docs[:k]:
            return 0.0
        
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        return len(retrieved_set & relevant_set) / len(retrieved_docs[:k])
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        MRR: Mean Reciprocal Rank - average of reciprocal ranks of first relevant item
        """
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """
        nDCG@k: Normalized Discounted Cumulative Gain
        Assumes binary relevance (relevant=1, not relevant=0)
        """
        def dcg(relevances: List[int], k: int) -> float:
            return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]) if rel > 0)
        
        # Create relevance vector for retrieved docs
        relevant_set = set(relevant_docs)
        relevances = [1 if doc in relevant_set else 0 for doc in retrieved_docs[:k]]
        
        # Compute DCG
        dcg_score = dcg(relevances, k)
        
        # Compute IDCG (ideal DCG)
        ideal_relevances = [1] * min(len(relevant_docs), k)
        idcg_score = dcg(ideal_relevances, k)
        
        return dcg_score / idcg_score if idcg_score > 0 else 0.0


class GroundingMetrics:
    """
    Grounding quality metrics (Significance Level: 4-5)
    Most important for RAG systems - measures factual accuracy and citation quality
    """
    
    @staticmethod
    def grounding_accuracy(verified_claims: List[Dict[str, Any]]) -> float:
        """
        Grounding Accuracy: fraction of claims that are supported by retrieved passages
        """
        if not verified_claims:
            return 0.0
        
        supported_count = sum(1 for claim in verified_claims if claim.get('supported', False))
        return supported_count / len(verified_claims)
    
    @staticmethod
    def unsupported_claim_rate(verified_claims: List[Dict[str, Any]]) -> float:
        """
        Unsupported Claim Rate: fraction of claims that lack support (lower is better)
        """
        return 1.0 - GroundingMetrics.grounding_accuracy(verified_claims)
    
    @staticmethod
    def mean_support_score(verified_claims: List[Dict[str, Any]]) -> float:
        """
        Mean Support Score: average confidence score for claim support
        """
        if not verified_claims:
            return 0.0
        
        scores = [claim.get('support_score', 0.0) for claim in verified_claims 
                 if claim.get('support_score') is not None]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def citation_coverage(generated_answer: str, citations: List[Dict[str, str]]) -> float:
        """
        Citation Coverage: fraction of factual statements that have citations
        Simple heuristic based on sentence count vs citation count
        """
        # Count sentences (simple heuristic)
        sentences = [s.strip() for s in re.split(r'[.!?]+', generated_answer) if s.strip()]
        
        # Filter out very short sentences (likely not factual claims)
        factual_sentences = [s for s in sentences if len(s.split()) > 5]
        
        if not factual_sentences:
            return 1.0  # No claims to cite
        
        citation_count = len(citations) if citations else 0
        return min(1.0, citation_count / len(factual_sentences))


class CompositeMetrics:
    """
    Composite evaluation metrics (Significance Level: 5)
    Most important for papers - combines multiple aspects into unified score
    """
    
    @staticmethod
    def ragita_score(recall_at_5: float, grounding_accuracy: float, 
                     recall_weight: float = 0.4, grounding_weight: float = 0.6) -> float:
        """
        RAGita Score: Composite metric combining retrieval and grounding quality
        Default weights: 40% retrieval, 60% grounding (grounding is more critical)
        """
        return recall_weight * recall_at_5 + grounding_weight * grounding_accuracy
    
    @staticmethod
    def comprehensive_score(metrics_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation with multiple composite scores
        """
        # Extract key metrics
        recall_5 = metrics_dict.get('recall@5', 0.0)
        precision_5 = metrics_dict.get('precision@5', 0.0)
        grounding_acc = metrics_dict.get('grounding_accuracy', 0.0)
        mean_support = metrics_dict.get('mean_support_score', 0.0)
        citation_cov = metrics_dict.get('citation_coverage', 0.0)
        
        # Compute F1 for retrieval
        retrieval_f1 = 0.0
        if recall_5 + precision_5 > 0:
            retrieval_f1 = 2 * (recall_5 * precision_5) / (recall_5 + precision_5)
        
        # Compute grounding quality score
        grounding_quality = (grounding_acc + mean_support + citation_cov) / 3
        
        return {
            'ragita_score': CompositeMetrics.ragita_score(recall_5, grounding_acc),
            'retrieval_f1': retrieval_f1,
            'grounding_quality': grounding_quality,
            'overall_score': (retrieval_f1 + grounding_quality) / 2
        }