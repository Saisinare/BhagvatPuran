# RAGita Evaluation Module
"""
Comprehensive evaluation framework for RAGita system.

Metrics implemented (in ascending order of significance):
1. Basic Answer Quality: BLEU, ROUGE-L
2. Semantic Answer Quality: BERTScore  
3. Retrieval Effectiveness: Recall@k, Precision@k, MRR, nDCG@k
4. Grounding Quality: Grounding Accuracy, Unsupported Claim Rate, Mean Support Score, Citation Coverage
5. Composite Score: RAGita Score = 0.4*Recall@5 + 0.6*GroundingAccuracy

Usage:
    from evaluation.evaluator import RAGitaEvaluator
    evaluator = RAGitaEvaluator()
    results = evaluator.evaluate_system(test_queries, ground_truth_data)
"""

from .evaluator import RAGitaEvaluator
from .metrics import (
    RetrievalMetrics,
    GroundingMetrics, 
    AnswerQualityMetrics,
    CompositeMetrics
)

__all__ = [
    'RAGitaEvaluator',
    'RetrievalMetrics',
    'GroundingMetrics',
    'AnswerQualityMetrics', 
    'CompositeMetrics'
]