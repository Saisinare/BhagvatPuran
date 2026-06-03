"""
Main RAGita Evaluator - orchestrates comprehensive evaluation of the RAG system.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from RAG.pipeline.orchestrator import answer_query
from .metrics import RetrievalMetrics, GroundingMetrics, AnswerQualityMetrics, CompositeMetrics
from .semantic_metrics import SemanticMetrics
from .statistical_analysis import EnhancedEvaluationAnalyzer

class RAGitaEvaluator:
    """
    Comprehensive evaluator for RAGita system.
    Evaluates retrieval, grounding, answer quality, and composite metrics.
    """
    
    def __init__(self):
        self.results_dir = project_root / "RAG" / "data" / "eval"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_metrics = SemanticMetrics()
        self.stat_analyzer = EnhancedEvaluationAnalyzer()
    
    def evaluate_query(self, query: str, ground_truth: Dict[str, Any], 
                      persona: str = "versatile") -> Dict[str, Any]:
        """
        Evaluate a single query against ground truth data.
        
        Args:
            query: The input question
            ground_truth: Dict containing:
                - relevant_docs: List[str] - relevant passage IDs (chapter:verse)
                - gold_answer: str (optional) - reference answer
                - expected_citations: List[str] (optional) - expected citations
            persona: Response style
        
        Returns:
            Dict with all computed metrics
        """
        # Get RAGita response
        start_time = time.time()
        result = answer_query(query, persona=persona)
        response_time = time.time() - start_time
        
        # Extract components
        retrieved_docs = [f"{c.get('chapter')}:{c.get('verse')}" 
                         for c in result.get('candidates', [])]
        generated_answer = result.get('answer_raw', '')
        verification = result.get('verification', [])
        citations = result.get('citations', [])
        
        # Ground truth data
        relevant_docs = ground_truth.get('relevant_docs', [])
        gold_answer = ground_truth.get('gold_answer', '')
        
        metrics = {
            'query': query,
            'persona': persona,
            'response_time': response_time,
        }
        
        # 1. Retrieval Metrics (if relevant docs provided)
        if relevant_docs:
            metrics.update({
                'recall@1': RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, 1),
                'recall@3': RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, 3),
                'recall@5': RetrievalMetrics.recall_at_k(retrieved_docs, relevant_docs, 5),
                'precision@1': RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, 1),
                'precision@3': RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, 3),
                'precision@5': RetrievalMetrics.precision_at_k(retrieved_docs, relevant_docs, 5),
                'mrr': RetrievalMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs),
                'ndcg@1': RetrievalMetrics.ndcg_at_k(retrieved_docs, relevant_docs, 1),
                'ndcg@3': RetrievalMetrics.ndcg_at_k(retrieved_docs, relevant_docs, 3),
                'ndcg@5': RetrievalMetrics.ndcg_at_k(retrieved_docs, relevant_docs, 5),
            })
        
        # 2. Grounding Metrics
        metrics.update({
            'grounding_accuracy': GroundingMetrics.grounding_accuracy(verification),
            'unsupported_claim_rate': GroundingMetrics.unsupported_claim_rate(verification),
            'mean_support_score': GroundingMetrics.mean_support_score(verification),
            'citation_coverage': GroundingMetrics.citation_coverage(generated_answer, citations),
            'total_claims': len(verification),
            'supported_claims': sum(1 for v in verification if v.get('supported', False)),
        })
        
        # 3. Answer Quality Metrics (if gold answer provided)
        if gold_answer:
            # Traditional metrics
            metrics.update({
                'bleu_score': AnswerQualityMetrics.bleu_score(gold_answer, generated_answer),
                'rouge_l': AnswerQualityMetrics.rouge_l(gold_answer, generated_answer),
            })
            
            # Semantic metrics (more sophisticated)
            bert_precision, bert_recall, bert_f1 = self.semantic_metrics.bert_score_simple(gold_answer, generated_answer)
            metrics.update({
                'bert_precision': bert_precision,
                'bert_recall': bert_recall, 
                'bert_f1': bert_f1,
                'semantic_similarity': self.semantic_metrics.semantic_similarity(gold_answer, generated_answer),
            })
        
        # Query-Answer relevance (always available)
        metrics['contextual_relevance'] = self.semantic_metrics.contextual_relevance(query, generated_answer)
        
        # Factual consistency with retrieved passages
        passage_texts = [c.get('english', '') for c in result.get('candidates', [])]
        metrics['factual_consistency'] = self.semantic_metrics.factual_consistency(passage_texts, generated_answer)
        
        # 4. Composite Metrics
        composite = CompositeMetrics.comprehensive_score(metrics)
        metrics.update(composite)
        
        # Add raw outputs for analysis
        metrics.update({
            'generated_answer': generated_answer,
            'retrieved_docs': retrieved_docs,
            'verification_details': verification,
            'citations': citations,
        })
        
        return metrics
    
    def evaluate_dataset(self, test_data: List[Dict[str, Any]], 
                        persona: str = "versatile") -> Dict[str, Any]:
        """
        Evaluate RAGita on a dataset of queries.
        
        Args:
            test_data: List of dicts, each containing:
                - query: str
                - ground_truth: Dict (see evaluate_query for format)
            persona: Response style
        
        Returns:
            Comprehensive evaluation results
        """
        individual_results = []
        
        print(f"🔍 Evaluating RAGita on {len(test_data)} queries...")
        
        for i, item in enumerate(test_data, 1):
            query = item['query']
            ground_truth = item.get('ground_truth', {})
            
            print(f"  [{i:2d}/{len(test_data)}] Evaluating: {query[:60]}...")
            
            try:
                result = self.evaluate_query(query, ground_truth, persona)
                individual_results.append(result)
            except Exception as e:
                print(f"    ❌ Error evaluating query {i}: {e}")
                individual_results.append({
                    'query': query,
                    'error': str(e),
                    'persona': persona
                })
        
        # Aggregate metrics
        successful_results = [r for r in individual_results if 'error' not in r]
        
        if not successful_results:
            return {
                'error': 'No successful evaluations',
                'individual_results': individual_results,
                'evaluation_time': datetime.now().isoformat(),
            }
        
        # Compute aggregated metrics
        aggregated = self._aggregate_metrics(successful_results)
        
        # Create comprehensive results
        evaluation_results = {
            'evaluation_summary': {
                'total_queries': len(test_data),
                'successful_evaluations': len(successful_results),
                'failed_evaluations': len(test_data) - len(successful_results),
                'persona_used': persona,
                'evaluation_time': datetime.now().isoformat(),
            },
            'aggregated_metrics': aggregated,
            'individual_results': individual_results,
            'test_data': test_data,  # Include for category analysis
        }
        
        # Add statistical analysis for larger datasets
        if len(successful_results) >= 5:
            try:
                statistical_analysis = self.stat_analyzer.analyze_comprehensive_results(evaluation_results)
                evaluation_results['statistical_analysis'] = statistical_analysis
            except Exception as e:
                evaluation_results['statistical_analysis_error'] = str(e)
        
        return evaluation_results
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregated metrics across all results."""
        numeric_metrics = {}
        
        # Collect all numeric metrics
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['response_time']:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Compute averages
        aggregated = {}
        for key, values in numeric_metrics.items():
            if values:  # Only compute if we have values
                aggregated[f'avg_{key}'] = sum(values) / len(values)
                aggregated[f'min_{key}'] = min(values)
                aggregated[f'max_{key}'] = max(values)
        
        # Compute overall performance metrics
        if 'recall@5' in numeric_metrics and 'grounding_accuracy' in numeric_metrics:
            recall_5_avg = sum(numeric_metrics['recall@5']) / len(numeric_metrics['recall@5'])
            grounding_avg = sum(numeric_metrics['grounding_accuracy']) / len(numeric_metrics['grounding_accuracy'])
            aggregated['overall_ragita_score'] = CompositeMetrics.ragita_score(recall_5_avg, grounding_avg)
        
        return aggregated
    
    def print_results_table(self, evaluation_results: Dict[str, Any]) -> None:
        """Print evaluation results as a formatted markdown table."""
        aggregated = evaluation_results.get('aggregated_metrics', {})
        summary = evaluation_results.get('evaluation_summary', {})
        
        print("\n" + "="*80)
        print("🏆 RAGita Evaluation Results")
        print("="*80)
        
        # Summary table
        print(f"\n📊 **Evaluation Summary**")
        print(f"- Total Queries: {summary.get('total_queries', 0)}")
        print(f"- Successful: {summary.get('successful_evaluations', 0)}")
        print(f"- Failed: {summary.get('failed_evaluations', 0)}")
        print(f"- Persona: {summary.get('persona_used', 'N/A')}")
        
        # Main metrics table
        print(f"\n📈 **Core Metrics** (in ascending order of significance)")
        print("| Metric Category | Metric | Value | Significance |")
        print("|----------------|---------|-------|--------------|")
        
        # Level 1-2: Basic Answer Quality
        if 'avg_bleu_score' in aggregated:
            print(f"| Answer Quality | BLEU Score | {aggregated['avg_bleu_score']:.3f} | ⭐ |")
        if 'avg_rouge_l' in aggregated:
            print(f"| Answer Quality | ROUGE-L | {aggregated['avg_rouge_l']:.3f} | ⭐ |")
        
        # Level 2-3: Semantic Answer Quality  
        if 'avg_bert_f1' in aggregated:
            print(f"| Answer Quality | BERTScore F1 | {aggregated['avg_bert_f1']:.3f} | ⭐⭐ |")
        if 'avg_semantic_similarity' in aggregated:
            print(f"| Answer Quality | Semantic Similarity | {aggregated['avg_semantic_similarity']:.3f} | ⭐⭐ |")
        if 'avg_contextual_relevance' in aggregated:
            print(f"| Answer Quality | Contextual Relevance | {aggregated['avg_contextual_relevance']:.3f} | ⭐⭐ |")
        if 'avg_factual_consistency' in aggregated:
            print(f"| Answer Quality | Factual Consistency | {aggregated['avg_factual_consistency']:.3f} | ⭐⭐ |")
        
        # Level 3-4: Retrieval Effectiveness  
        if 'avg_recall@5' in aggregated:
            print(f"| Retrieval | Recall@5 | {aggregated['avg_recall@5']:.3f} | ⭐⭐⭐ |")
        if 'avg_precision@5' in aggregated:
            print(f"| Retrieval | Precision@5 | {aggregated['avg_precision@5']:.3f} | ⭐⭐⭐ |")
        if 'avg_mrr' in aggregated:
            print(f"| Retrieval | MRR | {aggregated['avg_mrr']:.3f} | ⭐⭐⭐ |")
        if 'avg_ndcg@5' in aggregated:
            print(f"| Retrieval | nDCG@5 | {aggregated['avg_ndcg@5']:.3f} | ⭐⭐⭐ |")
        
        # Level 4-5: Grounding Quality (Most Important)
        if 'avg_grounding_accuracy' in aggregated:
            print(f"| Grounding | Accuracy | {aggregated['avg_grounding_accuracy']:.3f} | ⭐⭐⭐⭐⭐ |")
        if 'avg_unsupported_claim_rate' in aggregated:
            print(f"| Grounding | Unsupported Rate | {aggregated['avg_unsupported_claim_rate']:.3f} | ⭐⭐⭐⭐ |")
        if 'avg_mean_support_score' in aggregated:
            print(f"| Grounding | Mean Support | {aggregated['avg_mean_support_score']:.3f} | ⭐⭐⭐⭐ |")
        if 'avg_citation_coverage' in aggregated:
            print(f"| Grounding | Citation Coverage | {aggregated['avg_citation_coverage']:.3f} | ⭐⭐⭐⭐ |")
        
        # Level 5: Composite Scores (Most Important for Papers)
        if 'overall_ragita_score' in aggregated:
            print(f"| **Composite** | **RAGita Score** | **{aggregated['overall_ragita_score']:.3f}** | **⭐⭐⭐⭐⭐** |")
        if 'avg_overall_score' in aggregated:
            print(f"| Composite | Overall Score | {aggregated['avg_overall_score']:.3f} | ⭐⭐⭐⭐⭐ |")
        
        print(f"\n💡 **Significance Levels:**")
        print(f"- ⭐: Basic metrics (BLEU, ROUGE) - traditional but less critical")
        print(f"- ⭐⭐⭐: Retrieval metrics - important for RAG systems")
        print(f"- ⭐⭐⭐⭐: Grounding metrics - critical for factual accuracy")
        print(f"- ⭐⭐⭐⭐⭐: Composite metrics - **most important for papers/resume**")
        
    def save_results(self, evaluation_results: Dict[str, Any], 
                    filename: str = "results_summary.json") -> Path:
        """Save evaluation results to JSON file."""
        filepath = self.results_dir / filename
        
        # Make results JSON serializable
        clean_results = self._clean_for_json(evaluation_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved evaluation results to: {filepath}")
        return filepath
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)