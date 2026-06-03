#!/usr/bin/env python3
"""
RAGita Evaluation Runner

Comprehensive evaluation of the RAGita system across multiple metrics.
Implements metrics in ascending order of significance for academic papers.

Usage:
    python run_evaluation.py --dataset full    # Full evaluation
    python run_evaluation.py --dataset minimal # Quick test
    python run_evaluation.py --custom-query "Your question here"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluator import RAGitaEvaluator
from evaluation.test_data import get_full_test_dataset, get_minimal_test_dataset, SAMPLE_TEST_DATA
from evaluation.comprehensive_test_data import (
    get_comprehensive_test_dataset, 
    get_balanced_evaluation_dataset,
    get_evaluation_statistics,
    get_category_specific_dataset
)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on the full dataset (25+ queries)."""
    print("🚀 Starting Comprehensive RAGita Evaluation")
    print("="*60)
    
    # Get dataset statistics
    stats = get_evaluation_statistics()
    print(f"📊 Dataset Overview:")
    print(f"   • Total Queries: {stats['total_queries']}")
    print(f"   • Categories: {', '.join(stats['categories'].keys())}")
    print(f"   • Avg Relevant Docs per Query: {stats['avg_relevant_docs']:.1f}")
    
    evaluator = RAGitaEvaluator()
    test_data = get_comprehensive_test_dataset()
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, persona="versatile")
    
    # Display results
    evaluator.print_results_table(results)
    
    # Additional analysis by category
    print(f"\n📈 Performance by Category:")
    print("-" * 40)
    
    categories = ['factual', 'philosophical', 'practical', 'spiritual', 'ethical']
    for category in categories:
        category_results = [r for r in results['individual_results'] 
                          if any(item['category'] == category 
                                for item in test_data 
                                if item['query'] == r.get('query', ''))]
        
        if category_results:
            avg_grounding = sum(r.get('grounding_accuracy', 0) for r in category_results) / len(category_results)
            avg_citation = sum(r.get('citation_coverage', 0) for r in category_results) / len(category_results)
            print(f"{category.capitalize():>12}: {avg_grounding:.2f} grounding, {avg_citation:.2f} citation ({len(category_results)} queries)")
    
    # Save results
    timestamp = results['evaluation_summary']['evaluation_time'][:19].replace(':', '-')
    filename = f"comprehensive_evaluation_{timestamp}.json"
    evaluator.save_results(results, filename)
    
    return results

def run_balanced_evaluation():
    """Run balanced evaluation across all categories."""
    print("⚖️ Starting Balanced Category Evaluation")
    print("="*50)
    
    evaluator = RAGitaEvaluator()
    test_data = get_balanced_evaluation_dataset(queries_per_category=3)
    
    print(f"📊 Testing {len(test_data)} queries across balanced categories")
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, persona="versatile")
    
    # Display results
    evaluator.print_results_table(results)
    
    # Save results
    evaluator.save_results(results, "balanced_evaluation.json")
    
    return results

def run_full_evaluation():
    """Legacy function - redirects to comprehensive evaluation."""
    return run_comprehensive_evaluation()

def run_minimal_evaluation():
    """Run quick evaluation on minimal dataset."""
    print("⚡ Starting Minimal RAGita Evaluation")
    print("="*40)
    
    evaluator = RAGitaEvaluator()
    test_data = get_minimal_test_dataset()
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data, persona="versatile")
    
    # Display results
    evaluator.print_results_table(results)
    
    # Save results
    evaluator.save_results(results, "minimal_evaluation.json")
    
    return results

def evaluate_custom_query(query: str):
    """Evaluate a single custom query."""
    print(f"🔍 Evaluating Custom Query: {query}")
    print("="*50)
    
    evaluator = RAGitaEvaluator()
    
    # Create dummy ground truth (no reference data available)
    ground_truth = {}
    
    # Run evaluation
    result = evaluator.evaluate_query(query, ground_truth)
    
    # Display key results
    print(f"\n📊 **Query:** {query}")
    print(f"**Answer:** {result.get('generated_answer', 'N/A')[:200]}...")
    print(f"\n**Key Metrics:**")
    print(f"- Grounding Accuracy: {result.get('grounding_accuracy', 0):.3f}")
    print(f"- Mean Support Score: {result.get('mean_support_score', 0):.3f}")
    print(f"- Citation Coverage: {result.get('citation_coverage', 0):.3f}")
    print(f"- Total Claims: {result.get('total_claims', 0)}")
    print(f"- Supported Claims: {result.get('supported_claims', 0)}")
    print(f"- Response Time: {result.get('response_time', 0):.2f}s")
    
    # Save individual result
    evaluator.save_results({
        'evaluation_summary': {'single_query': True, 'query': query},
        'individual_results': [result]
    }, f"custom_query_evaluation.json")
    
    return result

def demo_evaluation_system():
    """Run a demonstration of the evaluation system with sample queries."""
    print("🎯 RAGita Evaluation System Demo")
    print("="*40)
    
    # Import improved test data
    try:
        from evaluation.improved_test_data import get_improved_test_dataset
        demo_data = get_improved_test_dataset()
    except ImportError:
        # Fallback to manual demo queries
        demo_data = [
            {
                "query": "What is the Bhagavad Gita about?",
                "ground_truth": {"relevant_docs": ["9:1", "9:2", "13:2"]}
            },
            {
                "query": "How can I find peace in difficult times?", 
                "ground_truth": {"relevant_docs": ["2:71", "6:15", "2:57"]}
            },
            {
                "query": "What does Krishna say about duty?",
                "ground_truth": {"relevant_docs": ["6:1", "3:8", "18:47"]}
            }
        ]
    
    evaluator = RAGitaEvaluator()
    results = []
    
    for i, item in enumerate(demo_data, 1):
        query = item['query']
        ground_truth = item.get('ground_truth', {})
        
        print(f"\n[{i}/{len(demo_data)}] Evaluating: {query}")
        
        result = evaluator.evaluate_query(query, ground_truth)
        results.append(result)
        
        # Show quick summary
        print(f"   ✅ Grounding Accuracy: {result.get('grounding_accuracy', 0):.3f}")
        print(f"   📊 Retrieval Recall@5: {result.get('recall@5', 0):.3f}")
        print(f"   📝 Claims Supported: {result.get('supported_claims', 0)}/{result.get('total_claims', 0)}")
        print(f"   🎯 RAGita Score: {result.get('ragita_score', 0):.3f}")
    
    # Create summary
    demo_results = {
        'evaluation_summary': {
            'demo_evaluation': True,
            'total_queries': len(demo_data)
        },
        'individual_results': results
    }
    
    print(f"\n🏆 Demo Evaluation Complete!")
    evaluator.save_results(demo_results, "demo_evaluation.json")
    
    return demo_results

def main():
    parser = argparse.ArgumentParser(
        description="Run RAGita evaluation with comprehensive metrics")
    
    parser.add_argument("--dataset", "-d", choices=["comprehensive", "balanced", "full", "minimal", "demo"], 
                       default="demo", help="Dataset to evaluate on")
    parser.add_argument("--custom-query", "-q", type=str, 
                       help="Evaluate a single custom query")
    parser.add_argument("--persona", "-p", type=str, default="versatile",
                       choices=["versatile", "scholarly", "devotional", "practical"],
                       help="Persona to use for evaluation")
    
    args = parser.parse_args()
    
    try:
        if args.custom_query:
            evaluate_custom_query(args.custom_query)
        elif args.dataset == "comprehensive":
            run_comprehensive_evaluation()
        elif args.dataset == "balanced":
            run_balanced_evaluation()
        elif args.dataset == "full":
            run_full_evaluation()  # Redirects to comprehensive
        elif args.dataset == "minimal":
            run_minimal_evaluation()
        elif args.dataset == "demo":
            demo_evaluation_system()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved in: {project_root}/RAG/data/eval/")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()