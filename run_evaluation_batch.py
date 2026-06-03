#!/usr/bin/env python3
"""
Run evaluation in smaller batches to avoid rate limits.
Splits comprehensive dataset into manageable chunks.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from evaluation.comprehensive_test_data import COMPREHENSIVE_TEST_DATA
from evaluation.evaluator import RAGitaEvaluator

def run_batched_evaluation(batch_size=5, delay_between_batches=180):
    """
    Run evaluation in batches with delays to respect rate limits.
    
    Args:
        batch_size: Number of queries per batch (default 5)
        delay_between_batches: Seconds to wait between batches (default 180s = 3min)
    """
    print(f"📊 Running batched evaluation: {batch_size} queries per batch")
    print(f"⏱️  Waiting {delay_between_batches}s between batches to respect rate limits\n")
    
    evaluator = RAGitaEvaluator()
    all_results = []
    
    total_queries = len(COMPREHENSIVE_TEST_DATA)
    num_batches = (total_queries + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_queries)
        batch = COMPREHENSIVE_TEST_DATA[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"📦 Batch {batch_num + 1}/{num_batches}: Queries {start_idx + 1}-{end_idx}")
        print(f"{'='*60}\n")
        
        try:
            batch_results = evaluator.evaluate_dataset(batch)
            all_results.extend(batch_results.get('individual_results', []))
            
            print(f"✅ Batch {batch_num + 1} complete: {len(batch)} queries evaluated")
            
            # Wait before next batch (except for last batch)
            if batch_num < num_batches - 1:
                print(f"\n⏳ Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
                
        except Exception as e:
            print(f"❌ Error in batch {batch_num + 1}: {e}")
            print(f"⚠️  Partial results saved. Continuing with next batch after delay...")
            time.sleep(delay_between_batches)
            continue
    
    # Generate final combined report
    print(f"\n{'='*60}")
    print(f"🎉 All batches complete! Total queries: {len(all_results)}/{total_queries}")
    print(f"{'='*60}\n")
    
    # Recalculate aggregated metrics
    final_report = evaluator.evaluate_dataset(COMPREHENSIVE_TEST_DATA[:len(all_results)])
    
    print("📈 Final Aggregated Metrics:")
    print(f"  Grounding Accuracy: {final_report.get('aggregated_metrics', {}).get('avg_grounding_accuracy', 0):.3f}")
    print(f"  Recall@5: {final_report.get('aggregated_metrics', {}).get('avg_recall@5', 0):.3f}")
    print(f"  RAGita Score: {final_report.get('aggregated_metrics', {}).get('avg_ragita_score', 0):.3f}")
    
    return final_report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGita evaluation in batches")
    parser.add_argument("--batch-size", type=int, default=5, 
                       help="Queries per batch (default: 5)")
    parser.add_argument("--delay", type=int, default=180,
                       help="Seconds between batches (default: 180)")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║         RAGita Batched Evaluation Runner               ║
║  Avoids rate limits by splitting into smaller batches   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    run_batched_evaluation(batch_size=args.batch_size, delay_between_batches=args.delay)
