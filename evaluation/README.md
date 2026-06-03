# RAGita Evaluation Framework

A comprehensive evaluation framework for the RAGita system, implementing metrics in **ascending order of significance** for academic papers and resumes.

## 🏆 Metrics Overview (By Significance)

### ⭐ Level 1-2: Basic Answer Quality
- **BLEU Score**: Traditional n-gram overlap metric
- **ROUGE-L**: Longest common subsequence metric
- *Significance*: Standard but less critical for modern RAG systems

### ⭐⭐ Level 2-3: Semantic Answer Quality  
- **BERTScore F1**: Transformer-based semantic similarity
- **Semantic Similarity**: Sentence embedding cosine similarity
- **Contextual Relevance**: Query-answer semantic alignment
- **Factual Consistency**: Answer consistency with source passages
- *Significance*: More sophisticated than traditional metrics

### ⭐⭐⭐ Level 3-4: Retrieval Effectiveness
- **Recall@k**: Fraction of relevant documents retrieved
- **Precision@k**: Fraction of retrieved documents that are relevant  
- **MRR**: Mean Reciprocal Rank of first relevant document
- **nDCG@k**: Normalized Discounted Cumulative Gain
- *Significance*: Critical for RAG system performance

### ⭐⭐⭐⭐ Level 4-5: Grounding Quality
- **Grounding Accuracy**: Fraction of claims supported by evidence
- **Unsupported Claim Rate**: Fraction of unsupported claims (lower is better)
- **Mean Support Score**: Average confidence of claim support
- **Citation Coverage**: Fraction of factual statements with citations
- *Significance*: **Most critical** for factual accuracy and trustworthiness

### ⭐⭐⭐⭐⭐ Level 5: Composite Metrics
- **RAGita Score**: `0.4×Recall@5 + 0.6×Grounding_Accuracy`
- **Overall Score**: Combined retrieval and grounding quality
- *Significance*: **Most important for papers/resume** - unified system performance

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Core dependencies (required)
pip install pandas numpy

# Semantic metrics (recommended)
pip install sentence-transformers torch transformers

# Or install all evaluation requirements
pip install -r evaluation/requirements.txt
```

### 2. Run Evaluation

```bash
# Demo evaluation (3 sample queries)
python run_evaluation.py --dataset demo

# Minimal evaluation (5 queries with ground truth)
python run_evaluation.py --dataset minimal

# Full evaluation (comprehensive test suite)
python run_evaluation.py --dataset full

# Custom single query
python run_evaluation.py --custom-query "How can I find inner peace?"
```

## 📊 Sample Output

```
🏆 RAGita Evaluation Results
================================================================================

📊 Evaluation Summary
- Total Queries: 5
- Successful: 5
- Failed: 0
- Persona: versatile

📈 Core Metrics (in ascending order of significance)
| Metric Category | Metric | Value | Significance |
|----------------|---------|-------|--------------|
| Answer Quality | BLEU Score | 0.234 | ⭐ |
| Answer Quality | ROUGE-L | 0.456 | ⭐ |
| Answer Quality | BERTScore F1 | 0.678 | ⭐⭐ |
| Answer Quality | Semantic Similarity | 0.723 | ⭐⭐ |
| Retrieval | Recall@5 | 0.834 | ⭐⭐⭐ |
| Retrieval | Precision@5 | 0.756 | ⭐⭐⭐ |
| Retrieval | MRR | 0.689 | ⭐⭐⭐ |
| Grounding | Accuracy | 0.912 | ⭐⭐⭐⭐⭐ |
| Grounding | Mean Support | 0.845 | ⭐⭐⭐⭐ |
| Grounding | Citation Coverage | 0.923 | ⭐⭐⭐⭐ |
| **Composite** | **RAGita Score** | **0.881** | **⭐⭐⭐⭐⭐** |
```

## 🔧 Programmatic Usage

```python
from evaluation.evaluator import RAGitaEvaluator
from evaluation.test_data import get_minimal_test_dataset

# Initialize evaluator
evaluator = RAGitaEvaluator()

# Evaluate single query
ground_truth = {
    "relevant_docs": ["2:47", "3:19"],
    "gold_answer": "Krishna teaches about selfless action..."
}
result = evaluator.evaluate_query("What is karma yoga?", ground_truth)

# Evaluate dataset
test_data = get_minimal_test_dataset()
results = evaluator.evaluate_dataset(test_data)

# Print results and save
evaluator.print_results_table(results)
evaluator.save_results(results, "my_evaluation.json")
```

## 📁 Output Files

All evaluation results are saved to `RAG/data/eval/`:

- `results_summary.json`: Complete evaluation data
- `demo_evaluation.json`: Demo run results  
- `minimal_evaluation.json`: Quick test results
- `full_evaluation_YYYY-MM-DD.json`: Timestamped full evaluations

## 🎯 For Academic Papers

**Most Important Metrics to Report:**

1. **RAGita Score** (0.881) - Primary composite metric
2. **Grounding Accuracy** (0.912) - Factual correctness  
3. **Recall@5** (0.834) - Retrieval effectiveness
4. **Citation Coverage** (0.923) - Citation quality

**Sample Academic Reporting:**
> "Our RAGita system achieved a composite score of 0.881, with 91.2% grounding accuracy and 83.4% recall@5 for relevant passage retrieval. The system demonstrated strong citation coverage at 92.3%, indicating reliable source attribution for factual claims."

## 🛠️ Extending the Framework

### Add New Metrics
```python
# In metrics.py
class CustomMetrics:
    @staticmethod 
    def my_metric(param1, param2):
        # Your implementation
        return score

# In evaluator.py
from .metrics import CustomMetrics

# Add to evaluate_query method:
metrics['my_metric'] = CustomMetrics.my_metric(data1, data2)
```

### Add New Test Data
```python
# In test_data.py
NEW_TEST_QUERIES = [
    {
        "query": "Your question here",
        "ground_truth": {
            "relevant_docs": ["chapter:verse", ...],
            "gold_answer": "Reference answer...",
        }
    }
]
```

## 📈 Significance Levels Explained

- **⭐ (1-2)**: Traditional metrics - good to have but less critical
- **⭐⭐ (2-3)**: Semantic metrics - better than traditional, moderate importance  
- **⭐⭐⭐ (3-4)**: Retrieval metrics - important for RAG systems
- **⭐⭐⭐⭐ (4-5)**: Grounding metrics - critical for trustworthiness
- **⭐⭐⭐⭐⭐ (5)**: Composite metrics - **most important for papers**

Focus on Level 4-5 metrics for academic publications and resume highlights.

## 🔍 Troubleshooting

**Semantic metrics not working?**
- Install sentence-transformers: `pip install sentence-transformers`
- System will fall back to word overlap if transformers unavailable

**Low grounding accuracy?**
- Check if your passages actually support the generated claims
- Verify the verifier threshold in orchestrator.py (default: 0.60)

**No relevant docs found?**
- Ensure ground truth data has correct chapter:verse format
- Check if retrieval system is finding the expected passages