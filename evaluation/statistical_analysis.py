"""
Enhanced statistical analysis for RAGita evaluation results.
Provides confidence intervals, significance testing, and advanced metrics.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
import math

class StatisticalAnalyzer:
    """Advanced statistical analysis for evaluation results."""
    
    @staticmethod
    def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for a dataset.
        Returns (mean, lower_bound, upper_bound)
        """
        if not data:
            return 0.0, 0.0, 0.0
        
        n = len(data)
        mean = sum(data) / n
        
        if n == 1:
            return mean, mean, mean
        
        # Calculate standard error
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))
        std_error = std_dev / math.sqrt(n)
        
        # Use t-distribution for small samples
        if n < 30:
            # Approximation of t-critical value for common confidence levels
            t_critical = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
            if n < 15:
                t_critical *= 1.2  # Conservative adjustment for very small samples
        else:
            t_critical = 1.96  # Normal distribution
        
        margin_error = t_critical * std_error
        return mean, mean - margin_error, mean + margin_error
    
    @staticmethod
    def effect_size(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size between two groups."""
        if not group1 or not group2:
            return 0.0
        
        mean1 = sum(group1) / len(group1)
        mean2 = sum(group2) / len(group2)
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1) if n2 > 1 else 0
        
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def performance_consistency(scores: List[float]) -> Dict[str, float]:
        """Analyze performance consistency across queries."""
        if not scores:
            return {}
        
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        # Coefficient of variation (lower = more consistent)
        cv = std_dev / mean_score if mean_score > 0 else float('inf')
        
        return {
            'mean': mean_score,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'min_score': min(scores),
            'max_score': max(scores),
            'range': max(scores) - min(scores)
        }
    
    @staticmethod
    def categorize_performance(score: float, metric_type: str = 'accuracy') -> str:
        """Categorize performance level based on score and metric type."""
        if metric_type in ['accuracy', 'precision', 'recall', 'f1']:
            if score >= 0.90: return "Excellent"
            elif score >= 0.80: return "Good" 
            elif score >= 0.70: return "Satisfactory"
            elif score >= 0.60: return "Needs Improvement"
            else: return "Poor"
        elif metric_type == 'response_time':
            if score <= 3.0: return "Excellent"
            elif score <= 5.0: return "Good"
            elif score <= 8.0: return "Satisfactory"
            elif score <= 12.0: return "Needs Improvement"
            else: return "Poor"
        else:
            return "Unknown"

class EnhancedEvaluationAnalyzer:
    """Enhanced analyzer for comprehensive evaluation results."""
    
    def __init__(self):
        self.stat_analyzer = StatisticalAnalyzer()
    
    def analyze_comprehensive_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of evaluation results."""
        individual_results = results.get('individual_results', [])
        
        if len(individual_results) < 5:
            return {"error": "Insufficient data for statistical analysis (minimum 5 queries required)"}
        
        analysis = {
            'sample_size': len(individual_results),
            'statistical_power': self._assess_statistical_power(len(individual_results)),
            'metrics_analysis': {},
            'category_analysis': {},
            'reliability_analysis': {},
            'performance_distribution': {}
        }
        
        # Analyze key metrics
        key_metrics = [
            'grounding_accuracy', 'citation_coverage', 'mean_support_score',
            'bert_f1', 'contextual_relevance', 'response_time'
        ]
        
        for metric in key_metrics:
            scores = [r.get(metric, 0) for r in individual_results if metric in r]
            if scores:
                mean, lower_ci, upper_ci = self.stat_analyzer.confidence_interval(scores)
                consistency = self.stat_analyzer.performance_consistency(scores)
                
                analysis['metrics_analysis'][metric] = {
                    'mean': mean,
                    'confidence_interval_95': (lower_ci, upper_ci),
                    'consistency': consistency,
                    'performance_level': self.stat_analyzer.categorize_performance(mean, 
                                        'response_time' if 'time' in metric else 'accuracy')
                }
        
        # Category-based analysis (if category info available)
        analysis['category_analysis'] = self._analyze_by_category(individual_results, results.get('test_data', []))
        
        # Reliability analysis
        analysis['reliability_analysis'] = self._assess_reliability(individual_results)
        
        return analysis
    
    def _assess_statistical_power(self, n: int) -> str:
        """Assess statistical power based on sample size."""
        if n >= 30: return "High"
        elif n >= 20: return "Moderate" 
        elif n >= 10: return "Low"
        else: return "Very Low"
    
    def _analyze_by_category(self, individual_results: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by query category if available."""
        # Create mapping from query to category
        query_to_category = {}
        for item in test_data:
            query_to_category[item.get('query', '')] = item.get('category', 'unknown')
        
        category_performance = {}
        for result in individual_results:
            query = result.get('query', '')
            category = query_to_category.get(query, 'unknown')
            
            if category not in category_performance:
                category_performance[category] = {
                    'grounding_accuracy': [],
                    'citation_coverage': [],
                    'response_time': []
                }
            
            for metric in ['grounding_accuracy', 'citation_coverage', 'response_time']:
                if metric in result:
                    category_performance[category][metric].append(result[metric])
        
        # Calculate statistics for each category
        category_stats = {}
        for category, metrics in category_performance.items():
            if category != 'unknown' and any(metrics.values()):
                category_stats[category] = {}
                for metric, scores in metrics.items():
                    if scores:
                        mean, lower_ci, upper_ci = self.stat_analyzer.confidence_interval(scores)
                        category_stats[category][metric] = {
                            'mean': mean,
                            'confidence_interval': (lower_ci, upper_ci),
                            'n': len(scores)
                        }
        
        return category_stats
    
    def _assess_reliability(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Assess system reliability and consistency."""
        # Calculate failure rates
        total_queries = len(individual_results)
        successful_queries = len([r for r in individual_results if 'error' not in r])
        
        # Performance consistency
        grounding_scores = [r.get('grounding_accuracy', 0) for r in individual_results 
                          if 'grounding_accuracy' in r]
        
        reliability = {
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'failure_rate': (total_queries - successful_queries) / total_queries if total_queries > 0 else 0,
            'performance_consistency': self.stat_analyzer.performance_consistency(grounding_scores),
        }
        
        # Response time consistency
        response_times = [r.get('response_time', 0) for r in individual_results 
                         if 'response_time' in r]
        if response_times:
            reliability['response_time_consistency'] = self.stat_analyzer.performance_consistency(response_times)
        
        return reliability
    
    def generate_statistical_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted statistical report."""
        if 'error' in analysis:
            return f"Statistical Analysis Error: {analysis['error']}"
        
        report = f"""
## Statistical Analysis Report

### Sample Characteristics
- **Sample Size**: {analysis['sample_size']} queries
- **Statistical Power**: {analysis['statistical_power']}

### Key Metrics (95% Confidence Intervals)
"""
        
        for metric, stats in analysis['metrics_analysis'].items():
            mean = stats['mean']
            ci_lower, ci_upper = stats['confidence_interval_95']
            level = stats['performance_level']
            cv = stats['consistency']['coefficient_of_variation']
            
            report += f"- **{metric.replace('_', ' ').title()}**: {mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] - {level}\n"
            report += f"  - Consistency (CV): {cv:.3f} {'(High variance)' if cv > 0.3 else '(Stable)'}\n"
        
        # Category analysis
        if analysis['category_analysis']:
            report += f"\n### Performance by Category\n"
            for category, metrics in analysis['category_analysis'].items():
                report += f"**{category.capitalize()}**:\n"
                for metric, stats in metrics.items():
                    if 'mean' in stats:
                        mean = stats['mean']
                        ci_lower, ci_upper = stats['confidence_interval']
                        n = stats['n']
                        report += f"  - {metric}: {mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] (n={n})\n"
        
        # Reliability assessment
        reliability = analysis['reliability_analysis']
        report += f"""
### System Reliability
- **Success Rate**: {reliability['success_rate']:.1%}
- **Performance Consistency**: CV = {reliability['performance_consistency']['coefficient_of_variation']:.3f}
- **Response Time Stability**: {reliability.get('response_time_consistency', {}).get('coefficient_of_variation', 0):.3f}
"""
        
        return report