"""
Evaluation module for DocuMind.
Tests different chunking and retrieval strategies and reports metrics.
"""

import json
import time
import csv
from typing import List, Dict, Tuple
import os

from config import EVAL_QA_PATH, EVAL_TOP_K_VALUES, EVAL_DELAY_SECONDS
from retriever import HybridRetriever
from generator import AnswerGenerator


class Evaluator:
    """Evaluate RAG system performance on test Q&A pairs."""
    
    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = AnswerGenerator()
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> List[Dict]:
        """
        Load test Q&A pairs from JSON file.
        
        Returns:
            List of test dictionaries with 'question', 'answer', and optional 'source' keys
        """
        if not os.path.exists(EVAL_QA_PATH):
            print(f"Warning: Test data file not found at {EVAL_QA_PATH}")
            print("Please create eval_qa.json with test Q&A pairs.")
            return []
        
        with open(EVAL_QA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} test Q&A pairs from {EVAL_QA_PATH}")
        return data
    
    def evaluate_retrieval(self, question: str, expected_source: str = None, 
                          chunking_strategy: str = "fixed_size",
                          retrieval_strategy: str = "hybrid") -> Dict:
        """
        Evaluate retrieval for a single question.
        
        Args:
            question: Test question
            expected_source: Expected source document (optional, for recall calculation)
            chunking_strategy: Either "fixed_size" or "sentence_based"
            retrieval_strategy: Either "vector" or "hybrid"
            
        Returns:
            Dictionary with retrieval metrics
        """
        use_hybrid = (retrieval_strategy == "hybrid")
        
        # Retrieve results
        results = self.retriever.retrieve(
            question, 
            k=max(EVAL_TOP_K_VALUES),
            use_hybrid=use_hybrid
        )
        
        # Calculate recall@k for each k value
        recall_scores = {}
        for k in EVAL_TOP_K_VALUES:
            if expected_source:
                # Check if expected source is in top-k
                top_k_sources = [r['metadata']['source'] for r in results[:k]]
                recall = 1.0 if expected_source in top_k_sources else 0.0
                recall_scores[f'recall@{k}'] = recall
            else:
                # If no expected source, assume perfect recall
                recall_scores[f'recall@{k}'] = 1.0
        
        return {
            'question': question,
            'num_retrieved': len(results),
            'recall_scores': recall_scores,
            'retrieval_results': results
        }
    
    def evaluate_generation(self, question: str, expected_answer: str,
                           retrieval_results: List[Dict]) -> Dict:
        """
        Evaluate answer generation for a single question.
        
        Args:
            question: Test question
            expected_answer: Expected answer
            retrieval_results: Retrieved chunks
            
        Returns:
            Dictionary with generation metrics
        """
        # Build context from retrieval results
        context = self.retriever.get_context_string(retrieval_results)
        
        # Generate answer
        try:
            result = self.generator.generate(context, question, retrieval_results)
            generated_answer = result['answer']
            success = result['success']
            backend = result['backend']
        except Exception as e:
            generated_answer = f"Generation failed: {str(e)}"
            success = False
            backend = 'error'
        
        # Simple correctness check (substring match)
        # This is a basic heuristic - for production, use more sophisticated metrics
        correctness = self._calculate_correctness(generated_answer, expected_answer)
        
        return {
            'question': question,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'correctness': correctness,
            'success': success,
            'backend': backend
        }
    
    def _calculate_correctness(self, generated: str, expected: str) -> float:
        """
        Calculate a simple correctness score.
        
        Args:
            generated: Generated answer
            expected: Expected answer
            
        Returns:
            Correctness score between 0 and 1
        """
        if not generated or not expected:
            return 0.0
        
        generated_lower = generated.lower()
        expected_lower = expected.lower()
        
        # Check if key phrases from expected answer appear in generated answer
        # Split expected answer into key phrases (simple approach)
        expected_phrases = [p.strip() for p in expected_lower.split('.') if p.strip()]
        
        if not expected_phrases:
            # Fallback to word overlap
            expected_words = set(expected_lower.split())
            generated_words = set(generated_lower.split())
            
            if not expected_words:
                return 0.0
            
            overlap = len(expected_words & generated_words)
            return overlap / len(expected_words)
        
        # Count how many phrases are present
        matches = 0
        for phrase in expected_phrases:
            if phrase in generated_lower:
                matches += 1
        
        return matches / len(expected_phrases)
    
    def run_full_evaluation(self) -> Dict:
        """
        Run evaluation on all test Q&A pairs with all strategy combinations.
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.test_data:
            print("No test data available. Skipping evaluation.")
            return {}
        
        # Strategy combinations to test
        chunking_strategies = ["fixed_size", "sentence_based"]
        retrieval_strategies = ["vector", "hybrid"]
        
        all_results = []
        
        for chunking_strategy in chunking_strategies:
            for retrieval_strategy in retrieval_strategies:
                print(f"\n{'='*60}")
                print(f"Evaluating: {chunking_strategy} + {retrieval_strategy}")
                print(f"{'='*60}")
                
                strategy_results = {
                    'chunking_strategy': chunking_strategy,
                    'retrieval_strategy': retrieval_strategy,
                    'results': []
                }
                
                total_recall = {f'recall@{k}': 0 for k in EVAL_TOP_K_VALUES}
                total_correctness = 0
                num_tests = len(self.test_data)
                
                for i, test_item in enumerate(self.test_data):
                    question = test_item['question']
                    expected_answer = test_item['answer']
                    expected_source = test_item.get('source', None)
                    
                    print(f"\n[{i+1}/{num_tests}] Question: {question[:50]}...")
                    
                    # Evaluate retrieval
                    retrieval_result = self.evaluate_retrieval(
                        question, expected_source, chunking_strategy, retrieval_strategy
                    )
                    
                    # Evaluate generation
                    generation_result = self.evaluate_generation(
                        question, expected_answer, retrieval_result['retrieval_results']
                    )
                    
                    # Combine results
                    test_result = {
                        **retrieval_result,
                        **generation_result
                    }
                    strategy_results['results'].append(test_result)
                    
                    # Accumulate metrics
                    for k in EVAL_TOP_K_VALUES:
                        total_recall[f'recall@{k}'] += retrieval_result['recall_scores'][f'recall@{k}']
                    total_correctness += generation_result['correctness']
                    
                    # Rate limiting delay
                    time.sleep(EVAL_DELAY_SECONDS)
                
                # Calculate averages
                avg_metrics = {
                    f'avg_recall@{k}': total_recall[f'recall@{k}'] / num_tests
                    for k in EVAL_TOP_K_VALUES
                }
                avg_metrics['avg_correctness'] = total_correctness / num_tests
                
                strategy_results['metrics'] = avg_metrics
                
                print(f"\nResults for {chunking_strategy} + {retrieval_strategy}:")
                for metric, value in avg_metrics.items():
                    print(f"  {metric}: {value:.4f}")
                
                all_results.append(strategy_results)
        
        # Generate summary table
        self._print_summary_table(all_results)
        
        # Save results to CSV
        self._save_results_to_csv(all_results)
        
        return {
            'all_results': all_results,
            'summary': self._generate_summary(all_results)
        }
    
    def _print_summary_table(self, all_results: List[Dict]):
        """
        Print a summary table of all strategy combinations.
        
        Args:
            all_results: List of strategy results
        """
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Chunking':<15} {'Retrieval':<15} ", end="")
        for k in EVAL_TOP_K_VALUES:
            print(f"Recall@{k:<10} ", end="")
        print("Correctness")
        print("-" * 80)
        
        for result in all_results:
            chunking = result['chunking_strategy']
            retrieval = result['retrieval_strategy']
            metrics = result['metrics']
            
            print(f"{chunking:<15} {retrieval:<15} ", end="")
            for k in EVAL_TOP_K_VALUES:
                print(f"{metrics[f'avg_recall@{k}']:<10.4f} ", end="")
            print(f"{metrics['avg_correctness']:.4f}")
    
    def _save_results_to_csv(self, all_results: List[Dict]):
        """
        Save evaluation results to a CSV file.
        
        Args:
            all_results: List of strategy results
        """
        csv_path = "evaluation_results.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Chunking Strategy', 'Retrieval Strategy']
            for k in EVAL_TOP_K_VALUES:
                header.append(f'Avg Recall@{k}')
            header.append('Avg Correctness')
            writer.writerow(header)
            
            # Write rows
            for result in all_results:
                row = [
                    result['chunking_strategy'],
                    result['retrieval_strategy']
                ]
                for k in EVAL_TOP_K_VALUES:
                    row.append(result['metrics'][f'avg_recall@{k}'])
                row.append(result['metrics']['avg_correctness'])
                writer.writerow(row)
        
        print(f"\nResults saved to {csv_path}")
    
    def _generate_summary(self, all_results: List[Dict]) -> Dict:
        """
        Generate a summary of the best performing configuration.
        
        Args:
            all_results: List of strategy results
            
        Returns:
            Dictionary with best configuration summary
        """
        best_config = None
        best_score = -1
        
        for result in all_results:
            # Use F1-like score combining recall@5 and correctness
            score = (result['metrics']['avg_recall@5'] + result['metrics']['avg_correctness']) / 2
            
            if score > best_score:
                best_score = score
                best_config = result
        
        return {
            'best_chunking': best_config['chunking_strategy'],
            'best_retrieval': best_config['retrieval_strategy'],
            'best_score': best_score,
            'best_metrics': best_config['metrics']
        }


# Main execution
if __name__ == "__main__":
    evaluator = Evaluator()
    
    if evaluator.test_data:
        print("Starting full evaluation...")
        results = evaluator.run_full_evaluation()
        
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        summary = results['summary']
        print(f"Chunking: {summary['best_chunking']}")
        print(f"Retrieval: {summary['best_retrieval']}")
        print(f"Combined Score: {summary['best_score']:.4f}")
        print("\nMetrics:")
        for metric, value in summary['best_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("No test data available. Please create data/eval_qa.json with test Q&A pairs.")
