import logging
from typing import Any, Dict, List, Tuple
from pipeline.utils import node_decorator, get_last_node_result
from runner.check_and_correct import sql_raw_parse
from runner.database_manager import DatabaseManager
import sqlite3
import math

class SelectionAgent:
    """
    Implements pairwise comparison-based selection of SQL candidates.
    Uses tournament-style scoring to identify the best query.
    Adapted for OpenSearch-SQL framework.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.comparison_cache = {}  # Cache pairwise comparisons
    
    def select_best_candidate(self, candidates: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        Select the best SQL candidate from a list using pairwise comparisons.
        
        Args:
            candidates: List of candidate queries with their metadata
            question: The original natural language question
            
        Returns:
            The best candidate with selection details
        """
        
        if len(candidates) == 0:
            raise ValueError("No candidates provided")
        
        if len(candidates) == 1:
            return {
                'selected_candidate': candidates[0],
                'selection_details': {
                    'total_candidates': 1,
                    'comparison_matrix': None,
                    'scores': [1.0],
                    'reasoning': 'Only one candidate available'
                }
            }
        
        # Build comparison matrix
        n = len(candidates)
        # Use list of lists instead of numpy
        comparison_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        scores = [0.0] * n
        
        # Perform pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                winner_idx = self._compare_candidates(
                    candidates[i], 
                    candidates[j], 
                    question,
                    i, j
                )
                
                if winner_idx == i:
                    comparison_matrix[i][j] = 1
                    comparison_matrix[j][i] = 0
                    scores[i] += 1
                else:
                    comparison_matrix[i][j] = 0
                    comparison_matrix[j][i] = 1
                    scores[j] += 1
        
        # Find candidate with highest score
        best_idx = scores.index(max(scores))
        
        return {
            'selected_candidate': candidates[best_idx],
            'selection_details': {
                'total_candidates': n,
                'comparison_matrix': comparison_matrix,
                'scores': scores,
                'winner_index': best_idx,
                'reasoning': self._generate_selection_reasoning(candidates, scores, best_idx)
            }
        }
    
    def _compare_candidates(self, candidate1: Dict, candidate2: Dict, 
                          question: str, idx1: int, idx2: int) -> int:
        """
        Compare two candidates and return the index of the better one.
        
        Returns:
            Index of the winning candidate (idx1 or idx2)
        """
        
        # Check cache first
        cache_key = (candidate1['sql'], candidate2['sql'])
        if cache_key in self.comparison_cache:
            return self.comparison_cache[cache_key]
        
        # Get execution results (prefer pre-calculated)
        result1 = self._get_execution_result(candidate1)
        result2 = self._get_execution_result(candidate2)
        
        # If execution results are the same, use other criteria
        # Compare sets for equality if results are lists/tuples
        res1_data = set(result1.get('results', [])) if result1.get('results') else set()
        res2_data = set(result2.get('results', [])) if result2.get('results') else set()
        
        if result1['success'] == result2['success'] and res1_data == res2_data:
            winner_idx = self._compare_by_quality(candidate1, candidate2, question, idx1, idx2)
        else:
            winner_idx = self._compare_by_execution(result1, result2, candidate1, candidate2, 
                                                   question, idx1, idx2)
        
        # Cache the result
        self.comparison_cache[cache_key] = winner_idx
        
        return winner_idx
    
    def _get_execution_result(self, candidate: Dict) -> Dict[str, Any]:
        """Get execution result from candidate or execute if missing."""
        if 'execution_result' in candidate:
            return candidate['execution_result']
        return self._execute_candidate(candidate['sql'])

    def _execute_candidate(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'results': results,
                'row_count': len(results),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'results': None,
                'row_count': 0,
                'error': str(e)
            }
    
    def _compare_by_execution(self, result1: Dict, result2: Dict, 
                            candidate1: Dict, candidate2: Dict,
                            question: str, idx1: int, idx2: int) -> int:
        """Compare candidates based on execution results."""
        
        # Prefer successful queries
        if result1['success'] and not result2['success']:
            return idx1
        elif result2['success'] and not result1['success']:
            return idx2
        
        # Both successful - compare result quality
        if result1['success'] and result2['success']:
            # Prefer non-empty results
            if result1['row_count'] > 0 and result2['row_count'] == 0:
                return idx1
            elif result2['row_count'] > 0 and result1['row_count'] == 0:
                return idx2
            
            # Both have results - use quality metrics
            return self._compare_by_quality(candidate1, candidate2, question, idx1, idx2)
        
        # Both failed - compare error types
        return self._compare_failed_queries(result1, result2, idx1, idx2)
    
    def _compare_by_quality(self, candidate1: Dict, candidate2: Dict, 
                          question: str, idx1: int, idx2: int) -> int:
        """Compare candidates based on query quality metrics."""
        
        score1 = self._calculate_quality_score(candidate1, question)
        score2 = self._calculate_quality_score(candidate2, question)
        
        return idx1 if score1 >= score2 else idx2
    
    def _calculate_quality_score(self, candidate: Dict, question: str) -> float:
        """Calculate quality score for a candidate."""
        
        score = 0.0
        sql = candidate['sql'].upper()
        question_lower = question.lower()
        
        # Confidence from generator
        if 'confidence' in candidate:
            score += candidate['confidence'] * 10
        
        # Query complexity alignment with question
        if 'join' in question_lower and 'JOIN' in sql:
            score += 2
        elif 'join' not in question_lower and 'JOIN' not in sql:
            score += 1
        
        # Aggregation alignment
        agg_keywords = ['count', 'average', 'sum', 'total', 'maximum', 'minimum']
        agg_functions = ['COUNT', 'AVG', 'SUM', 'MAX', 'MIN']
        
        question_needs_agg = any(kw in question_lower for kw in agg_keywords)
        query_has_agg = any(func in sql for func in agg_functions)
        
        if question_needs_agg == query_has_agg:
            score += 2
        
        # Filtering alignment
        if 'where' in question_lower or 'filter' in question_lower:
            if 'WHERE' in sql:
                score += 1.5
        
        # Ordering alignment
        if any(word in question_lower for word in ['top', 'highest', 'lowest', 'best', 'worst']):
            if 'ORDER BY' in sql:
                score += 1.5
            if 'LIMIT' in sql:
                score += 1
        
        # Generator type preferences (based on paper insights)
        reasoning = candidate.get('reasoning', {})
        approach = reasoning.get('approach', '')
        
        if approach == 'divide_and_conquer' and any(word in question_lower for word in ['complex', 'multiple', 'and']):
            score += 1
        elif approach == 'query_plan' and 'join' in question_lower:
            score += 1
        elif approach == 'synthetic_examples':
            score += 0.5  # Slight preference for synthetic examples
        
        # Penalize overly simple queries for complex questions
        question_words = len(question.split())
        if question_words > 10 and sql.count('FROM') == 1 and 'JOIN' not in sql:
            score -= 1
        
        return score
    
    def _compare_failed_queries(self, result1: Dict, result2: Dict, idx1: int, idx2: int) -> int:
        """Compare two failed queries based on error types."""
        
        error1 = str(result1.get('error', '')).lower()
        error2 = str(result2.get('error', '')).lower()
        
        # Prefer syntax errors over semantic errors (easier to fix)
        syntax_error_keywords = ['syntax error', 'parse error']
        semantic_error_keywords = ['no such table', 'no such column', 'ambiguous']
        
        is_syntax1 = any(kw in error1 for kw in syntax_error_keywords)
        is_syntax2 = any(kw in error2 for kw in syntax_error_keywords)
        is_semantic1 = any(kw in error1 for kw in semantic_error_keywords)
        is_semantic2 = any(kw in error2 for kw in semantic_error_keywords)
        
        if is_syntax1 and is_semantic2:
            return idx1
        elif is_syntax2 and is_semantic1:
            return idx2
        
        # Default to first candidate if no clear winner
        return idx1
    
    def _generate_selection_reasoning(self, candidates: List[Dict], scores: List[float], 
                                    winner_idx: int) -> str:
        """Generate explanation for why a candidate was selected."""
        
        winner = candidates[winner_idx]
        winner_score = scores[winner_idx]
        total_comparisons = len(candidates) - 1
        
        reasoning_parts = [
            f"Selected candidate {winner_idx + 1} out of {len(candidates)} candidates.",
            f"Won {int(winner_score)} out of {total_comparisons} pairwise comparisons."
        ]
        
        # Add approach-specific reasoning
        approach = winner.get('reasoning', {}).get('approach', 'unknown')
        if approach == 'divide_and_conquer':
            reasoning_parts.append("Used divide-and-conquer approach which broke down the complex query.")
        
        # Add confidence info
        if 'confidence' in winner:
            reasoning_parts.append(f"Generator confidence: {winner['confidence']:.2f}")
        
        return " ".join(reasoning_parts)


@node_decorator(check_schema_status=False)
def vote(task: Any, execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Voting/Fusion Node.
    Uses Pairwise Selection Agent to select the best SQL from candidates.
    """
    # Get candidates from align_correct output
    vote_data = get_last_node_result(execution_history, "align_correct")
    vote_list = vote_data["vote"] if vote_data else []
    
    if not vote_list:
        # Fallback if no candidates
        return {
            "SQL": "SELECT * FROM table",
            "SQL_correct_vote": "SELECT * FROM table",
            "nonecase": True
        }

    # Transform vote_list into candidates format for SelectionAgent
    candidates = []
    for item in vote_list:
        # Check if answer implies success (set vs list/None)
        ans = item.get('answer')
        is_success = isinstance(ans, set)
        results = list(ans) if is_success else []
        error = "Execution failed" if not is_success else None
        
        # Simple heuristic for confidence based on count (recurrence)
        confidence = 0.5 + (0.1 * item.get('count', 1))
        
        cand = {
            'sql': item['sql'],
            'execution_result': {
                'success': is_success,
                'results': results,
                'row_count': len(results),
                'error': error
            },
            'confidence': confidence,
            'reasoning': {
                'approach': 'ensemble' # default
            }
        }
        candidates.append(cand)

    # Initialize SelectionAgent
    paths = DatabaseManager()
    agent = SelectionAgent(str(paths.db_path))
    
    try:
        selection = agent.select_best_candidate(candidates, task.question)
        best_sql = selection['selected_candidate']['sql']
        reasoning = selection['selection_details']['reasoning']
    except Exception as e:
        logging.error(f"SelectionAgent failed: {e}")
        # Fallback to first candidate
        best_sql = candidates[0]['sql']
        reasoning = "Fallback due to error"

    # For compatibility, we can also pick 'SQL_correct_vote' using the same logic 
    # or just use the same result if we don't differentiate anymore.
    # The original code distinguished answer vs correct_ans. 
    # Here we simplify to just "best_sql" as the main output.
    
    nonecase = all(not c['execution_result']['success'] for c in candidates)

    print(f"****** Selected SQL: {best_sql}")
    print(f"****** Reasoning: {reasoning}")

    response = {
        "SQL": best_sql,
        "SQL_correct_vote": best_sql, # Use same for now or implement separate pass if needed
        "nonecase": nonecase,
        "selection_details": selection.get('selection_details', {}) if 'selection' in locals() else {}
    }

    return response
