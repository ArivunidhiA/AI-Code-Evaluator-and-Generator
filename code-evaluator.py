import os
import ast
import radon.metrics as metrics
from radon.complexity import cc_visit
import logging
from typing import Dict, List, Tuple, Optional
import requests
import json
from datetime import datetime
import openai
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

@dataclass
class CodeSolution:
    code: str
    model: str
    task: str
    score: float = 0.0
    feedback: List[str] = None

class AICodeEvaluator:
    def __init__(self, openai_api_key: str = None, github_token: str = None):
        self.github_token = github_token
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def generate_solutions(self, task_description: str, 
                               num_variations: int = 3) -> List[CodeSolution]:
        """
        Generate multiple solutions for the same coding task using different AI models
        or parameters.
        """
        try:
            solutions = []
            
            # Generate solution using OpenAI
            if self.openai_api_key:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert Python programmer. Generate clean, efficient code solutions."},
                        {"role": "user", "content": f"Generate a Python solution for: {task_description}"}
                    ],
                    n=num_variations,
                    temperature=0.7
                )
                
                for choice in response.choices:
                    solutions.append(CodeSolution(
                        code=choice.message.content,
                        model="gpt-4",
                        task=task_description
                    ))
            
            return solutions
        except Exception as e:
            self.logger.error(f"Error generating solutions: {str(e)}")
            return []

    def evaluate_ai_solutions(self, solutions: List[CodeSolution]) -> List[CodeSolution]:
        """
        Evaluate multiple AI-generated solutions and rank them.
        """
        try:
            for solution in solutions:
                evaluation = self.evaluate_code(solution.code)
                solution.score = evaluation['overall_score']
                solution.feedback = evaluation['feedback']
            
            # Sort solutions by score
            return sorted(solutions, key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error evaluating AI solutions: {str(e)}")
            return solutions

    def analyze_ai_patterns(self, solutions: List[CodeSolution]) -> Dict:
        """
        Analyze patterns and trends in AI-generated code.
        """
        analysis = {
            'common_patterns': [],
            'style_consistency': 0.0,
            'approach_diversity': 0.0,
            'recommendations': []
        }
        
        try:
            # Analyze common patterns
            trees = [ast.parse(sol.code) for sol in solutions]
            
            # Check for common approaches
            approach_patterns = self._identify_approach_patterns(trees)
            analysis['common_patterns'] = approach_patterns
            
            # Calculate style consistency
            style_scores = [self._check_style(sol.code) for sol in solutions]
            analysis['style_consistency'] = self._calculate_consistency(style_scores)
            
            # Measure approach diversity
            analysis['approach_diversity'] = self._calculate_approach_diversity(trees)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_ai_recommendations(solutions)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing AI patterns: {str(e)}")
            return analysis

    def evaluate_code(self, code: str) -> Dict:
        """
        Evaluates code quality using multiple metrics.
        Returns a dictionary with scores and feedback.
        """
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Calculate metrics
            maintainability = metrics.mi_visit(code, True)
            complexity = self._calculate_complexity(code)
            style_score = self._check_style(code)
            best_practices_score = self._check_best_practices(tree)
            ai_patterns_score = self._evaluate_ai_patterns(tree)
            
            # Calculate overall score (weighted average)
            overall_score = (
                maintainability * 0.25 +
                (100 - complexity) * 0.25 +
                style_score * 0.2 +
                best_practices_score * 0.15 +
                ai_patterns_score * 0.15
            )
            
            return {
                'overall_score': round(overall_score, 2),
                'maintainability': round(maintainability, 2),
                'complexity': complexity,
                'style_score': style_score,
                'best_practices_score': best_practices_score,
                'ai_patterns_score': ai_patterns_score,
                'feedback': self._generate_feedback(maintainability, complexity, 
                                                 style_score, best_practices_score,
                                                 ai_patterns_score),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating code: {str(e)}")
            return {'error': str(e)}

    def _evaluate_ai_patterns(self, tree: ast.AST) -> float:
        """
        Evaluate patterns commonly found in AI-generated code.
        """
        score = 100
        
        try:
            # Check for overly verbose variable names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if len(node.id) > 30:
                        score -= 5
                
                # Check for unnecessary complexity
                if isinstance(node, ast.If):
                    if len(node.body) > 10:
                        score -= 5
                
                # Check for repetitive code patterns
                if isinstance(node, ast.FunctionDef):
                    body_str = ast.dump(node)
                    if body_str.count('(') > 15:  # Complex nested calls
                        score -= 5
            
            return max(0, score)
            
        except Exception:
            return 50  # Default score on error

    def _identify_approach_patterns(self, trees: List[ast.AST]) -> List[str]:
        """
        Identify common patterns in solution approaches.
        """
        patterns = []
        
        # Count common node types
        node_types = {}
        for tree in trees:
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Identify patterns based on node frequencies
        if node_types.get('List', 0) > len(trees) * 0.7:
            patterns.append("Heavy reliance on list operations")
        if node_types.get('Dict', 0) > len(trees) * 0.7:
            patterns.append("Frequent dictionary usage")
        if node_types.get('Generator', 0) > len(trees) * 0.5:
            patterns.append("Preference for generator expressions")
            
        return patterns

    def _calculate_consistency(self, scores: List[float]) -> float:
        """
        Calculate consistency score based on variation in metrics.
        """
        if not scores:
            return 0.0
            
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        
        # Convert variance to consistency score (0-100)
        return max(0, 100 - (variance * 10))

    def _calculate_approach_diversity(self, trees: List[ast.AST]) -> float:
        """
        Calculate how diverse the solution approaches are.
        """
        if not trees:
            return 0.0
            
        # Count unique node type combinations
        approaches = set()
        for tree in trees:
            node_types = tuple(sorted(type(node).__name__ for node in ast.walk(tree)))
            approaches.add(node_types)
            
        # Calculate diversity score
        diversity = (len(approaches) / len(trees)) * 100
        return min(100, diversity)

    def _generate_ai_recommendations(self, solutions: List[CodeSolution]) -> List[str]:
        """
        Generate recommendations for improving AI-generated code.
        """
        recommendations = []
        
        # Analyze patterns across solutions
        if solutions:
            avg_score = sum(s.score for s in solutions) / len(solutions)
            
            if avg_score < 70:
                recommendations.append("Consider adjusting AI temperature for more focused solutions")
            
            # Check for consistent issues across solutions
            common_feedback = set.intersection(*[set(s.feedback) for s in solutions])
            for feedback in common_feedback:
                recommendations.append(f"Consistent issue across solutions: {feedback}")
            
            # Analyze complexity patterns
            complexity_scores = [self._calculate_complexity(s.code) for s in solutions]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            if avg_complexity > 8:
                recommendations.append("AI tends to generate overly complex solutions. Consider simplifying prompts.")
                
        return recommendations

    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity score."""
        try:
            complexity_blocks = cc_visit(code)
            if not complexity_blocks:
                return 0
            return sum(block.complexity for block in complexity_blocks) / len(complexity_blocks)
        except:
            return 100  # Maximum complexity on error

    def _check_style(self, code: str) -> float:
        """
        Check code style using basic rules.
        Returns a score from 0 to 100.
        """
        score = 100
        lines = code.split('\n')
        
        # Check line length
        for line in lines:
            if len(line) > 79:
                score -= 2
            
        # Check naming conventions
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.islower():
                    score -= 5
            elif isinstance(node, ast.ClassDef):
                if not node.name[0].isupper():
                    score -= 5
                    
        return max(0, score)

    def _check_best_practices(self, tree: ast.AST) -> float:
        """
        Check adherence to Python best practices.
        Returns a score from 0 to 100.
        """
        score = 100
        
        for node in ast.walk(tree):
            # Check for proper exception handling
            if isinstance(node, ast.Try):
                if not node.handlers:
                    score -= 10
                for handler in node.handlers:
                    if handler.type is None:  # bare except
                        score -= 10
            
            # Check for documentation
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    score -= 5
                    
        return max(0, score)

    def _generate_feedback(self, maintainability: float, complexity: float,
                         style_score: float, best_practices_score: float,
                         ai_patterns_score: float) -> List[str]:
        """Generate specific feedback based on metrics."""
        feedback = []
        
        if maintainability < 65:
            feedback.append("Code maintainability needs improvement. Consider breaking down complex functions.")
        
        if complexity > 10:
            feedback.append("High cyclomatic complexity. Consider simplifying control flow.")
            
        if style_score < 80:
            feedback.append("Style issues detected. Review PEP 8 guidelines.")
            
        if best_practices_score < 80:
            feedback.append("Consider adding proper documentation and exception handling.")
            
        if ai_patterns_score < 80:
            feedback.append("AI-specific patterns detected. Consider simplifying variable names and reducing nested complexity.")
            
        return feedback

if __name__ == "__main__":
    # Example usage with AI integration
    evaluator = AICodeEvaluator(
        openai_api_key='your_openai_api_key_here'
    )
    
    # Example task
    task = """
    Create a function that finds the longest palindromic substring in a given string.
    The function should be efficient and handle edge cases appropriately.
    """
    
    # Generate and evaluate multiple solutions
    import asyncio
    
    async def main():
        # Generate solutions
        solutions = await evaluator.generate_solutions(task, num_variations=3)
        
        # Evaluate solutions
        evaluated_solutions = evaluator.evaluate_ai_solutions(solutions)
        
        # Analyze patterns
        analysis = evaluator.analyze_ai_patterns(evaluated_solutions)
        
        # Print results
        print("\nEvaluated Solutions:")
        for i, solution in enumerate(evaluated_solutions, 1):
            print(f"\nSolution {i} (Score: {solution.score}):")
            print(solution.code)
            print("\nFeedback:", solution.feedback)
        
        print("\nAI Pattern Analysis:")
        print(json.dumps(analysis, indent=2))

    asyncio.run(main())
