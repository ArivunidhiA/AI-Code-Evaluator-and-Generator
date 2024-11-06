# AI Code Evaluator and Generator

## Overview
The AI Code Evaluator is an advanced Python tool that generates, analyzes, and ranks code solutions using AI models. It provides comprehensive evaluation of code quality with a specific focus on AI-generated code patterns and improvements.

## Key Features
- AI Code Generation with multiple variations
- Automatic code quality assessment
- AI-specific pattern recognition and analysis
- Multiple evaluation metrics:
  - Maintainability Index
  - Cyclomatic Complexity
  - Style conformance
  - Best practices adherence
  - AI pattern detection
- Comparison of different AI-generated solutions
- Detailed feedback generation
- GitHub repository evaluation
- Pattern analysis across multiple solutions
- Real-time analysis capabilities

## AI Integration
The tool integrates with AI models to:
1. Generate multiple solution variations for the same problem
2. Analyze common patterns in AI-generated code
3. Provide AI-specific recommendations
4. Compare different AI approaches
5. Track and improve AI code generation quality

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-code-evaluator
cd ai-code-evaluator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY='your_openai_api_key'
export GITHUB_TOKEN='your_github_token'  # Optional
```

## Dependencies
- Python 3.8+
- openai
- radon
- requests
- ast (