Intelligent Form Agent (Agentic)

An LLM-powered system for intelligent form processing.
Instead of fixed rules or templates, this agent uses an LLM to understand, extract, and reason over any form type dynamically.

Overview

The Intelligent Form Agent makes real LLM API calls to:

Extract structured fields from documents

Detect form type automatically

Answer natural language questions

Generate summaries

Analyze and compare multiple forms

Each operation returns structured outputs such as answers, evidence, confidence scores, and insights.

Key Capabilities

Field Extraction
Automatically identifies all key-value pairs from a document.

Form Type Detection
Determines the form type based on semantic understanding, not templates.

Question Answering
Answers questions using document reasoning and extracted fields.

Summarization
Produces concise or detailed summaries with key values.

Cross-Document Analysis
Compares multiple forms and synthesizes insights and statistics.

Architecture
User Request
     |
     v
Intelligent Form Agent
     |
     +-- Document Processor
     +-- Tool Selector (Agent Logic)
     +-- LLM API Calls
     |
     v
Structured Output
{ answer, confidence, evidence, reasoning, insights }

Project Structure
intelligent-form-agent-v2/
├── src/
│   ├── agent.py              # Agent orchestration
│   ├── llm_client.py         # LLM API wrapper
│   ├── document_processor.py # Raw document extraction
│   └── tools.py              # LLM-powered tools
├── data/sample_forms/
├── cli.py                    # CLI interface
├── app.py                    # Streamlit UI
├── demo.py                   # Demo script
├── requirements.txt
└── README.md

Installation
pip install -r requirements.txt


Set the API key:

export ANTHROPIC_API_KEY="your-api-key"

Quick Example
from src.agent import IntelligentFormAgent

agent = IntelligentFormAgent()

form = agent.process_form("tax_form.pdf")

result = agent.ask("What is the employee's total wages?", form)

print(result.answer)
print(result.confidence)
print(result.evidence)

Cross-Form Analysis Example
forms = agent.process_forms([
    "onboarding_1.txt",
    "onboarding_2.txt",
    "onboarding_3.txt"
])

analysis = agent.analyze(
    "What is the average salary by department?",
    forms
)

print(analysis.answer)
print(analysis.insights)
print(analysis.statistics)

Command Line Interface
# Process a form
python cli.py process tax_form.pdf

# Ask a question
python cli.py ask -f tax_form.pdf -q "What is the total income?"

# Summarize documents
python cli.py summarize form1.pdf form2.pdf

# Cross-form analysis
python cli.py analyze -f form1.pdf -f form2.pdf -q "Compare employees"

How the LLM Is Used

One LLM call per intelligent operation

Prompts include document text + extracted fields

Outputs are structured and include confidence and evidence

Example prompt pattern:

Analyze the document and answer the question.
Provide evidence from the document.
Return a confidence score.
