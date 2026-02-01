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


You can run the app using streamlit as well -
streamlit run app.py

result = agent.ask("What is the employee's total wages?", form)

print(result.answer)
print(result.confidence)
print(result.evidence)
