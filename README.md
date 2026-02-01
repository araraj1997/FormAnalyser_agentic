# Intelligent Form Agent (Agentic Version)

An **LLM-powered** agent for intelligent form processing. Unlike traditional rule-based approaches, this agent uses Claude's reasoning capabilities to understand and process any form type dynamically.

##  What Makes This Agentic?

This system makes **actual LLM API calls** for every intelligent operation:

| Operation | LLM Usage |
|-----------|-----------|
| **Field Extraction** | LLM analyzes document and identifies all key-value pairs |
| **Form Type Detection** | LLM determines form type based on content understanding |
| **Question Answering** | LLM reasons over document to answer natural language questions |
| **Summarization** | LLM generates contextual summaries with key insights |
| **Cross-Form Analysis** | LLM compares multiple documents and synthesizes insights |

Each operation produces **reasoning traces** and **confidence scores** from the LLM.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER REQUEST                                 │
│        "What is the total income across all forms?"             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INTELLIGENT FORM AGENT                          │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Document    │    │    Tool      │    │    Claude LLM    │  │
│  │  Processor   │───▶│  Selector    │───▶│    API Calls     │  │
│  │  (Extract)   │    │  (Agent)     │    │    (Reasoning)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                     │             │
│         ▼                   ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    AVAILABLE TOOLS                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │   │
│  │  │  Extract    │ │  Answer     │ │    Summarize      │  │   │
│  │  │  Fields     │ │  Question   │ │    Document       │  │   │
│  │  │  (LLM)      │ │  (LLM)      │ │    (LLM)          │  │   │
│  │  └─────────────┘ └─────────────┘ └───────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────┐│   │
│  │  │           Cross-Document Analysis (LLM)              ││   │
│  │  └─────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STRUCTURED OUTPUT                           │
│   { answer, confidence, evidence, reasoning, insights }         │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
intelligent-form-agent-v2/
├── src/
│   ├── __init__.py
│   ├── agent.py              # Main agent orchestrator
│   ├── llm_client.py         # Claude API wrapper
│   ├── document_processor.py # Raw document extraction
│   └── tools.py              # LLM-powered tools
├── data/sample_forms/        # Sample test forms
├── cli.py                    # Command-line interface
├── app.py                    # Streamlit web UI
├── demo.py                   # Demonstration script
├── requirements.txt          # Dependencies
└── README.md
```

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### 3. Run the Demo

```bash
python demo.py
```

## 📊 Example Runs

### Example 1: Single Form Question Answering

```python
from src.agent import IntelligentFormAgent

agent = IntelligentFormAgent()

# Process a W-2 tax form
form = agent.process_form("tax_form.pdf")

# Ask a question (LLM reasons over the document)
result = agent.ask("What is the employee's total wages?", form)

print(result.answer)      # "The employee's total wages are $75,000.00"
print(result.confidence)  # 0.95
print(result.evidence)    # ["Box 1 - Wages: $75,000.00"]
print(result.reasoning)   # "I found the wages in Box 1 of the W-2 form..."
```

### Example 2: Form Summarization

```python
# Generate an intelligent summary (LLM creates summary)
summary = agent.summarize(form, style="detailed")

print(summary.summary)
# "This W-2 Wage and Tax Statement shows that employee John Smith
#  (SSN: XXX-XX-6789) earned $75,000 in wages from Acme Corporation
#  during tax year 2024. Federal income tax withheld was $11,250..."

print(summary.key_points)
# ["Employee: John Smith", "Total Wages: $75,000", 
#  "Federal Tax Withheld: $11,250", "State: CA"]

print(summary.important_values)
# {"Total Wages": "$75,000", "Federal Tax": "$11,250", ...}
```

### Example 3: Cross-Form Analysis

```python
# Load multiple employee onboarding forms
forms = agent.process_forms([
    "onboarding_1.txt",
    "onboarding_2.txt", 
    "onboarding_3.txt"
])

# Ask a question across all forms (LLM synthesizes information)
result = agent.analyze(
    "What is the average salary by department?",
    forms
)

print(result.answer)
# "The average salaries by department are:
#  - Engineering: $110,000
#  - Marketing: $80,000
#  - Sales: $65,000
#  The overall average is $85,000."

print(result.insights)
# ["Engineering has the highest starting salary",
#  "All employees report to managers in the SF office",
#  "2 of 3 employees chose PPO health insurance"]

print(result.statistics)
# {"average_salary": 85000, "total_salaries": 255000,
#  "salary_range": {"min": 65000, "max": 110000}}
```

## 🖥️ Web Interface

Launch the Streamlit UI:

```bash
streamlit run app.py
```

Features:
- Drag-and-drop file upload
- Real-time field extraction with confidence scores
- Interactive Q&A with evidence display
- Summary generation with multiple styles
- Cross-document analysis dashboard



## 🔧 Python API

```python
from src.agent import IntelligentFormAgent

# Initialize with your API key
agent = IntelligentFormAgent(
    api_key="your-key",  # or set ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-20250514",
    verbose=True
)

# Process a form (1 LLM call for extraction)
form = agent.process_form("document.pdf")

# Access extracted data
print(form.extracted_fields)  # All key-value pairs
print(form.form_type)         # Detected form type
print(form.extraction_confidence)  # Confidence score

# Ask questions (1 LLM call per question)
answer = agent.ask("What is the due date?", form)

# Generate summary (1 LLM call)
summary = agent.summarize(form)

# Analyze multiple forms (1 LLM call for analysis)
analysis = agent.analyze("Compare salaries", [form1, form2, form3])

# Check statistics
print(agent.total_llm_calls)  # Number of API calls made
```

## 🧠 How the LLM is Used

### Field Extraction Prompt
```
Analyze this form document and extract all fields:

<document>
[Document content]
</document>

Extract all key-value pairs, identify the form type, and rate your confidence.
```

### Question Answering Prompt
```
Answer this question about the form document:

Question: [User's question]

<document>
[Document content]
</document>

Previously extracted fields:
[Structured fields]

Provide a clear answer with evidence from the document.
```

### Cross-Document Analysis Prompt
```
Analyze these [N] documents to answer the question.

Question: [Analysis question]

<documents>
[All documents with extracted fields]
</documents>

Provide a comprehensive analysis with statistics, comparisons, and insights.
```
