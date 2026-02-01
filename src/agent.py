"""
Intelligent Form Agent - Main Agent Module

This is the core agentic system that orchestrates LLM-powered form processing.
Uses a state machine architecture for flexible, multi-step reasoning.
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from src.llm_client import BaseLLMClient, ClaudeClient, get_llm_client
from src.document_processor import DocumentProcessor, ExtractedDocument
from src.tools import (
    FieldExtractionTool,
    QuestionAnsweringTool, 
    SummarizationTool,
    CrossDocumentAnalysisTool,
    ExtractionResult,
    QAResult,
    SummaryResult,
    AnalysisResult,
    get_all_tools
)


# ============================================================================
# State Management
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the agent."""
    # Input
    task: str
    task_type: str  # "extract", "qa", "summarize", "analyze"
    question: Optional[str]
    
    # Documents
    documents: List[ExtractedDocument]
    extracted_fields: List[Dict[str, Any]]
    
    # Processing
    current_step: str
    steps_taken: List[str]
    
    # Output
    result: Optional[Any]
    error: Optional[str]
    
    # Metadata
    llm_calls: int
    start_time: str


@dataclass
class ProcessedForm:
    """Represents a fully processed form document."""
    file_path: str
    file_type: str
    raw_text: str
    extracted_fields: Dict[str, Any]
    form_type: Optional[str]
    extraction_confidence: float
    tables: List[List[List[str]]]
    metadata: Dict[str, Any]
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ============================================================================
# Main Agent Class
# ============================================================================

class IntelligentFormAgent:
    """
    An LLM-powered agent for intelligent form processing.
    
    This agent can:
    - Extract structured fields from any form type
    - Answer natural language questions about forms
    - Generate intelligent summaries
    - Analyze patterns across multiple forms
    
    Each capability is powered by LLM reasoning, making it adaptive
    to new form types without explicit programming.
    
    Example:
        agent = IntelligentFormAgent(api_key="your-key")
        
        # Process a form
        form = agent.process_form("tax_form.pdf")
        print(form.extracted_fields)
        
        # Ask questions
        answer = agent.ask("What is the total income?", form)
        print(answer.answer)
        
        # Summarize
        summary = agent.summarize(form)
        print(summary.summary)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = False
    ):
        """
        Initialize the agent.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use
            verbose: Whether to print debug information
        """
        self.llm = get_llm_client("claude", api_key=api_key, model=model)
        self.doc_processor = DocumentProcessor()
        self.verbose = verbose
        
        # Initialize tools
        self.tools = get_all_tools(self.llm)
        self.extraction_tool = self.tools["extract_fields"]
        self.qa_tool = self.tools["answer_question"]
        self.summary_tool = self.tools["summarize_document"]
        self.analysis_tool = self.tools["analyze_documents"]
        
        # Document cache
        self._processed_forms: Dict[str, ProcessedForm] = {}
        
        # Statistics
        self.total_llm_calls = 0
    
    def _log(self, message: str):
        """Log a message if verbose mode is on."""
        if self.verbose:
            print(f"[Agent] {message}")
    
    # ========================================================================
    # Core Methods
    # ========================================================================
    
    def process_form(self, file_path: str, force_reprocess: bool = False) -> ProcessedForm:
        """
        Process a form document: extract text and fields.
        
        This is the primary entry point for loading a form. The agent will:
        1. Extract raw text from the document (PDF, image, or text)
        2. Use LLM to identify and extract all fields
        3. Determine the form type
        
        Args:
            file_path: Path to the form file
            force_reprocess: Force reprocessing even if cached
            
        Returns:
            ProcessedForm with extracted data
        """
        # Check cache
        if not force_reprocess and file_path in self._processed_forms:
            self._log(f"Returning cached form: {file_path}")
            return self._processed_forms[file_path]
        
        self._log(f"Processing form: {file_path}")
        
        # Step 1: Extract raw content
        extracted = self.doc_processor.process(file_path)
        self._log(f"Extracted {len(extracted.raw_text)} chars, {len(extracted.tables)} tables")
        
        # Step 2: Use LLM to extract fields
        self._log("Calling LLM for field extraction...")
        extraction_result = self.extraction_tool.run(extracted)
        self.total_llm_calls += 1
        
        self._log(f"Extracted {len(extraction_result.fields)} fields, "
                  f"form type: {extraction_result.form_type}, "
                  f"confidence: {extraction_result.confidence:.1%}")
        
        # Create ProcessedForm
        form = ProcessedForm(
            file_path=file_path,
            file_type=extracted.file_type,
            raw_text=extracted.raw_text,
            extracted_fields=extraction_result.fields,
            form_type=extraction_result.form_type,
            extraction_confidence=extraction_result.confidence,
            tables=extracted.tables,
            metadata={
                **extracted.metadata,
                "extraction_reasoning": extraction_result.reasoning
            }
        )
        
        # Cache it
        self._processed_forms[file_path] = form
        
        return form
    
    def process_forms(self, file_paths: List[str]) -> List[ProcessedForm]:
        """
        Process multiple forms.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of ProcessedForm objects
        """
        return [self.process_form(path) for path in file_paths]
    
    def ask(
        self,
        question: str,
        form: ProcessedForm,
        include_reasoning: bool = False
    ) -> QAResult:
        """
        Ask a question about a form.
        
        The LLM will reason over the document content and extracted fields
        to provide an accurate answer with evidence.
        
        Args:
            question: Natural language question
            form: ProcessedForm to query
            include_reasoning: Whether to include detailed reasoning
            
        Returns:
            QAResult with answer, confidence, and evidence
        """
        self._log(f"Answering question: {question}")
        
        # Reconstruct ExtractedDocument for the tool
        doc = ExtractedDocument(
            file_path=form.file_path,
            file_type=form.file_type,
            raw_text=form.raw_text,
            tables=form.tables,
            metadata=form.metadata
        )
        
        result = self.qa_tool.run(
            question=question,
            document=doc,
            extracted_fields=form.extracted_fields
        )
        self.total_llm_calls += 1
        
        self._log(f"Answer confidence: {result.confidence:.1%}")
        
        return result
    
    def ask_multiple(
        self,
        question: str,
        forms: List[ProcessedForm]
    ) -> AnalysisResult:
        """
        Ask a question across multiple forms.
        
        Args:
            question: Question that requires multiple forms to answer
            forms: List of forms to query
            
        Returns:
            AnalysisResult with aggregated answer
        """
        return self.analyze(question, forms)
    
    def summarize(
        self,
        form: ProcessedForm,
        style: str = "detailed"
    ) -> SummaryResult:
        """
        Generate a summary of a form.
        
        Args:
            form: Form to summarize
            style: Summary style ("brief", "detailed", "bullet_points")
            
        Returns:
            SummaryResult with summary and key points
        """
        self._log(f"Summarizing form: {form.file_path}")
        
        doc = ExtractedDocument(
            file_path=form.file_path,
            file_type=form.file_type,
            raw_text=form.raw_text,
            tables=form.tables,
            metadata=form.metadata
        )
        
        result = self.summary_tool.run(
            document=doc,
            extracted_fields=form.extracted_fields,
            style=style
        )
        self.total_llm_calls += 1
        
        return result
    
    def analyze(
        self,
        question: str,
        forms: List[ProcessedForm]
    ) -> AnalysisResult:
        """
        Analyze multiple forms to answer a question or find patterns.
        
        The LLM will:
        - Compare fields across documents
        - Calculate statistics
        - Identify patterns and trends
        - Provide insights
        
        Args:
            question: Analysis question
            forms: List of forms to analyze
            
        Returns:
            AnalysisResult with insights and statistics
        """
        self._log(f"Analyzing {len(forms)} forms: {question}")
        
        documents = [
            ExtractedDocument(
                file_path=f.file_path,
                file_type=f.file_type,
                raw_text=f.raw_text,
                tables=f.tables,
                metadata=f.metadata
            )
            for f in forms
        ]
        
        fields_list = [f.extracted_fields for f in forms]
        
        result = self.analysis_tool.run(
            question=question,
            documents=documents,
            extracted_fields_list=fields_list
        )
        self.total_llm_calls += 1
        
        return result
    
    def compare(
        self,
        form1: ProcessedForm,
        form2: ProcessedForm
    ) -> AnalysisResult:
        """
        Compare two forms.
        
        Args:
            form1: First form
            form2: Second form
            
        Returns:
            AnalysisResult with comparison
        """
        return self.analyze(
            "Compare these two forms. What are the similarities and differences? "
            "Identify matching fields and highlight any discrepancies.",
            [form1, form2]
        )
    
    # ========================================================================
    # Export Methods
    # ========================================================================
    
    def export_json(self, form: ProcessedForm, output_path: Optional[str] = None) -> str:
        """Export form data to JSON."""
        json_str = form.to_json()
        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
        return json_str
    
    def export_summary(
        self,
        form: ProcessedForm,
        output_path: Optional[str] = None
    ) -> str:
        """Export a human-readable summary."""
        summary = self.summarize(form, style="detailed")
        
        output = f"""# Form Summary Report
Generated: {datetime.now().isoformat()}
File: {form.file_path}
Type: {form.form_type}
Extraction Confidence: {form.extraction_confidence:.1%}

## Summary
{summary.summary}

## Key Points
{chr(10).join(f"- {point}" for point in summary.key_points)}

## Important Values
{chr(10).join(f"- **{k}**: {v}" for k, v in summary.important_values.items())}

## All Extracted Fields
{chr(10).join(f"- **{k}**: {v}" for k, v in form.extracted_fields.items())}
"""
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
        
        return output
    
    # ========================================================================
    # Agentic Workflow Methods
    # ========================================================================
    
    def run_workflow(
        self,
        task: str,
        file_paths: List[str],
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete agentic workflow.
        
        The agent will decide what steps to take based on the task.
        
        Args:
            task: Description of what to do
            file_paths: Paths to form files
            question: Optional specific question
            
        Returns:
            Workflow results
        """
        self._log(f"Running workflow: {task}")
        
        # Step 1: Process all documents
        forms = self.process_forms(file_paths)
        
        # Step 2: Determine what the user wants
        task_analysis = self._analyze_task(task, question, len(forms))
        
        # Step 3: Execute based on task type
        if task_analysis["task_type"] == "extract":
            return {
                "task": task,
                "type": "extraction",
                "forms": [f.to_dict() for f in forms]
            }
        
        elif task_analysis["task_type"] == "qa":
            if len(forms) == 1:
                result = self.ask(question or task, forms[0])
            else:
                result = self.ask_multiple(question or task, forms)
            
            return {
                "task": task,
                "type": "qa",
                "answer": result.answer if hasattr(result, "answer") else str(result),
                "confidence": getattr(result, "confidence", None),
                "evidence": getattr(result, "evidence", []),
                "insights": getattr(result, "insights", [])
            }
        
        elif task_analysis["task_type"] == "summarize":
            summaries = [self.summarize(f) for f in forms]
            return {
                "task": task,
                "type": "summarization",
                "summaries": [
                    {
                        "file": forms[i].file_path,
                        "summary": s.summary,
                        "key_points": s.key_points
                    }
                    for i, s in enumerate(summaries)
                ]
            }
        
        elif task_analysis["task_type"] == "analyze":
            result = self.analyze(question or task, forms)
            return {
                "task": task,
                "type": "analysis",
                "answer": result.answer,
                "insights": result.insights,
                "statistics": result.statistics,
                "comparisons": result.comparisons
            }
        
        else:
            # Default: extract and summarize
            summaries = [self.summarize(f) for f in forms]
            return {
                "task": task,
                "type": "general",
                "forms": [f.to_dict() for f in forms],
                "summaries": [s.summary for s in summaries]
            }
    
    def _analyze_task(
        self,
        task: str,
        question: Optional[str],
        num_docs: int
    ) -> Dict[str, str]:
        """
        Analyze the task to determine what type of operation to perform.
        
        Uses LLM to understand user intent.
        """
        prompt = f"""Analyze this task and determine what type of form processing is needed.

Task: {task}
{f"Question: {question}" if question else ""}
Number of documents: {num_docs}

What type of task is this?
- "extract": User wants to extract/see the fields from forms
- "qa": User is asking a specific question about the forms
- "summarize": User wants a summary of the forms
- "analyze": User wants analysis/comparison across multiple forms"""

        schema = {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["extract", "qa", "summarize", "analyze"]
                },
                "reasoning": {"type": "string"}
            },
            "required": ["task_type"]
        }
        
        result = self.llm.generate_structured(prompt, schema)
        self.total_llm_calls += 1
        
        return result
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_llm_calls": self.total_llm_calls,
            "cached_forms": len(self._processed_forms),
            "cached_form_paths": list(self._processed_forms.keys())
        }
    
    def clear_cache(self):
        """Clear the form cache."""
        self._processed_forms.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_agent(api_key: Optional[str] = None, verbose: bool = False) -> IntelligentFormAgent:
    """
    Create an agent instance.
    
    Args:
        api_key: Anthropic API key
        verbose: Enable verbose logging
        
    Returns:
        Configured agent
    """
    return IntelligentFormAgent(api_key=api_key, verbose=verbose)
