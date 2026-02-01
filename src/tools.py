"""
Agent Tools Module

Defines the tools the agent can use, each powered by LLM calls.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from src.llm_client import BaseLLMClient, LLMResponse
from src.document_processor import ExtractedDocument


# ============================================================================
# Tool Response Types
# ============================================================================

@dataclass
class ExtractionResult:
    """Result from field extraction tool."""
    fields: Dict[str, Any]
    form_type: Optional[str]
    confidence: float
    reasoning: str


@dataclass
class QAResult:
    """Result from question answering tool."""
    answer: str
    confidence: float
    evidence: List[str]
    reasoning: str


@dataclass 
class SummaryResult:
    """Result from summarization tool."""
    summary: str
    key_points: List[str]
    form_type: str
    important_values: Dict[str, Any]


@dataclass
class AnalysisResult:
    """Result from cross-document analysis."""
    answer: str
    insights: List[str]
    comparisons: Dict[str, Any]
    statistics: Dict[str, Any]


# ============================================================================
# Base Tool Class
# ============================================================================

class BaseTool(ABC):
    """Abstract base class for agent tools."""
    
    name: str
    description: str
    
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool."""
        pass
    
    def to_tool_definition(self) -> Dict:
        """Convert to Claude tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_input_schema()
        }
    
    @abstractmethod
    def get_input_schema(self) -> Dict:
        """Get JSON schema for tool inputs."""
        pass


# ============================================================================
# Field Extraction Tool
# ============================================================================

class FieldExtractionTool(BaseTool):
    """
    Extracts structured fields from form documents using LLM.
    
    The LLM analyzes the document content and identifies key-value pairs,
    form type, and provides confidence scores.
    """
    
    name = "extract_fields"
    description = "Extract structured fields and form type from a document"
    
    SYSTEM_PROMPT = """You are an expert form analyzer. Your task is to extract structured information from form documents.

Analyze the document carefully and:
1. Identify all key-value fields (names, dates, amounts, IDs, etc.)
2. Determine the form type (W-2, insurance claim, job application, etc.)
3. Rate your confidence in the extraction (0.0-1.0)
4. Explain your reasoning

For sensitive data like SSN, mask all but the last 4 digits (e.g., XXX-XX-1234).

Be thorough but precise. Only extract fields you're confident about."""

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "document_text": {
                    "type": "string",
                    "description": "The raw text content of the document"
                }
            },
            "required": ["document_text"]
        }
    
    def run(self, document: ExtractedDocument) -> ExtractionResult:
        """
        Extract fields from a document.
        
        Args:
            document: ExtractedDocument to analyze
            
        Returns:
            ExtractionResult with extracted fields
        """
        prompt = f"""Analyze this form document and extract all fields:

<document>
{document.raw_text[:8000]}
</document>

{f"Tables found in document: {json.dumps(document.tables[:3])}" if document.tables else ""}

Extract all key-value pairs, identify the form type, and rate your confidence."""

        schema = {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "object",
                    "description": "Dictionary of field names to values"
                },
                "form_type": {
                    "type": "string",
                    "description": "Type of form (e.g., W-2, insurance_claim, job_application)"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of extraction process"
                }
            },
            "required": ["fields", "form_type", "confidence", "reasoning"]
        }
        
        result = self.llm.generate_structured(prompt, schema, system=self.SYSTEM_PROMPT)
        
        return ExtractionResult(
            fields=result.get("fields", {}),
            form_type=result.get("form_type"),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", "")
        )


# ============================================================================
# Question Answering Tool
# ============================================================================

class QuestionAnsweringTool(BaseTool):
    """
    Answers questions about form documents using LLM reasoning.
    
    The LLM reads the document content and extracted fields to provide
    accurate, evidence-based answers.
    """
    
    name = "answer_question"
    description = "Answer a question about a form document"
    
    SYSTEM_PROMPT = """You are an expert at answering questions about form documents.

Given a document and a question:
1. Carefully read and understand the document content
2. Find the relevant information to answer the question
3. Provide a clear, accurate answer
4. Cite specific evidence from the document
5. Rate your confidence (0.0-1.0)
6. Explain your reasoning

If the answer cannot be determined from the document, say so clearly.
Be precise and factual. Don't make assumptions beyond what's in the document."""

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "document_text": {"type": "string"},
                "extracted_fields": {"type": "object"}
            },
            "required": ["question", "document_text"]
        }
    
    def run(
        self,
        question: str,
        document: ExtractedDocument,
        extracted_fields: Optional[Dict] = None
    ) -> QAResult:
        """
        Answer a question about a document.
        
        Args:
            question: The question to answer
            document: Source document
            extracted_fields: Optional pre-extracted fields
            
        Returns:
            QAResult with answer and evidence
        """
        fields_context = ""
        if extracted_fields:
            fields_context = f"\n\nPreviously extracted fields:\n{json.dumps(extracted_fields, indent=2)}"
        
        prompt = f"""Answer this question about the form document:

Question: {question}

<document>
{document.raw_text[:8000]}
</document>
{fields_context}

Provide a clear answer with evidence from the document."""

        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the question"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0.0-1.0"
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quotes or references from the document supporting the answer"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of how you arrived at the answer"
                }
            },
            "required": ["answer", "confidence", "evidence", "reasoning"]
        }
        
        result = self.llm.generate_structured(prompt, schema, system=self.SYSTEM_PROMPT)
        
        return QAResult(
            answer=result.get("answer", "Unable to determine"),
            confidence=result.get("confidence", 0.0),
            evidence=result.get("evidence", []),
            reasoning=result.get("reasoning", "")
        )


# ============================================================================
# Summarization Tool
# ============================================================================

class SummarizationTool(BaseTool):
    """
    Generates intelligent summaries of form documents.
    
    The LLM creates concise, informative summaries highlighting
    the most important information.
    """
    
    name = "summarize_document"
    description = "Generate a summary of a form document"
    
    SYSTEM_PROMPT = """You are an expert at summarizing form documents.

Create summaries that:
1. Identify the form type and purpose
2. Highlight the most important information
3. List key data points and values
4. Note any unusual or notable items
5. Be concise but comprehensive

Format the summary clearly with key points and important values."""

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "document_text": {"type": "string"},
                "extracted_fields": {"type": "object"},
                "style": {"type": "string", "enum": ["brief", "detailed", "bullet_points"]}
            },
            "required": ["document_text"]
        }
    
    def run(
        self,
        document: ExtractedDocument,
        extracted_fields: Optional[Dict] = None,
        style: str = "detailed"
    ) -> SummaryResult:
        """
        Generate a summary of a document.
        
        Args:
            document: Document to summarize
            extracted_fields: Optional pre-extracted fields
            style: Summary style (brief, detailed, bullet_points)
            
        Returns:
            SummaryResult with summary and key points
        """
        fields_context = ""
        if extracted_fields:
            fields_context = f"\n\nExtracted fields:\n{json.dumps(extracted_fields, indent=2)}"
        
        style_instruction = {
            "brief": "Create a 2-3 sentence summary.",
            "detailed": "Create a comprehensive summary with all important details.",
            "bullet_points": "Create a bullet-point summary of key information."
        }.get(style, "Create a detailed summary.")
        
        prompt = f"""Summarize this form document.

{style_instruction}

<document>
{document.raw_text[:8000]}
</document>
{fields_context}"""

        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The document summary"
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key points from the document"
                },
                "form_type": {
                    "type": "string",
                    "description": "Type of form"
                },
                "important_values": {
                    "type": "object",
                    "description": "Dictionary of the most important values"
                }
            },
            "required": ["summary", "key_points", "form_type", "important_values"]
        }
        
        result = self.llm.generate_structured(prompt, schema, system=self.SYSTEM_PROMPT)
        
        return SummaryResult(
            summary=result.get("summary", ""),
            key_points=result.get("key_points", []),
            form_type=result.get("form_type", "unknown"),
            important_values=result.get("important_values", {})
        )


# ============================================================================
# Cross-Document Analysis Tool
# ============================================================================

class CrossDocumentAnalysisTool(BaseTool):
    """
    Analyzes multiple documents to find patterns and insights.
    
    The LLM compares documents, aggregates information, and provides
    holistic insights across the document set.
    """
    
    name = "analyze_documents"
    description = "Analyze multiple documents to find patterns and answer questions"
    
    SYSTEM_PROMPT = """You are an expert at analyzing multiple form documents together.

When analyzing multiple documents:
1. Compare similar fields across documents
2. Calculate statistics for numeric values (totals, averages, ranges)
3. Identify patterns and trends
4. Answer questions that require information from multiple documents
5. Provide insights that wouldn't be visible from individual documents

Be analytical and data-driven. Support conclusions with specific evidence."""

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "fields": {"type": "object"}
                        }
                    }
                }
            },
            "required": ["question", "documents"]
        }
    
    def run(
        self,
        question: str,
        documents: List[ExtractedDocument],
        extracted_fields_list: List[Dict]
    ) -> AnalysisResult:
        """
        Analyze multiple documents.
        
        Args:
            question: Analysis question
            documents: List of documents
            extracted_fields_list: List of extracted fields for each document
            
        Returns:
            AnalysisResult with insights
        """
        # Prepare document summaries for the prompt
        doc_summaries = []
        for doc, fields in zip(documents, extracted_fields_list):
            doc_summaries.append({
                "file": doc.file_path.split("/")[-1],
                "fields": fields,
                "preview": doc.raw_text[:1000]
            })
        
        prompt = f"""Analyze these {len(documents)} documents to answer the question.

Question: {question}

<documents>
{json.dumps(doc_summaries, indent=2)}
</documents>

Provide a comprehensive analysis with statistics, comparisons, and insights."""

        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Direct answer to the question"
                },
                "insights": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key insights discovered from the analysis"
                },
                "comparisons": {
                    "type": "object",
                    "description": "Comparisons between documents"
                },
                "statistics": {
                    "type": "object",
                    "description": "Calculated statistics (totals, averages, etc.)"
                }
            },
            "required": ["answer", "insights", "comparisons", "statistics"]
        }
        
        result = self.llm.generate_structured(prompt, schema, system=self.SYSTEM_PROMPT)
        
        return AnalysisResult(
            answer=result.get("answer", ""),
            insights=result.get("insights", []),
            comparisons=result.get("comparisons", {}),
            statistics=result.get("statistics", {})
        )


# ============================================================================
# Tool Registry
# ============================================================================

def get_all_tools(llm_client: BaseLLMClient) -> Dict[str, BaseTool]:
    """
    Get all available tools.
    
    Args:
        llm_client: LLM client to use for tools
        
    Returns:
        Dictionary of tool name to tool instance
    """
    return {
        "extract_fields": FieldExtractionTool(llm_client),
        "answer_question": QuestionAnsweringTool(llm_client),
        "summarize_document": SummarizationTool(llm_client),
        "analyze_documents": CrossDocumentAnalysisTool(llm_client)
    }
