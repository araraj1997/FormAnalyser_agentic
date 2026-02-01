"""
Intelligent Form Agent - Agentic Version

An LLM-powered system for intelligent form processing.
"""

__version__ = "2.0.0"

from src.agent import IntelligentFormAgent, ProcessedForm, create_agent
from src.llm_client import ClaudeClient, get_llm_client
from src.tools import (
    ExtractionResult,
    QAResult,
    SummaryResult,
    AnalysisResult
)

__all__ = [
    "IntelligentFormAgent",
    "ProcessedForm", 
    "create_agent",
    "ClaudeClient",
    "get_llm_client",
    "ExtractionResult",
    "QAResult",
    "SummaryResult",
    "AnalysisResult"
]
