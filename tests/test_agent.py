"""
Tests for Intelligent Form Agent (Agentic Version)

These tests use a mock LLM client to avoid API calls during testing.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import MockLLMClient, LLMResponse
from src.document_processor import DocumentProcessor, ExtractedDocument
from src.tools import (
    FieldExtractionTool,
    QuestionAnsweringTool,
    SummarizationTool,
    CrossDocumentAnalysisTool
)
from src.agent import IntelligentFormAgent, ProcessedForm


class TestDocumentProcessor:
    """Tests for document processing."""
    
    def setup_method(self):
        self.processor = DocumentProcessor()
    
    def test_process_text_file(self):
        """Test processing a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Name: John Doe\nAmount: $1,000")
            temp_path = f.name
        
        try:
            result = self.processor.process(temp_path)
            
            assert isinstance(result, ExtractedDocument)
            assert result.file_type == "txt"
            assert "John Doe" in result.raw_text
            assert "$1,000" in result.raw_text
        finally:
            os.unlink(temp_path)
    
    def test_process_json_file(self):
        """Test processing a JSON file."""
        data = [
            {"name": "Alice", "amount": 100},
            {"name": "Bob", "amount": 200}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = self.processor.process(temp_path)
            
            assert result.file_type == "json"
            assert len(result.tables) == 1
            assert result.tables[0][0] == ["name", "amount"]
        finally:
            os.unlink(temp_path)
    
    def test_process_csv_file(self):
        """Test processing a CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Name,Amount\nAlice,100\nBob,200")
            temp_path = f.name
        
        try:
            result = self.processor.process(temp_path)
            
            assert result.file_type == "csv"
            assert len(result.tables) == 1
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test handling of missing files."""
        with pytest.raises(FileNotFoundError):
            self.processor.process("nonexistent_file.pdf")


class TestMockLLMClient:
    """Tests for the mock LLM client."""
    
    def test_generate(self):
        """Test basic generation."""
        client = MockLLMClient()
        
        result = client.generate("Test prompt")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Mock response for testing"
        assert len(client.call_history) == 1
    
    def test_generate_structured_fields(self):
        """Test structured generation for fields."""
        client = MockLLMClient()
        
        result = client.generate_structured(
            "Extract fields",
            {"properties": {"fields": {}}}
        )
        
        assert "fields" in result
        assert isinstance(result["fields"], dict)
    
    def test_generate_structured_answer(self):
        """Test structured generation for QA."""
        client = MockLLMClient()
        
        result = client.generate_structured(
            "Answer question",
            {"properties": {"answer": {}}}
        )
        
        assert "answer" in result
        assert "confidence" in result


class TestTools:
    """Tests for LLM-powered tools with mock client."""
    
    def setup_method(self):
        self.mock_llm = MockLLMClient()
        self.sample_doc = ExtractedDocument(
            file_path="test.txt",
            file_type="txt",
            raw_text="Name: John Doe\nAmount: $1,000\nDate: 2024-01-15"
        )
    
    def test_field_extraction_tool(self):
        """Test field extraction tool."""
        tool = FieldExtractionTool(self.mock_llm)
        
        result = tool.run(self.sample_doc)
        
        assert result.fields is not None
        assert len(self.mock_llm.call_history) == 1
    
    def test_qa_tool(self):
        """Test question answering tool."""
        tool = QuestionAnsweringTool(self.mock_llm)
        
        result = tool.run(
            question="What is the name?",
            document=self.sample_doc
        )
        
        assert result.answer is not None
        assert result.confidence is not None
    
    def test_summarization_tool(self):
        """Test summarization tool."""
        tool = SummarizationTool(self.mock_llm)
        
        result = tool.run(self.sample_doc)
        
        assert result.summary is not None
        assert result.key_points is not None
    
    def test_cross_document_tool(self):
        """Test cross-document analysis tool."""
        tool = CrossDocumentAnalysisTool(self.mock_llm)
        
        docs = [self.sample_doc, self.sample_doc]
        fields_list = [{"Name": "John"}, {"Name": "Jane"}]
        
        result = tool.run(
            question="Compare the names",
            documents=docs,
            extracted_fields_list=fields_list
        )
        
        assert result.answer is not None
        assert result.insights is not None


class TestToolDefinitions:
    """Test tool definition schemas."""
    
    def test_extraction_tool_schema(self):
        """Test extraction tool has valid schema."""
        tool = FieldExtractionTool(MockLLMClient())
        schema = tool.get_input_schema()
        
        assert schema["type"] == "object"
        assert "document_text" in schema["properties"]
    
    def test_qa_tool_schema(self):
        """Test QA tool has valid schema."""
        tool = QuestionAnsweringTool(MockLLMClient())
        schema = tool.get_input_schema()
        
        assert "question" in schema["properties"]
        assert "document_text" in schema["properties"]


class TestAgentIntegration:
    """Integration tests with mock LLM."""
    
    def create_test_agent(self):
        """Create an agent with mock LLM for testing."""
        # We'll need to patch the agent to use mock LLM
        # For now, test the components work together
        pass
    
    def test_process_form_flow(self):
        """Test the form processing flow."""
        processor = DocumentProcessor()
        mock_llm = MockLLMClient()
        extraction_tool = FieldExtractionTool(mock_llm)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Employee: John\nSalary: $50,000")
            temp_path = f.name
        
        try:
            # Process document
            doc = processor.process(temp_path)
            
            # Extract fields via LLM
            result = extraction_tool.run(doc)
            
            assert result.fields is not None
            assert len(mock_llm.call_history) == 1
        finally:
            os.unlink(temp_path)
    
    def test_qa_flow(self):
        """Test the QA flow."""
        processor = DocumentProcessor()
        mock_llm = MockLLMClient()
        qa_tool = QuestionAnsweringTool(mock_llm)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Total: $1,500")
            temp_path = f.name
        
        try:
            doc = processor.process(temp_path)
            
            result = qa_tool.run(
                question="What is the total?",
                document=doc
            )
            
            assert result.answer is not None
        finally:
            os.unlink(temp_path)


class TestExampleScenarios:
    """Test the three required example scenarios."""
    
    def setup_method(self):
        self.mock_llm = MockLLMClient()
        self.processor = DocumentProcessor()
    
    def test_example1_single_form_qa(self):
        """
        Example 1: Answering a question from a single form.
        """
        # Create a W-2 form
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Form W-2 Wage and Tax Statement
            Employee Name: John Smith
            Wages: $75,000
            Federal Tax Withheld: $11,250
            """)
            temp_path = f.name
        
        try:
            # Process
            doc = self.processor.process(temp_path)
            
            # Ask question
            qa_tool = QuestionAnsweringTool(self.mock_llm)
            result = qa_tool.run(
                question="What is the employee's name?",
                document=doc
            )
            
            assert result.answer is not None
            assert result.confidence is not None
            print(f"Example 1 - Answer: {result.answer}")
        finally:
            os.unlink(temp_path)
    
    def test_example2_summarization(self):
        """
        Example 2: Generating a summary of one form.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Insurance Claim Form
            Patient: Jane Doe
            Date of Service: 01/15/2024
            Total Charges: $450.00
            """)
            temp_path = f.name
        
        try:
            doc = self.processor.process(temp_path)
            
            summary_tool = SummarizationTool(self.mock_llm)
            result = summary_tool.run(doc)
            
            assert result.summary is not None
            assert result.key_points is not None
            print(f"Example 2 - Summary: {result.summary}")
        finally:
            os.unlink(temp_path)
    
    def test_example3_cross_form_analysis(self):
        """
        Example 3: Providing a holistic answer across multiple forms.
        """
        temp_files = []
        
        try:
            # Create multiple forms
            for name, salary in [("Alice", "110000"), ("Bob", "80000"), ("Carol", "65000")]:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(f"Employee: {name}\nSalary: ${salary}")
                    temp_files.append(f.name)
            
            # Process all
            docs = [self.processor.process(p) for p in temp_files]
            
            # Analyze
            analysis_tool = CrossDocumentAnalysisTool(self.mock_llm)
            result = analysis_tool.run(
                question="What is the average salary?",
                documents=docs,
                extracted_fields_list=[
                    {"Employee": "Alice", "Salary": "$110,000"},
                    {"Employee": "Bob", "Salary": "$80,000"},
                    {"Employee": "Carol", "Salary": "$65,000"}
                ]
            )
            
            assert result.answer is not None
            assert result.insights is not None
            print(f"Example 3 - Analysis: {result.answer}")
        finally:
            for p in temp_files:
                os.unlink(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
