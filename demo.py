#!/usr/bin/env python3
"""
Demonstration Script for Intelligent Form Agent (Agentic Version)

This script demonstrates the three required example runs:
1. Answering a question from a single form
2. Generating a summary of one form
3. Providing a holistic answer across multiple forms

IMPORTANT: Set your ANTHROPIC_API_KEY environment variable before running.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import IntelligentFormAgent


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---\n")


def check_api_key():
    """Check if API key is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("=" * 70)
        print("  ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nTo run this demo, set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nOr run with:")
        print("  ANTHROPIC_API_KEY='your-key' python demo.py")
        print()
        sys.exit(1)


def demo_single_form_qa(agent: IntelligentFormAgent):
    """
    EXAMPLE 1: Answering a question from a single form.
    
    Demonstrates the LLM-powered QA capability on a W-2 tax form.
    """
    print_header("EXAMPLE 1: Single Form Question Answering (LLM-Powered)")
    
    form_path = "data/sample_forms/sample_w2.txt"
    
    if not os.path.exists(form_path):
        print(f"Sample form not found at {form_path}")
        return
    
    print(f"📄 Loading form: {form_path}")
    print("   [Making LLM call for field extraction...]\n")
    
    form = agent.process_form(form_path)
    
    print_subheader("Extracted Fields (via LLM)")
    print(json.dumps(form.extracted_fields, indent=2))
    print(f"\n✅ Form Type: {form.form_type}")
    print(f"✅ Extraction Confidence: {form.extraction_confidence:.1%}")
    
    print_subheader("Question & Answer (via LLM)")
    
    questions = [
        "What is the employee's name and SSN?",
        "How much were the total wages reported?",
        "What was the total federal tax withheld?",
    ]
    
    for question in questions:
        print(f"\n🔍 Q: {question}")
        print("   [Making LLM call for reasoning...]\n")
        
        result = agent.ask(question, form)
        
        print(f"💬 A: {result.answer}")
        print(f"   Confidence: {result.confidence:.1%}")
        if result.evidence:
            print(f"   Evidence: {result.evidence[0][:100]}...")
    
    print(f"\n📊 LLM calls so far: {agent.total_llm_calls}")


def demo_form_summary(agent: IntelligentFormAgent):
    """
    EXAMPLE 2: Generating a summary of one form.
    
    Demonstrates the LLM-powered summarization capability.
    """
    print_header("EXAMPLE 2: Form Summarization (LLM-Powered)")
    
    form_path = "data/sample_forms/sample_insurance_claim.txt"
    
    if not os.path.exists(form_path):
        print(f"Sample form not found at {form_path}")
        return
    
    print(f"📄 Loading form: {form_path}")
    form = agent.process_form(form_path)
    
    print_subheader("Form Metadata")
    print(f"  Type: {form.form_type}")
    print(f"  Confidence: {form.extraction_confidence:.1%}")
    print(f"  Fields Extracted: {len(form.extracted_fields)}")
    
    print_subheader("Generated Summary (via LLM)")
    print("   [Making LLM call for summarization...]\n")
    
    summary = agent.summarize(form, style="detailed")
    
    print(f"📝 {summary.summary}\n")
    
    print("📌 Key Points:")
    for point in summary.key_points:
        print(f"   • {point}")
    
    print("\n💰 Important Values:")
    for k, v in summary.important_values.items():
        print(f"   • {k}: {v}")
    
    print(f"\n📊 LLM calls so far: {agent.total_llm_calls}")


def demo_cross_form_analysis(agent: IntelligentFormAgent):
    """
    EXAMPLE 3: Providing a holistic answer across multiple forms.
    
    Demonstrates the LLM-powered cross-document analysis.
    """
    print_header("EXAMPLE 3: Cross-Form Analysis (LLM-Powered)")
    
    form_paths = [
        "data/sample_forms/onboarding_1.txt",
        "data/sample_forms/onboarding_2.txt",
        "data/sample_forms/onboarding_3.txt"
    ]
    
    existing_paths = [p for p in form_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("Sample forms not found")
        return
    
    print(f"📄 Loading {len(existing_paths)} employee onboarding forms...")
    forms = agent.process_forms(existing_paths)
    
    print_subheader("Loaded Forms Overview")
    for form in forms:
        name = form.extracted_fields.get("Full Name", "Unknown")
        dept = form.extracted_fields.get("Department", "Unknown")
        salary = form.extracted_fields.get("Annual Salary", "Unknown")
        print(f"  📋 {Path(form.file_path).name}")
        print(f"     Employee: {name}")
        print(f"     Department: {dept}")
        print(f"     Salary: {salary}")
        print()
    
    print_subheader("Cross-Form Analysis Questions (via LLM)")
    
    questions = [
        "What departments are represented and what are their salaries?",
        "What is the average salary across all employees?",
        "Who has the highest salary and what is their position?"
    ]
    
    for question in questions:
        print(f"\n🔍 Q: {question}")
        print("   [Making LLM call for cross-document reasoning...]\n")
        
        result = agent.analyze(question, forms)
        
        print(f"💬 A: {result.answer}")
        
        if result.insights:
            print("\n   💡 Insights:")
            for insight in result.insights[:3]:
                print(f"      • {insight}")
        
        if result.statistics:
            print(f"\n   📈 Statistics: {json.dumps(result.statistics, indent=6)}")
    
    print(f"\n📊 Total LLM calls: {agent.total_llm_calls}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("       INTELLIGENT FORM AGENT - AGENTIC DEMONSTRATION")
    print("         (Powered by Claude LLM API)")
    print("=" * 70)
    
    # Check for API key
    check_api_key()
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Initialize the agent
    print("\n🚀 Initializing Intelligent Form Agent with Claude...")
    agent = IntelligentFormAgent(verbose=True)
    print("✅ Agent initialized!\n")
    
    # Run demonstrations
    try:
        demo_single_form_qa(agent)
        demo_form_summary(agent)
        demo_cross_form_analysis(agent)
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print_header("DEMONSTRATION COMPLETE")
    print("The Intelligent Form Agent demonstrated:")
    print("  ✅ LLM-powered field extraction")
    print("  ✅ LLM-powered question answering with reasoning")
    print("  ✅ LLM-powered summarization")
    print("  ✅ LLM-powered cross-form analysis")
    print(f"\n📊 Total LLM API calls made: {agent.total_llm_calls}")
    print("\nTo use the web interface:")
    print("  $ streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
