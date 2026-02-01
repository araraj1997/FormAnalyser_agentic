#!/usr/bin/env python3
"""
Command Line Interface for Intelligent Form Agent

Provides command-line access to all agent capabilities.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import IntelligentFormAgent, ProcessedForm


def get_console():
    """Get console for output."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_output(message: str, style: str = None):
    """Print with optional rich formatting."""
    console = get_console()
    if console and style:
        console.print(message, style=style)
    else:
        print(message)


def print_json(data: dict):
    """Print JSON data."""
    console = get_console()
    if console:
        console.print_json(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, indent=2, default=str))


def print_panel(content: str, title: str = None):
    """Print a panel."""
    console = get_console()
    if console:
        console.print(Panel(content, title=title))
    else:
        print(f"\n{'='*50}")
        if title:
            print(f"  {title}")
            print(f"{'='*50}")
        print(content)
        print(f"{'='*50}\n")


def cmd_process(args):
    """Process forms and extract fields."""
    agent = IntelligentFormAgent(api_key=args.api_key, verbose=args.verbose)
    
    forms = []
    for file_path in args.files:
        print_output(f"Processing: {file_path}", "blue")
        form = agent.process_form(file_path)
        forms.append(form)
        
        print_output(f"  Form Type: {form.form_type}", "green")
        print_output(f"  Confidence: {form.extraction_confidence:.1%}", "green")
        print_output(f"  Fields Extracted: {len(form.extracted_fields)}", "green")
    
    if args.output:
        output_data = [f.to_dict() for f in forms]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print_output(f"\nSaved to: {args.output}", "blue")
    else:
        for form in forms:
            print_panel(
                json.dumps(form.extracted_fields, indent=2, default=str),
                title=f"Extracted Fields: {form.file_path}"
            )
    
    print_output(f"\nTotal LLM calls: {agent.total_llm_calls}", "dim")


def cmd_ask(args):
    """Ask a question about forms."""
    agent = IntelligentFormAgent(api_key=args.api_key, verbose=args.verbose)
    
    # Process forms
    forms = [agent.process_form(f) for f in args.files]
    
    # Ask question
    if len(forms) == 1:
        result = agent.ask(args.question, forms[0])
        
        print_panel(result.answer, title="Answer")
        print_output(f"Confidence: {result.confidence:.1%}", "green" if result.confidence > 0.7 else "yellow")
        
        if result.evidence:
            print_output("\nEvidence:", "blue")
            for e in result.evidence:
                print_output(f"  • {e}")
        
        if args.verbose and result.reasoning:
            print_output(f"\nReasoning: {result.reasoning}", "dim")
    else:
        result = agent.analyze(args.question, forms)
        
        print_panel(result.answer, title="Answer")
        
        if result.insights:
            print_output("\nInsights:", "blue")
            for insight in result.insights:
                print_output(f"  💡 {insight}")
        
        if result.statistics:
            print_output("\nStatistics:", "blue")
            print_json(result.statistics)
    
    print_output(f"\nTotal LLM calls: {agent.total_llm_calls}", "dim")


def cmd_summarize(args):
    """Generate summaries of forms."""
    agent = IntelligentFormAgent(api_key=args.api_key, verbose=args.verbose)
    
    for file_path in args.files:
        form = agent.process_form(file_path)
        summary = agent.summarize(form, style=args.style)
        
        print_panel(summary.summary, title=f"Summary: {file_path}")
        
        print_output("\nKey Points:", "blue")
        for point in summary.key_points:
            print_output(f"  • {point}")
        
        print_output(f"\nForm Type: {summary.form_type}", "green")
        
        if summary.important_values:
            print_output("\nImportant Values:", "blue")
            for k, v in summary.important_values.items():
                print_output(f"  {k}: {v}")
        
        print_output("")
    
    print_output(f"Total LLM calls: {agent.total_llm_calls}", "dim")


def cmd_analyze(args):
    """Analyze multiple forms."""
    agent = IntelligentFormAgent(api_key=args.api_key, verbose=args.verbose)
    
    # Process all forms
    print_output(f"Processing {len(args.files)} forms...", "blue")
    forms = [agent.process_form(f) for f in args.files]
    
    # Run analysis
    result = agent.analyze(args.question, forms)
    
    print_panel(result.answer, title="Analysis Result")
    
    if result.insights:
        print_output("\n📊 Insights:", "blue bold")
        for insight in result.insights:
            print_output(f"  💡 {insight}")
    
    if result.statistics:
        print_output("\n📈 Statistics:", "blue bold")
        print_json(result.statistics)
    
    if result.comparisons:
        print_output("\n🔄 Comparisons:", "blue bold")
        print_json(result.comparisons)
    
    print_output(f"\nTotal LLM calls: {agent.total_llm_calls}", "dim")


def cmd_workflow(args):
    """Run a complete workflow."""
    agent = IntelligentFormAgent(api_key=args.api_key, verbose=args.verbose)
    
    print_output(f"Running workflow: {args.task}", "blue bold")
    
    result = agent.run_workflow(
        task=args.task,
        file_paths=args.files,
        question=args.question
    )
    
    print_output(f"\nWorkflow Type: {result['type']}", "green")
    print_json(result)
    
    print_output(f"\nTotal LLM calls: {agent.total_llm_calls}", "dim")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Form Agent - LLM-powered form processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract fields from a form
  python cli.py process tax_form.pdf
  
  # Ask a question
  python cli.py ask -f tax_form.pdf -q "What is the total income?"
  
  # Summarize forms
  python cli.py summarize form1.pdf form2.pdf
  
  # Cross-form analysis
  python cli.py analyze -f form1.pdf -f form2.pdf -q "Compare the salaries"
  
  # Run a workflow
  python cli.py workflow -t "Extract and summarize" -f form.pdf
"""
    )
    
    parser.add_argument("--api-key", "-k", help="Anthropic API key")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process forms and extract fields")
    process_parser.add_argument("files", nargs="+", help="Form files to process")
    process_parser.add_argument("--output", "-o", help="Output JSON file")
    process_parser.set_defaults(func=cmd_process)
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about forms")
    ask_parser.add_argument("--files", "-f", nargs="+", required=True, help="Form files")
    ask_parser.add_argument("--question", "-q", required=True, help="Question to ask")
    ask_parser.set_defaults(func=cmd_ask)
    
    # Summarize command
    sum_parser = subparsers.add_parser("summarize", help="Summarize forms")
    sum_parser.add_argument("files", nargs="+", help="Form files to summarize")
    sum_parser.add_argument("--style", "-s", default="detailed",
                           choices=["brief", "detailed", "bullet_points"],
                           help="Summary style")
    sum_parser.set_defaults(func=cmd_summarize)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze multiple forms")
    analyze_parser.add_argument("--files", "-f", nargs="+", required=True, help="Form files")
    analyze_parser.add_argument("--question", "-q", required=True, help="Analysis question")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run a complete workflow")
    workflow_parser.add_argument("--task", "-t", required=True, help="Task description")
    workflow_parser.add_argument("--files", "-f", nargs="+", required=True, help="Form files")
    workflow_parser.add_argument("--question", "-q", help="Optional question")
    workflow_parser.set_defaults(func=cmd_workflow)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check for API key
    if not args.api_key and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: API key required. Set ANTHROPIC_API_KEY or use --api-key")
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
