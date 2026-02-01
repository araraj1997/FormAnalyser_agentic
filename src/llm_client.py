"""
LLM Client Module

Provides a unified interface for LLM interactions using Claude API.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system: str = None, **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict, system: str = None, **kwargs) -> Dict:
        """Generate a structured JSON response."""
        pass


class ClaudeClient(BaseLLMClient):
    """
    Claude API client for LLM operations.
    
    Uses Anthropic's Claude models for intelligent form processing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0
    ):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from Claude.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            LLMResponse object
        """
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            system=system or "You are a helpful assistant.",
            messages=messages,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            raw_response=response
        )
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict,
        system: str = None,
        **kwargs
    ) -> Dict:
        """
        Generate a structured JSON response.
        
        Args:
            prompt: User prompt
            schema: JSON schema for expected output
            system: System prompt
            
        Returns:
            Parsed JSON dictionary
        """
        schema_str = json.dumps(schema, indent=2)
        
        structured_system = f"""{system or "You are a helpful assistant."}

You must respond with valid JSON that matches this schema:
{schema_str}

Respond ONLY with the JSON object, no other text or markdown formatting."""

        response = self.generate(prompt, system=structured_system, **kwargs)
        
        # Parse JSON from response
        content = response.content.strip()
        
        # Handle potential markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON response: {e}\nContent: {content}")
    
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system: str = None,
        **kwargs
    ) -> Dict:
        """
        Generate a response with tool use capability.
        
        Args:
            prompt: User prompt
            tools: List of tool definitions
            system: System prompt
            
        Returns:
            Response with potential tool calls
        """
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You are a helpful assistant.",
            messages=messages,
            tools=tools,
            **kwargs
        )
        
        result = {
            "content": None,
            "tool_calls": [],
            "stop_reason": response.stop_reason
        }
        
        for block in response.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        return result


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API calls.
    """
    
    def __init__(self):
        self.call_history = []
    
    def generate(self, prompt: str, system: str = None, **kwargs) -> LLMResponse:
        self.call_history.append({"type": "generate", "prompt": prompt, "system": system})
        return LLMResponse(
            content="Mock response for testing",
            model="mock-model",
            usage={"input_tokens": 100, "output_tokens": 50}
        )
    
    def generate_structured(self, prompt: str, schema: Dict, system: str = None, **kwargs) -> Dict:
        self.call_history.append({"type": "structured", "prompt": prompt, "schema": schema})
        # Return a mock response matching common schemas
        if "fields" in str(schema):
            return {"fields": {"Name": "John Doe", "Amount": "$1,000"}}
        if "answer" in str(schema):
            return {"answer": "Mock answer", "confidence": 0.9}
        if "summary" in str(schema):
            return {"summary": "Mock summary", "key_points": ["Point 1", "Point 2"]}
        return {}


def get_llm_client(
    provider: str = "claude",
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: LLM provider ("claude" or "mock")
        api_key: API key for the provider
        **kwargs: Additional arguments for the client
        
    Returns:
        LLM client instance
    """
    if provider == "claude":
        return ClaudeClient(api_key=api_key, **kwargs)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
