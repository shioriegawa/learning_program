import os
import json
from typing import Dict, Any
from prompts import get_runtime_prompt

class LLMClient:
    """Abstract LLM client interface"""
    
    def generate_sql(self, user_query: str) -> Dict[str, Any]:
        """Generate SQL from user query"""
        raise NotImplementedError

class OpenAIClient(LLMClient):
    """OpenAI GPT client"""
    
    def __init__(self):
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: uv add openai")
    
    def generate_sql(self, user_query: str) -> Dict[str, Any]:
        """Generate SQL using OpenAI GPT"""
        try:
            prompt = get_runtime_prompt(user_query)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            return {
                "success": True,
                "sql": result.get("sql", ""),
                "chart_suggestion": result.get("chart_suggestion", "table"),
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "sql": "",
                "chart_suggestion": "table",
                "error": f"Invalid JSON response: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "sql": "",
                "chart_suggestion": "table", 
                "error": f"OpenAI API error: {str(e)}"
            }

class AnthropicClient(LLMClient):
    """Anthropic Claude client"""
    
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: uv add anthropic")
    
    def generate_sql(self, user_query: str) -> Dict[str, Any]:
        """Generate SQL using Anthropic Claude"""
        try:
            prompt = get_runtime_prompt(user_query)
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Extract JSON from response (in case there's extra text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
            else:
                json_content = content
            
            result = json.loads(json_content)
            
            return {
                "success": True,
                "sql": result.get("sql", ""),
                "chart_suggestion": result.get("chart_suggestion", "table"),
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "sql": "",
                "chart_suggestion": "table",
                "error": f"Invalid JSON response: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "sql": "",
                "chart_suggestion": "table",
                "error": f"Anthropic API error: {str(e)}"
            }

def get_llm_client() -> LLMClient:
    """Get LLM client based on environment variable"""
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if provider == 'anthropic':
        if not os.getenv('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicClient()
    else:
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient()

def generate_sql_from_query(user_query: str) -> Dict[str, Any]:
    """Main function to generate SQL from user query"""
    try:
        client = get_llm_client()
        return client.generate_sql(user_query)
    except Exception as e:
        return {
            "success": False,
            "sql": "",
            "chart_suggestion": "table",
            "error": f"LLM client error: {str(e)}"
        }