from typing import Optional
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from langsmith.wrappers import wrap_openai

class ModelClients:
    """Simple class to manage different model clients"""
    
    def __init__(self, use_langsmith: bool = False):
        load_dotenv()
        self.use_langsmith = use_langsmith
        
        # Initialize clients as None
        self.openai_client = None
        self.gemini_client = None
        self.anthropic_client = None
        
        # Setup LangSmith if enabled (only affects OpenAI)
        if use_langsmith:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default_project")
    
    def get_openai(self, api_key: Optional[str] = None) -> OpenAI:
        """Get OpenAI client with optional LangSmith wrapping"""
        if not self.openai_client:
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            # LangSmith wrapping only available for OpenAI
            if self.use_langsmith:
                client = wrap_openai(client)
            self.openai_client = client
        return self.openai_client
    
    def get_gemini(self, api_key: Optional[str] = None) -> genai:
        """Get Gemini client (no direct LangSmith support)"""
        if not self.gemini_client:
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            self.gemini_client = genai
        return self.gemini_client
    
    def get_anthropic(self, api_key: Optional[str] = None) -> Anthropic:
        """Get Anthropic client (no direct LangSmith support)"""
        if not self.anthropic_client:
            self.anthropic_client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        return self.anthropic_client

# Usage example:
# model_clients = ModelClients(use_langsmith=True)  # LangSmith will only affect OpenAI
# openai = model_clients.get_openai()  # Will be wrapped with LangSmith
# gemini = model_clients.get_gemini()  # Standard client, no LangSmith wrapping
# anthropic = model_clients.get_anthropic()  # Standard client, no LangSmith wrapping