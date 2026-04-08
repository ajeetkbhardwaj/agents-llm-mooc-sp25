import os
import json
from typing import List, Dict, Optional, Union


class MultiAPI_Agent:
    def __init__(self, model: str = "qwen2.5-coder:7b", provider: str = "ollama"):
        self.model = model
        self.provider = provider
        self._setup_client()

    def _setup_client(self):
        if self.provider == "ollama":
            try:
                import ollama

                self.client = ollama
                self._use_ollama = True
            except ImportError:
                print("Ollama not installed, falling back to OpenAI")
                self.provider = "openai"
                self._setup_client()
        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
            self._use_ollama = False
        elif self.provider == "openrouter":
            from openai import OpenAI

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/v1")
            self._use_ollama = False
        elif self.provider == "gemini":
            import google.generativeai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            genai.configure(api_key=api_key)
            self.client = genai
            self._use_ollama = False
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        if self._use_ollama:
            return self._get_ollama_response(messages)
        elif self.provider == "openai" or self.provider == "openrouter":
            return self._get_openai_response(messages)
        elif self.provider == "gemini":
            return self._get_gemini_response(messages)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _get_ollama_response(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._messages_to_prompt(messages)
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_ctx": 16000,
            },
        )
        return response.response

    def _get_openai_response(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return completion.choices[0].message.content

    def _get_gemini_response(self, messages: List[Dict[str, str]]) -> str:
        prompt = self._messages_to_prompt(messages)
        model = self.client.GenerativeModel(self.model)
        response = model.generate_content(prompt)
        return response.text

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt


class LLM_Agent(MultiAPI_Agent):
    def __init__(self, model: str = "qwen2.5-coder:7b", provider: str = "ollama"):
        super().__init__(model=model, provider=provider)


class Reasoning_Agent(MultiAPI_Agent):
    def __init__(self, model: str = "qwen2.5-coder:7b", provider: str = "ollama"):
        super().__init__(model=model, provider=provider)


def get_best_agent(
    model: Optional[str] = None, provider: Optional[str] = None
) -> MultiAPI_Agent:
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")
    if model is None:
        if provider == "ollama":
            model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        elif provider == "openai":
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
        elif provider == "openrouter":
            model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
        elif provider == "gemini":
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return MultiAPI_Agent(model=model, provider=provider)


if __name__ == "__main__":
    agent = LLM_Agent(model="qwen2.5-coder:7b", provider="ollama")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant specialized in Lean 4.",
        },
        {
            "role": "user",
            "content": "Write a simple Lean 4 function that doubles a number.",
        },
    ]
    response = agent.get_response(messages)
    print(response)
