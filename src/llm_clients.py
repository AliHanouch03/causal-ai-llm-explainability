import os
import re
import time
from google import genai
from groq import Groq
from mistralai.client import Mistral
import cohere
from openai import OpenAI


class NvidiaClient:
    def __init__(self, model_name="nvidia/nemotron-3-super-120b-a12b"):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set in environment")
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.2,  # Low temperature for consistent judging
        )
        text = response.choices[0].message.content
        # Strip any reasoning tags if the model emits them
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

class GeminiClient:
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        # Stay under the free tier 5 RPM limit
        time.sleep(13)
        return response.text


class GroqClient:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        text = response.choices[0].message.content
        # Strip out <think>...</think> blocks from reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


class MistralClient:
    def __init__(self, model_name="mistral-small-latest"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment")
        # Increase timeout to 120s — Magistral is a reasoning model and can be slow
        self.client = Mistral(api_key=api_key, timeout_ms=120000)
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content

        # If content is a string, return it directly
        if isinstance(content, str):
            return content.replace("${response}", "").strip()

        # If content is a list (reasoning model), extract only the final text chunks
        text_parts = []
        for chunk in content:
            if hasattr(chunk, "type") and chunk.type == "text":
                text_parts.append(chunk.text)

        result = "\n".join(text_parts)
        return result.replace("${response}", "").strip()


class CohereClient:
    def __init__(self, model_name="command-r-plus"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set in environment")
        self.client = cohere.ClientV2(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt):
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content[0].text


# Full registry of available LLM clients (judges + candidates)
LLM_REGISTRY = {
    "gemini-3-flash-preview": lambda: GeminiClient("gemini-3-flash-preview"),
    "gemini-2.5-flash": lambda: GeminiClient("gemini-2.5-flash"),
    "llama-3.3-70b": lambda: GroqClient("llama-3.3-70b-versatile"),
    "qwen3-32b": lambda: GroqClient("qwen/qwen3-32b"),
    "magistral-medium": lambda: MistralClient("magistral-medium-latest"),
    "c4ai-aya-expanse-32b": lambda: CohereClient("c4ai-aya-expanse-32b"),
    "nemotron-3-super": lambda: NvidiaClient("nvidia/nemotron-3-super-120b-a12b"),
}

# Models used as candidates being evaluated (excludes the judge)
EVALUATED_LLMS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "llama-3.3-70b",
    "qwen3-32b",
    "magistral-medium",
    "c4ai-aya-expanse-32b",
]


def get_client(name):
    if name not in LLM_REGISTRY:
        raise ValueError(f"Unknown LLM: {name}. Available: {list(LLM_REGISTRY.keys())}")
    return LLM_REGISTRY[name]()


def list_llms():
    """Return only the LLMs that are evaluated (excludes the judge)."""
    return EVALUATED_LLMS