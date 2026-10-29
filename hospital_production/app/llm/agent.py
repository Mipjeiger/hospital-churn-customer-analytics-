from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / ".env")

import json
import logging
from huggingface_hub import InferenceClient
from groq import Groq
from openai import OpenAI
from app.api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Langfused traced LLM agent"""

# Hunggingface Configuration
hf = (
    InferenceClient(
        token=settings.HUGGINGFACE_API_KEY,
        base_url="https://router.huggingface.co",
    )
    if settings.HUGGINGFACE_API_KEY
    else None
)
groq = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
ninerouter = OpenAI(api_key=settings.NINEROUTER_API_KEY, base_url=settings.NINEROUTER_BASE_URL) if settings.NINEROUTER_API_KEY else None

# Add function to normalize tool calls
def _coerce_content(c):
    if c is None:
        return None
    if isinstance(c, str):
        return c
    if hasattr(c, "to_csv"):
        return c.to_csv(index=False)
    if hasattr(c, "to_dict"):
        return json.dumps(c.to_dict(), default=str)
    return json.dumps(c, default=str)

# Define allowed keys for messages
_ALLOWED_MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}

def _normalize_messages(messages):
    clean = []
    for m in messages:
        m = dict(m)
        role = m.get("role")
        content = _coerce_content(m.get("content"))

        # Non-assistant messages must have string content
        if role != "assistant" and content is None:
            content = ""

        m["content"] = content
        m = {k: v for k, v in m.items() if k in _ALLOWED_MESSAGE_KEYS}
        clean.append(m)
    return clean

def _call(client, model, messages, tools, max_tokens, temperature):
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)

def llm_router(messages, tools=None, temperature=0.3, max_tokens=512, provider_name="router"):
    """Langfuse traced LLM agent for multiple LLM providers"""
    messages = _normalize_messages(messages)

    providers = [
        ("huggingface", hf, "Qwen/Qwen2.5-Coder-32B-Instruct", _call),
        ("groq", groq, "openai/gpt-oss-20b", _call),
        ("ninerouter", ninerouter, "sragent", _call),
    ]

    valid_providers = {"router", "huggingface", "groq", "ninerouter"}
    if provider_name not in valid_providers:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Choose one of: {sorted(valid_providers)}"
        )

    if provider_name != "router":
        providers = [p for p in providers if p[0] == provider_name]

    for selected_provider, client_obj, model_name, call_fn in providers:
        if client_obj is None:
            continue  # Skip if the client is not configured

        try:
            r = call_fn(client_obj, model_name, messages, tools, max_tokens, temperature)
            message = r.choices[0].message
            content = message.content or ""

            return {
                "text": content,
                "tool_calls": getattr(message, "tool_calls", None),
                "provider": selected_provider
            }

        except Exception:
            logger.exception("%s provider failed", selected_provider)
            
    raise RuntimeError("All LLM providers failed")

def run_agent(user_message, tools_schema, tool_map, history=None, max_steps=5):
    """Langfuse running agent with tools and observation"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI Hospital Operations Analyst.\n\n"
                "Capabilities:\n"
                "- Answer questions about patients, services, churn, beds, staff morale, fraud.\n"
                "- Call tools when data is needed. NEVER invent numbers.\n"
                "- After tool results, give a short, clear, decision-oriented answer.\n"
                "- Use bullet points, percentages, and concise clinical language.\n"
                "- End with a one-line actionable recommendation when relevant."
            )
        }
    ]

    if history:
        messages += history
    messages.append({"role": "user", "content": str(user_message)})

    for _ in range(max_steps):
        r = llm_router(messages, tools=tools_schema)
        tcs = r.get("tool_calls")

        if not tcs:
            return r["text"]

        # Append tool calls to messages
        messages.append({
            "role": "assistant",
            "content": r["text"] or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tcs
            ]
        })

        for tc in tcs:
            args = json.loads(tc.function.arguments or "{}")
            result = tool_map[tc.function.name](**args) if tc.function.name in tool_map else {"error": f"Unknown tool '{tc.function.name}'"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(result, default=str)
            })

    return "Max steps reached."