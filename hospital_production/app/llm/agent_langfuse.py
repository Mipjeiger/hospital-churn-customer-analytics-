import os
import json
from langfuse import Langfuse, observe, get_client
from huggingface_hub import InferenceClient
from groq import Groq
from openai import OpenAI
from app.api.config import settings

"""Langfused traced LLM agent"""

# Configuration LLM API keys
langfuse = Langfuse(public_key=settings.LANGFUSE_PUBLIC_KEY, secret_key=settings.LANGFUSE_SECRET_KEY, host=settings.LANGFUSE_HOST)
hf = InferenceClient(token=settings.HUGGINGFACE_API_KEY) if settings.HUGGINGFACE_API_KEY else None
groq = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None
ninerouter = OpenAI(api_key=settings.NINEROUTER_API_KEY, base_url=settings.NINEROUTER_BASE_URL) if settings.NINEROUTER_API_KEY else None

# Observe langfuse endpoint LLM agent
@observe(name="llm-router")
def llm_router(messages, tools=None, temperature=0.3, max_tokens=512):
    """Langfuse traced LLM agent for multiple LLM providers"""
    providers = [
        ("huggingface", hf, "Qwen/Qwen2.5-Coder-32B-Instruct", lambda client, model: client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, tools=tools)),
        ("groq", groq, "openai/gpt-oss-20b", lambda client, model: client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, tools=tools)),
        ("ninerouter", ninerouter, "sragent", lambda client, model: client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, tools=tools)),
    ]

    for provider_name, client_obj, model_name, call_fn in providers:
        if not client_obj:
            continue

        try:
            r = call_fn(client_obj, model_name)
            message = r.choices[0].message
            content = message.content or ""

            # Update observation using get_client() SDK method
            get_client().update_observation(
                model=model_name,
                provider={"provider": provider_name},
                output=content
            )

            return {
                "text": content,
                "tool_calls": getattr(message, "tool_calls", None),
                "provider": provider_name
            }

        except Exception as e:
            print(f"⚠️ {provider_name} failed: {e}")
            
    raise RuntimeError("All LLM providers failed")

@observe(name="hospital-agent")
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
    messages.append({"role": "user", "content": user_message})

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