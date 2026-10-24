import time
import mlflow
from mlflow.entities import SpanType
from app.api.config import settings
from app.llm.agent_langfuse import llm_router

"""Instrument LLM calls as MLflow Traces"""

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

# ========== Traced LLM router ==========
@mlflow.trace(span_type=SpanType.LLM, name="llm_router")
def traced_llm(messages, tools=None, temperature=0.3, max_tokens=512, model_name="router"):
    """MLflow traced LLM router"""
    start = time.time()

    # Define the provider models which is integrated to LLM
    out = llm_router(messages, 
                     tools=tools, 
                     temperature=temperature, 
                     max_tokens=max_tokens, 
                     provider_name=model_name)
    
    latency = time.time() - start

    # Attach metadata to current span
    mlflow.update_current_trace(
        tags={
            "provider": out["provider"],
            "has_tools": str(tools is not None)
        },
        metadata={
            "request_preview": str(messages[-1]["content"][:500]),
            "response_preview": str(out["text"][:500]),
        }
    )
    mlflow.log_metric("latency", latency)
    return out

# ========= Traced LLM agent ==========
@mlflow.trace(span_type=SpanType.AGENT, name="hospital_agent")
def traced_agent(user_message, tools_schema, tool_map, history=None, max_steps=5):
    """MLflow traced LLM agent on the JSON schema of tools"""
    import json
    messages = [{"role": "system",
                 "content": "You are an AI Hospital Operations Analyst. "
                            "Call tools for facts. Never invent numbers."}]
    if history:
        messages += history

    messages.append({"role": "user", "content": user_message})

    tool_calls_total = 0
    for step in range(max_steps):
        with mlflow.start_span(name=f"step_{step}", span_type=SpanType.CHAIN) as span:
            span.set_inputs({"messages_len": len(messages)})
            r = traced_llm(messages, tools=tools_schema)
            tcs = r.get("tool_calls")

            if not tcs:
                span.set_outputs({"final": r["text"][:500]})
                mlflow.log_metric({
                    "agent_steps": step + 1,
                    "agent_tool_calls": tool_calls_total
                })
                return r["text"]

            messages.append({"role": "assistant", "content": r["text"] or None,
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in tcs]})

            for tc in tcs:
                with mlflow.start_span(name=f"tool_{tc.function.name}", span_type=SpanType.TOOL) as tool_span:
                    args = json.loads(tc.function.arguments or "{}")
                    tool_span.set_inputs(args)
                    result = tool_map[tc.function.name](**args)
                    tool_span.set_outputs(result)

                    # Count total tool calls for metrics
                    tool_calls_total += 1
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "name": tc.function.name,
                                     "content": json.dumps(result, default=str)})
    
    return "Max steps reached."