import mlflow
import logging
from app.api.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

"""MLflow Prompt Registry"""

# --- Prompt templates for LLM ---
PROMPTS = {
    "hospital_agent_system": {
        "template": """You are an AI Hospital Operations Analyst for a management system.

Capabilities:
- Answer questions about patients, services, churn, beds, staff morale.
- Call tools when data is needed. NEVER invent numbers.
- After tool results, give a short, decision-oriented answer.
- Use bullet points, percentages, and concise clinical language.

Role context: {{ role }}
End with a one-line actionable recommendation when relevant.""",
        "tags": {"team": "hospital-ai", "version": "v1"},
    },
    "churn_insight_generator": {
        "template": """Given this patient cohort: {{ cohort_json }}

Write a concise executive memo (max 180 words) covering:
1. Key risk drivers visible in this cohort.
2. Immediate 2-3 operational actions.
3. What data to monitor weekly.""",
        "tags": {"team": "hospital-ai", "version": "v1"},
    },
    "service_performance_summary": {
        "template": """Service metrics: {{ metrics_json }}
Summarize in 3 bullets: top performer, weakest signal, recommended next step.""",
        "tags": {"team": "hospital-ai", "version": "v1"},
    },
}

def register_all_prompts():
    """Push each prompt to MLFlow prompt registry."""
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

    registered = {}
    for name, p in PROMPTS.items():
        try:
            v = mlflow.genai.register_prompt(
                name=name,
                template=p["template"],
                tags=p["tags"]
            )
            registered[name] = v
            logger.info(f"✅ Registered prompt '{name}' with version {v.version}")
        except Exception as e:
            logger.error(f"❌ Failed to register prompt '{name}': {e}")

    return registered

def load_prompt(name: str, version: int = None) -> str:
    p = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{name}/{version or 'latest'}")
    return p.template

def render_prompt(name: str, **kwargs) -> str:
    """Render a registered prompt with variables."""
    tmpl = load_prompt(name)
    for k, v in kwargs.items():
        tmpl = tmpl.replace("{{ " + k + " }}", str(v))
    return tmpl