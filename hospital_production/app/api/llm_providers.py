import os
import pandas as pd
import numpy as np
import logging

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from api.config import settings
from openai import OpenAI
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

# Define literal ML Models
MODELS_KEYS = ['logistic_regression', 'gradient_boosting', 'random_forest', 'xgboost', 'decision_tree']

# Build system prompt for LLM
SYSTEM_PROMPT = """You are an AI Hospital Operations Analyst for a hospital management system.

Capabilities:
- Answer questions about patients, services, churn, beds, staff morale, fraud.
- Call tools when data is needed. NEVER invent numbers.
- After tool results, give a short, clear, decision-oriented answer.
- Use bullet points, percentages, and concise clinical language.
- End with a one-line actionable recommendation when relevant.
"""

class LLMClient:
    """Setup LLM Client for inference using HuggingFace Inference API, Groq API, or NineRouter API based on the environment variables."""
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.hf_api_key = settings.HUGGINGFACE_API_KEY
        self.groq_api_key = settings.GROQ_API_KEY
        self.ninerouter_base_url = settings.NINEROUTER_BASE_URL
        self.ninerouter_api_key = settings.NINEROUTER_API_KEY

        # Load config json
        self.role_config = settings.ROLE_CONFIG if settings.ROLE_CONFIG.exists() else None

        # Load database
        self.data = settings.DATA_PATH if settings.DATA_PATH.exists() else None

        # Configure the API key
        if self.hf_api_key:
            self.client = InferenceClient(token=self.hf_api_key) if self.hf_api_key else None

        if self.groq_api_key:
            self.groq_client = InferenceClient(token=self.groq_api_key) if self.groq_api_key else None

        if self.ninerouter_api_key:
            self.ninerouter_client = OpenAI(api_key=self.ninerouter_api_key) if self.ninerouter_api_key else None

    # --- Load ML Models ---
    def load_models(self):
        """Load ML models from the specified path."""
        self.models = {}
        for model_key in MODELS_KEYS:
            model_path = settings.ML_MODEL_PATH / f"{model_key}.joblib"

            if model_path.exists():
                try:
                    self.models[model_key] = model_path
                    logger.info(f"Loaded model: {model_key} from {model_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading model {model_key}: {e}")

        return self.models

    # --- ML Scoring ---
    def score(self, df:pd.DataFrame,  row: pd.Series, model_name: str = None,) -> dict[str, Any]:
        """Scoring by exists ML models retriveved from the Path to fetch as LLM agentic Answer."""
        if model_name not in self.models:
            raise KeyError(f"❌ Model '{model_name}' not found. Available models: {list(self.models.keys())}")

        # Ensure row is a PD.Series
        if isinstance(row, dict):
            row = pd.Series(row)

        # Encode the row
        label_encoders = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        # Predict using the model
        model = self.models[model_name]
        pred = model.predict(row.values.reshape(1, -1))
        proba = model.predict_proba(row.values.reshape(1, -1))

        return {
            "model_user": model_name,
            "prediction": int(pred[0]),
            "probability": proba[0].tolist()
        }

    # ---- LLM Inference ----
    def llm(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature=0.3):
        """LLM Inference using HuggingFace, Groq, or NineRouter API based on the environment variables."""

        # Try HuggingFace Inference API
        if self.hf_client:
            kwargs = dict(
                model="Qwen/Qwen2.5-Coder-32B-Instruct",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = self.hf_client.chat(**kwargs)
            msg = response.choices[0].message
            print(f"🧑‍💻 Using Huggingface")

            return {
                "text": msg.content,
                "role": msg.role,
                "provider": "HuggingFace"
            }            

        # Second fallback to Groq Inference API
        if self.groq_client:
            kwargs = dict(
                model="openai/gpt-oss-20b",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = self.groq_client.chat(**kwargs)
            msg = response.choices[0].message
            print(f"🧑‍💻 Fallback to using Groq")

            return {
                "text": msg.content,
                "role": msg.role,
                "provider": "Groq"
            }

        # Third fallback to NineRouter Inference API
        if self.ninerouter_client:
            kwargs = dict(
                model="sragent",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = self.ninerouter_client.chat(**kwargs)
            msg = response.choices[0].message
            print(f"🧑‍💻 Fallback to using NineRouter")

            return {
                "text": msg.content,
                "role": msg.role,
                "provider": "NineRouter"
            }

        raise RuntimeError("❌ No LLM provider is configured. Please set the appropriate API keys in the environment variables.")

    # --- Build Agentic AI ---
    def agentic_ai(self, user_query: str, role: str) -> dict[str, Any]:
        """Agentic AI for Hospital Operations using LLM and ML models retrieval"""

        # Load ML models
        if self.load_models():
            logging.info(f"✅ Loaded ML models: {list(self.models.keys())}")

        # Build messages for LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        # Insert role-specific instructions if available
        if self.role_config and role in self.role_config:
            role_instructions = self.role_config[role]
            messages.append({"role": "system", "content": role_instructions})

        # Call LLM for inference
        llm_response = self.llm(messages)
        return {
            "user_query": user_query,
            "role": role,
            "llm_response": llm_response,
            "provider": llm_response.get("provider", "Unknown")
        }

# Usage LLM
llm_client = LLMClient()