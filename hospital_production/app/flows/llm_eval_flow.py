from prefect import flow, task
from app.llm.prompts import register_all_prompts
from app.llm.eval import run_eval
from app.flows._metrics import LLM_EVAL_RUN

"""Prefect eval pipeline"""

@task(name="register-prompts")
def t_register():
    return register_all_prompts()

@task(name="eval-hf")
def t_eval_hf():
    return run_eval(model_name="huggingface")

@task(name="eval-groq")
def t_eval_groq():
    return run_eval(model_name="groq")

@task(name="eval-ninerouter")
def t_eval_ninerouter():
    return run_eval(model_name="ninerouter")

@flow(name="llm-observability-flow", log_prints=True)
def llm_eval_flow():
    LLM_EVAL_RUN.inc()  # increment Prometheus counter
    t_register()
    t_eval_hf()
    t_eval_groq()
    t_eval_ninerouter()
    print("✅ LLM eval complete. Compare in MLflow UI → Experiments → Runs.")

if __name__ == "__main__":
    """Run weekly via Prefect deployment to catch prompt/model regressions."""
    llm_eval_flow()