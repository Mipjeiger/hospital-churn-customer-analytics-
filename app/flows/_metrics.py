from prometheus_client import Counter

# Counters for Prefect flow executions
TRAIN_RUN = Counter('prefect_train_runs_total','Prefect train flow runs')
PREDICT_RUN = Counter('prefect_predict_runs_total','Prefect predict flow runs')
LLM_EVAL_RUN = Counter('prefect_llm_eval_runs_total','Prefect LLM eval runs')
