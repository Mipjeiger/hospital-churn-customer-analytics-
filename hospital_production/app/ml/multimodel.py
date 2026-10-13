import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from app.ml.preprocess import build_preprocessor

"""Multi-model trainer for all 5 concepts ML Models system design."""

# ---- Model Zoos ----
CLF_ZOO = {
    "logistic": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced", min_samples_split=5, min_samples_leaf=2),
    "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=300, random_state=42),
    "xgboost": lambda: XGBClassifier(n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
}

REG_ZOO = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "decision_tree": lambda: DecisionTreeRegressor(max_depth=12, min_samples_leaf=2, random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "xgboost": lambda: XGBRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1, subsample=0.9, colsample_bytree=0.9, tree_method='hist')
}

# ---- Metrics ----
def _clf_metrics(y, p, prob=None):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, p, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if prob is not None else None
    }

def _reg_metrics(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "mse": float(mean_squared_error(y, p)),
        "rmse": float(root_mean_squared_error(y, p)),
        "r2": float(r2_score(y, p))
    }

# --- Core multi-model training function ---
def train_multimodel(X_train, y_train, X_test, y_test, task="classification", zoo=None):
    """
    task: 'classification' | 'regression'
    Returns: {name: {'pipeline': ...., 'metrics': ...}}
    """
    zoo = zoo or (CLF_ZOO if task == "classification" else REG_ZOO)

    # Append the preprocessor to the model pipeline
    results = {}
    for name, factory in zoo.items():
        pipe = Pipeline([("pre", build_preprocessor(X_train)), ("est", factory())])
        pipe.fit(X_train, y_train) 
        p = pipe.predict(X_test)

        if task == "classification":
            prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None
            metrics = _clf_metrics(y_test, p, prob)
        else:
            metrics = _reg_metrics(y_test, p)
        results[name] = {"pipeline": pipe, "metrics": metrics}

    return results

def pick_best(results, task):
    key = "roc_auc" if task == "classification" else "r2"
    return max(results.items(), key=lambda x: x[1]["metrics"][key])