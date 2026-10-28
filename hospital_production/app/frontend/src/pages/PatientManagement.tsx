import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import { LoadingSpinner, UnavailableState } from "../components/ui/States";
import { fmtFloat, fmtPct } from "../lib/fmt";

const VALID_PAT_RE = /^PAT-\w{8}$/;

const ChurnPredictor: React.FC = () => {
  const [pid, setPid] = useState("");
  const [pidError, setPidError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: (patient_id: string) => api.predictChurn(patient_id),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!VALID_PAT_RE.test(pid.trim())) {
      setPidError("Patient ID must match PAT-XXXXXXXX (8 alphanumeric chars)");
      return;
    }
    setPidError(null);
    mutation.mutate(pid.trim());
  };

  return (
    <div className="card">
      <h3 className="card-title">Single-Patient Churn Prediction</h3>
      <p className="card-desc">
        Calls <code>POST /predict/churn</code> with the Feast feature store.
      </p>
      <form onSubmit={handleSubmit} className="form-row">
        <input
          className={`form-input${pidError ? " form-input--error" : ""}`}
          type="text"
          placeholder="PAT-12345678"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
          aria-label="Patient ID"
        />
        <button className="btn btn-primary" type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Predicting…" : "Predict Churn"}
        </button>
      </form>
      {pidError && <p className="field-error">{pidError}</p>}

      {mutation.isPending && <LoadingSpinner label="Calling /predict/churn…" />}

      {mutation.isSuccess && mutation.data && (
        <div className="result-card">
          <div className="result-row">
            <span>Patient ID</span>
            <strong>{mutation.data.patient_id}</strong>
          </div>
          <div className="result-row">
            <span>Churn Probability</span>
            <strong>{fmtPct(mutation.data.churn_probability)}</strong>
          </div>
          <div className="result-row">
            <span>Prediction</span>
            <strong
              style={{
                color:
                  mutation.data.churn_prediction === 1 ? "#D64545" : "#0E7C7B",
              }}
            >
              {mutation.data.churn_prediction === 1 ? "Churned" : "Retained"}
            </strong>
          </div>
          <div className="result-row">
            <span>Model Used</span>
            <strong>{mutation.data.best_submodel}</strong>
          </div>
          <div className="result-row">
            <span>Cached</span>
            <strong>{mutation.data.cached ? "Yes" : "No"}</strong>
          </div>
        </div>
      )}

      {mutation.isError && (
        <div className="error-inline">
          ⚠️ {(mutation.error as Error).message}
        </div>
      )}
    </div>
  );
};

const PatientManagement: React.FC = () => (
  <div className="page">
    <div className="page-header">
      <h2 className="page-title">Patient Management</h2>
    </div>
    <p className="page-desc">
      Churn prediction per patient via <code>POST /predict/churn</code>.
      Patient listing requires a backend listing endpoint.
    </p>

    <ChurnPredictor />

    <UnavailableState
      feature="Patient Table / Listing / CSV Export"
      reason="The API does not expose a paginated patient listing endpoint. To implement this, add GET /patients?page=&size=&service=&date_from=&date_to= backed by a PostgreSQL SELECT with pagination."
    />
  </div>
);

export default PatientManagement;

