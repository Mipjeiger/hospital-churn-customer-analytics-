import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import { LoadingSpinner, UnavailableState } from "../components/ui/States";
import { fmtPct } from "../lib/fmt";

const VALID_PAT_RE = /^PAT-\w{8}$/;

// ── Single-patient churn prediction ──────────────────────────────────────────
const ChurnPredictor: React.FC = () => {
  const [pid, setPid] = useState("");
  const [pidError, setPidError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: (patient_id: string) => api.predictChurn(patient_id),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!VALID_PAT_RE.test(pid.trim())) {
      setPidError("Patient ID must match PAT-XXXXXXXX");
      return;
    }
    setPidError(null);
    mutation.mutate(pid.trim());
  };

  return (
    <div className="card">
      <h3 className="card-title">Single-Patient Churn Prediction</h3>
      <p className="card-desc">
        Calls <code>POST /predict/churn</code> with Feast feature store.
      </p>
      <form onSubmit={handleSubmit} className="form-row">
        <input
          className={`form-input${pidError ? " form-input--error" : ""}`}
          type="text"
          placeholder="PAT-12345678"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
        />
        <button className="btn btn-primary" type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Predicting…" : "Predict Churn"}
        </button>
      </form>
      {pidError && <p className="field-error">{pidError}</p>}
      {mutation.isPending && <LoadingSpinner label="Calling /predict/churn…" />}
      {mutation.isSuccess && mutation.data && (
        <div className="result-card">
          <div className="result-row"><span>Patient ID</span><strong>{mutation.data.patient_id}</strong></div>
          <div className="result-row"><span>Churn Probability</span><strong>{fmtPct(mutation.data.churn_probability)}</strong></div>
          <div className="result-row">
            <span>Prediction</span>
            <strong style={{ color: mutation.data.churn_prediction === 1 ? "#D64545" : "#0E7C7B" }}>
              {mutation.data.churn_prediction === 1 ? "Churned" : "Retained"}
            </strong>
          </div>
          <div className="result-row"><span>Model</span><strong>{mutation.data.best_submodel}</strong></div>
          <div className="result-row"><span>Cached</span><strong>{mutation.data.cached ? "Yes" : "No"}</strong></div>
        </div>
      )}
      {mutation.isError && (
        <div className="error-inline">⚠️ {(mutation.error as Error).message}</div>
      )}
    </div>
  );
};

// ── Batch churn + risk prediction (merged) ───────────────────────────────────
const BatchPredictor: React.FC = () => {
  const [idsText, setIdsText] = useState("");
  const [rows, setRows] = useState<
    Array<{ patient_id: string; prediction: unknown; probability: number | null; submodel: string; risk_score: number | null; risk_level: string }>
  >([]);

  const mutation = useMutation({
    mutationFn: async (patientIds: string[]) => {
      const [churn, risk] = await Promise.all([
        api.predictBatch("patient_churn", patientIds),
        api.predictBatch("risk_level", patientIds),
      ]);

      return churn.predictions.map((item) => {
        const riskItem = risk.predictions.find((r) => r.patient_id === item.patient_id);
        return {
          patient_id: item.patient_id,
          prediction: item.prediction,
          probability: item.probability,
          submodel: item.submodel,
          risk_score: null as number | null,
          risk_level: String(riskItem?.prediction ?? "Unknown"),
        };
      });
    },
    onSuccess: (merged) => setRows(merged),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const ids = idsText.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
    if (!ids.length) return;
    mutation.mutate(ids);
  };

  return (
    <div className="card">
      <h3 className="card-title">Batch Churn + Risk Prediction</h3>
      <p className="card-desc">
        Calls <code>POST /predict/patient_churn/batch</code> and{" "}
        <code>POST /predict/risk_level/batch</code>, then merges by patient ID.
      </p>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <label className="form-label">Patient IDs (one per line or comma-separated)</label>
        <textarea
          className="form-textarea"
          value={idsText}
          onChange={(e) => setIdsText(e.target.value)}
          placeholder={"PAT-12345678\nPAT-87654321"}
          rows={4}
        />
        <button className="btn btn-primary" type="submit" disabled={mutation.isPending || !idsText.trim()}>
          {mutation.isPending ? "Predicting…" : "Run Batch Prediction"}
        </button>
      </form>
      {mutation.isPending && <LoadingSpinner label="Running batch prediction…" />}
      {mutation.isError && (
        <div className="error-inline">⚠️ {(mutation.error as Error).message}</div>
      )}
      {rows.length > 0 && (
        <div className="table-wrapper" style={{ marginTop: 12 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Churn</th>
                <th>Churn Prob</th>
                <th>Risk Level</th>
                <th>Submodel</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 100).map((p) => (
                <tr key={p.patient_id} className="table-row">
                  <td className="table-cell text-mono">{p.patient_id}</td>
                  <td className="table-cell">{String(p.prediction)}</td>
                  <td className="table-cell">{p.probability != null ? `${(p.probability * 100).toFixed(1)}%` : "—"}</td>
                  <td className="table-cell">{p.risk_level}</td>
                  <td className="table-cell text-sm">{p.submodel}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 100 && <p className="chart-note">Showing first 100 of {rows.length} results.</p>}
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
    <BatchPredictor />
    <UnavailableState
      feature="Patient Table / Listing / CSV Export"
      reason="Add GET /patients?page=&size=&service= for listing backed by PostgreSQL."
    />
  </div>
);

export default PatientManagement;