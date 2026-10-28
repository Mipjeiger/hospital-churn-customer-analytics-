import React, { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { LoadingSpinner, UnavailableState } from "../components/ui/States";
import { fmtFloat } from "../lib/fmt";

const VALID_PAT_RE = /^PAT-\w{8}$/;

const FeatureViewer: React.FC = () => {
  const [pid, setPid] = useState("");
  const [pidError, setPidError] = useState<string | null>(null);
  const [submittedPid, setSubmittedPid] = useState<string | null>(null);

  const { data, isFetching, error } = useQuery({
    queryKey: ["patient-features", submittedPid],
    queryFn: () => api.patientFeatures(submittedPid!),
    enabled: !!submittedPid,
  });

  const { data: schemaData } = useQuery({
    queryKey: ["feature-schema"],
    queryFn: () => api.featureSchema(),
    staleTime: 300_000,
  });

  const handleLookup = (e: React.FormEvent) => {
    e.preventDefault();
    if (!VALID_PAT_RE.test(pid.trim())) {
      setPidError("Patient ID must match PAT-XXXXXXXX");
      return;
    }
    setPidError(null);
    setSubmittedPid(pid.trim());
  };

  return (
    <div className="card">
      <h3 className="card-title">Patient Feature Lookup</h3>
      <p className="card-desc">
        Calls <code>GET /predict/features/&#123;patient_id&#125;</code> (Feast online store).
      </p>
      <form onSubmit={handleLookup} className="form-row">
        <input
          className={`form-input${pidError ? " form-input--error" : ""}`}
          type="text"
          placeholder="PAT-12345678"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
        />
        <button className="btn btn-primary" type="submit" disabled={isFetching}>
          Lookup
        </button>
      </form>
      {pidError && <p className="field-error">{pidError}</p>}

      {isFetching && <LoadingSpinner label="Fetching features from Feast…" />}
      {error && (
        <div className="error-inline">⚠️ {(error as Error).message}</div>
      )}

      {data && !isFetching && (
        <div className="table-wrapper" style={{ marginTop: 12 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data).map(([k, v]) => (
                <tr key={k} className="table-row">
                  <td className="table-cell text-mono">{k}</td>
                  <td className="table-cell">
                    {v == null
                      ? <span className="muted">null</span>
                      : typeof v === "number"
                      ? fmtFloat(v, 4)
                      : String(v)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {schemaData && (
        <div style={{ marginTop: 16 }}>
          <h4 className="card-subtitle">Feature Schema ({schemaData.features.length} features)</h4>
          <div className="feature-chips">
            {schemaData.features.map((f) => (
              <span key={f} className="feature-chip">{f}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const BatchPredictor: React.FC = () => {
  const [concept, setConcept] = useState("patient_churn");
  const [pidsText, setPidsText] = useState("");

  const mutation = useMutation({
    mutationFn: () => {
      const ids = pidsText.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
      return api.predictBatch(concept, ids);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate();
  };

  const CONCEPTS = [
    "patient_churn",
    "risk_level",
    "risk_score",
    "service_performance",
    "integrity_audit",
  ];

  return (
    <div className="card">
      <h3 className="card-title">Batch Prediction Explorer</h3>
      <p className="card-desc">
        Calls <code>POST /predict/&#123;concept&#125;/batch</code> with a list of patient IDs.
      </p>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <div className="form-row">
          <label className="form-label">Concept</label>
          <select
            className="form-select"
            value={concept}
            onChange={(e) => setConcept(e.target.value)}
          >
            {CONCEPTS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="form-label">Patient IDs (one per line or comma-separated)</label>
          <textarea
            className="form-textarea"
            value={pidsText}
            onChange={(e) => setPidsText(e.target.value)}
            placeholder={"PAT-12345678\nPAT-87654321"}
            rows={4}
          />
        </div>
        <button
          className="btn btn-primary"
          type="submit"
          disabled={mutation.isPending || !pidsText.trim()}
        >
          {mutation.isPending ? "Predicting…" : "Run Batch Prediction"}
        </button>
      </form>

      {mutation.isPending && <LoadingSpinner label="Running batch prediction…" />}
      {mutation.isError && (
        <div className="error-inline">⚠️ {(mutation.error as Error).message}</div>
      )}

      {mutation.data && (
        <div style={{ marginTop: 12 }}>
          <p className="text-sm">
            <strong>{mutation.data.count}</strong> predictions returned for{" "}
            <code>{mutation.data.concept}</code> ({mutation.data.task})
          </p>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Submodel</th>
                </tr>
              </thead>
              <tbody>
                {mutation.data.predictions.slice(0, 100).map((p) => (
                  <tr key={p.patient_id} className="table-row">
                    <td className="table-cell text-mono">{p.patient_id}</td>
                    <td className="table-cell">{String(p.prediction)}</td>
                    <td className="table-cell">
                      {p.probability != null ? `${(p.probability * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="table-cell text-sm">{p.submodel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {mutation.data.count > 100 && (
              <p className="chart-note">Showing first 100 of {mutation.data.count} results.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const DataExplorer: React.FC = () => (
  <div className="page">
    <div className="page-header">
      <h2 className="page-title">Data Explorer</h2>
    </div>
    <p className="page-desc">
      Feature inspection via Feast online store and interactive batch
      prediction.
    </p>

    <FeatureViewer />
    <BatchPredictor />

    <UnavailableState
      feature="Raw patient data table / CSV export"
      reason="The API does not expose a patient listing endpoint. Raw table access would require a GET /patients endpoint with filtering and pagination backed by PostgreSQL."
    />
  </div>
);

export default DataExplorer;

