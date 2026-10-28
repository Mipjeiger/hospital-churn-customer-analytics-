import React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { LoadingSpinner, ErrorState } from "../components/ui/States";
import { StatusDot } from "../components/ui/Badge";
import { fmtNum } from "../lib/fmt";

const CONCEPTS = [
  "patient_churn",
  "risk_level",
  "risk_score",
  "service_performance",
  "integrity_audit",
  "staff_staff_morale",
  "staff_staff_attendance_rate",
];

const ModelRow: React.FC<{ name: string }> = ({ name }) => {
  const { data, isLoading } = useQuery({
    queryKey: ["model-health", name],
    queryFn: () => api.modelHealth(name),
    staleTime: 60_000,
  });

  return (
    <tr className="table-row">
      <td className="table-cell">
        <code>{name}</code>
      </td>
      <td className="table-cell">
        {isLoading ? (
          <span className="badge" style={{ background: "#6B7C93", color: "#fff" }}>
            Checking…
          </span>
        ) : (
          <span className="flex-row-center gap-6">
            <StatusDot ok={!!data?.loaded} />
            {data?.loaded ? "Loaded" : "Not loaded"}
          </span>
        )}
      </td>
      <td className="table-cell text-mono text-sm">
        {data?.object ?? "—"}
      </td>
    </tr>
  );
};

const ModelList: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["model-list"],
    queryFn: () => api.listModels(),
    staleTime: 60_000,
  });

  if (isLoading) return <LoadingSpinner label="Listing models from MinIO…" />;
  if (error)
    return (
      <ErrorState
        message="Could not list models from MinIO. The MinIO service may be offline."
        retry={refetch}
      />
    );

  return (
    <div className="card">
      <h3 className="card-title">
        Models in MinIO bucket: <code>{data?.bucket ?? "—"}</code>
        <span className="badge" style={{ marginLeft: 8, background: "#0F2A44", color: "#fff" }}>
          {data?.count ?? 0} found
        </span>
      </h3>
      {data?.count === 0 ? (
        <p className="empty-desc">No model bundles found in the bucket.</p>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Status</th>
                <th>Object Path</th>
                <th>Size</th>
                <th>Last Modified</th>
              </tr>
            </thead>
            <tbody>
              {data?.models.map((m) => (
                <tr key={m.model} className="table-row">
                  <td className="table-cell"><code>{m.model}</code></td>
                  <td className="table-cell">
                    <span className="flex-row-center gap-6">
                      <StatusDot ok={m.loaded} />
                      {m.loaded ? "Loaded" : "Not loaded"}
                    </span>
                  </td>
                  <td className="table-cell text-mono text-sm">{m.object}</td>
                  <td className="table-cell">{fmtNum(m.size_bytes)} B</td>
                  <td className="table-cell text-sm">
                    {new Date(m.last_modified).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const MLModels: React.FC = () => (
  <div className="page">
    <div className="page-header">
      <h2 className="page-title">ML Model Performance</h2>
    </div>
    <p className="page-desc">
      Model health from <code>GET /model</code> and{" "}
      <code>GET /model/&#123;name&#125;/health</code>. Training metrics
      (accuracy, ROC-AUC, F1) are stored in MinIO manifests.
    </p>

    <ModelList />

    <div className="section-header">
      <h3 className="section-title">Per-Concept Health Check</h3>
    </div>
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Concept</th>
            <th>Status</th>
            <th>MinIO Object</th>
          </tr>
        </thead>
        <tbody>
          {CONCEPTS.map((c) => (
            <ModelRow key={c} name={c} />
          ))}
        </tbody>
      </table>
    </div>

    <div className="insight-box" style={{ marginTop: 16 }}>
      <strong>📌 Note on training metrics:</strong> ROC-AUC, accuracy, and F1
      scores are embedded in the MinIO manifest JSON files (
      <code>manifest/&#123;concept&#125;.json</code>). To surface them here, add a{" "}
      <code>GET /model/&#123;name&#125;/metrics</code> endpoint that reads the manifest and
      returns the metrics dict.
    </div>
  </div>
);

export default MLModels;
