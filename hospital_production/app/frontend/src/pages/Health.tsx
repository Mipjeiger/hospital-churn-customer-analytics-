import React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState } from "../components/ui/States";
import { StatusDot } from "../components/ui/Badge";

const MODEL_NAMES = [
  "patient_churn",
  "risk_level",
  "risk_score",
  "service_performance",
  "integrity_audit",
  "staff_staff_morale",
  "staff_staff_attendance_rate",
];

const EXTERNAL_LINKS = [
  { label: "Grafana Dashboards", href: "http://localhost:3000" },
  { label: "MLflow Tracking", href: "http://localhost:5002" },
  { label: "Prefect Workflows", href: "http://localhost:4200" },
  { label: "Langfuse Traces", href: "http://localhost:4000" },
  { label: "Prometheus Metrics", href: "http://localhost:9090" },
  { label: "MinIO Console", href: "http://localhost:9001" },
];

const ApiHealthCard: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    refetchInterval: 15_000,
  });

  if (isLoading) return <LoadingSpinner label="Checking API health…" />;
  if (error)
    return (
      <ErrorState
        message="FastAPI is unreachable. Is the backend running on port 8000?"
        retry={refetch}
      />
    );

  return (
    <div className="card">
      <div className="flex-row-center gap-8">
        <StatusDot ok={data?.status === "healthy"} />
        <h3 className="card-title" style={{ margin: 0 }}>
          FastAPI: {data?.status ?? "unknown"}
        </h3>
      </div>
      <p className="card-desc">{data?.message}</p>
    </div>
  );
};

/** Each model's health check in its own component so hooks are not called in loops */
const ModelHealthRow: React.FC<{ name: string }> = ({ name }) => {
  const { data, isLoading } = useQuery({
    queryKey: ["model-health", name],
    queryFn: () => api.modelHealth(name),
    staleTime: 60_000,
  });

  return (
    <tr className="table-row">
      <td className="table-cell text-mono">{name}</td>
      <td className="table-cell text-mono">{data?.stored_name ?? "—"}</td>
      <td className="table-cell">
        {isLoading ? (
          <span className="muted">Checking…</span>
        ) : (
          <span className="flex-row-center gap-6">
            <StatusDot ok={!!data?.loaded} />
            {data?.loaded ? "Loaded" : "Not loaded"}
          </span>
        )}
      </td>
      <td className="table-cell text-mono text-sm">{data?.object ?? "—"}</td>
    </tr>
  );
};

const ModelHealthTable: React.FC = () => (
  <div className="card">
    <h3 className="card-title">ML Model Health</h3>
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Stored Name</th>
            <th>Status</th>
            <th>Object</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_NAMES.map((name) => (
            <ModelHealthRow key={name} name={name} />
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

const SystemHealth: React.FC = () => {
  const { data: kpi, isLoading: kpiLoading } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
    staleTime: 60_000,
  });

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">System Health</h2>
      </div>
      <p className="page-desc">
        Live checks against <code>GET /health</code> and{" "}
        <code>GET /model/&#123;name&#125;/health</code>.
      </p>

      <ApiHealthCard />

      {!kpiLoading && kpi && (
        <KpiGrid>
          <KpiCard
            label="Total Patients"
            value={kpi.total_patients.toLocaleString()}
            sub="DB rows"
            accent="#0F2A44"
          />
          <KpiCard
            label="Avg Satisfaction"
            value={kpi.average_satisfaction.toFixed(1)}
            sub="/ 100"
            accent="#0E7C7B"
          />
          <KpiCard
            label="Churn Rate"
            value={`${(kpi.churn_rate * 100).toFixed(1)}%`}
            sub="historical"
            accent="#D64545"
          />
          <KpiCard
            label="Admission Rate"
            value={`${(kpi.admission_rate * 100).toFixed(1)}%`}
            sub="admitted / requested"
            accent="#2A9D8F"
          />
          <KpiCard
            label="Staff Attendance"
            value={`${(kpi.staff_attendance_rate * 100).toFixed(1)}%`}
            sub="present / total"
            accent="#E6A817"
          />
        </KpiGrid>
      )}

      <ModelHealthTable />

      <div className="card">
        <h3 className="card-title">External Services</h3>
        <p className="card-desc">
          Links open in a new tab. Reachability is not verified from the browser.
        </p>
        <div className="link-grid">
          {EXTERNAL_LINKS.map(({ label, href }) => (
            <a
              key={href}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="external-link"
            >
              {label} ↗
            </a>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SystemHealth;
