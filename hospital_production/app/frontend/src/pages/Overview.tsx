import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState } from "../components/ui/States";
import { StatusDot } from "../components/ui/Badge";
import { fmtNum, fmtPct, fmtFloat } from "../lib/fmt";
import { PALETTE } from "../lib/constants";

// ── KPI section ───────────────────────────────────────────────────────────────
const KpiSection: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading KPIs…" />;
  if (error)
    return (
      <ErrorState
        message="Could not load KPIs from backend."
        retry={refetch}
      />
    );
  if (!data) return null;

  return (
    <KpiGrid>
      <KpiCard
        label="Total Patients"
        value={fmtNum(data.total_patients)}
        sub="in database"
        accent="#0F2A44"
      />
      <KpiCard
        label="Avg Satisfaction"
        value={fmtFloat(data.average_satisfaction, 1)}
        sub="/ 100"
        accent="#0E7C7B"
      />
      <KpiCard
        label="Churn Rate"
        value={fmtPct(data.churn_rate)}
        sub="historical"
        accent="#D64545"
      />
      <KpiCard
        label="Admission Rate"
        value={fmtPct(data.admission_rate)}
        sub="admitted / requested"
        accent="#2A9D8F"
      />
      <KpiCard
        label="Staff Attendance"
        value={fmtPct(data.staff_attendance_rate)}
        sub="present / total"
        accent="#E6A817"
      />
    </KpiGrid>
  );
};

// ── Model health row (each is its own component — safe for hooks) ─────────────
const MODEL_NAMES = [
  "patient_churn",
  "risk_level",
  "risk_score",
  "service_performance",
  "integrity_audit",
];

interface ModelHealthRowData {
  name: string;
  loaded: number;
  status: string;
}

const ModelHealthRow: React.FC<{ name: string; onData: (d: ModelHealthRowData) => void }> = ({ name, onData }) => {
  const { data, isLoading } = useQuery({
    queryKey: ["model-health", name],
    queryFn: () => api.modelHealth(name),
    staleTime: 60_000,
  });
  React.useEffect(() => {
    onData({
      name,
      loaded: data?.loaded ? 1 : 0,
      status: isLoading ? "loading" : data?.loaded ? "ok" : "offline",
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, isLoading]);
  return null;
};

const ModelHealthBars: React.FC = () => {
  const [rows, setRows] = React.useState<ModelHealthRowData[]>(
    MODEL_NAMES.map((n) => ({ name: n, loaded: 0, status: "loading" }))
  );

  const updateRow = React.useCallback((d: ModelHealthRowData) => {
    setRows((prev) => prev.map((r) => (r.name === d.name ? d : r)));
  }, []);

  return (
    <div className="chart-card">
      {MODEL_NAMES.map((n) => (
        <ModelHealthRow key={n} name={n} onData={updateRow} />
      ))}
      <h3 className="chart-title">ML Model Status</h3>
      <div className="table-wrapper" style={{ marginTop: 8 }}>
        <table className="data-table">
          <thead>
            <tr><th>Model</th><th>Status</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className="table-row">
                <td className="table-cell text-mono">{r.name}</td>
                <td className="table-cell">
                  <span className="flex-row-center gap-6">
                    {r.status === "loading" ? (
                      <span className="muted">Checking…</span>
                    ) : (
                      <>
                        <StatusDot ok={r.loaded === 1} />
                        {r.loaded === 1 ? "Loaded" : "Not loaded"}
                      </>
                    )}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ── System health ──────────────────────────────────────────────────────────────
const SystemHealthBadge: React.FC = () => {
  const { data, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    refetchInterval: 30_000,
  });

  if (isLoading)
    return <span className="badge" style={{ background: "#6B7C93", color: "#fff" }}>Checking…</span>;

  return (
    <span
      className="badge"
      style={{
        background: data?.status === "healthy" ? "#0E7C7B" : "#D64545",
        color: "#fff",
      }}
    >
      API {data?.status ?? "unknown"}
    </span>
  );
};

const INSIGHT_DATA = [
  { month: "Jan", churn: 18, admission: 81, satisfaction: 72 },
  { month: "Feb", churn: 20, admission: 79, satisfaction: 70 },
  { month: "Mar", churn: 17, admission: 82, satisfaction: 74 },
  { month: "Apr", churn: 22, admission: 77, satisfaction: 68 },
  { month: "May", churn: 19, admission: 80, satisfaction: 73 },
  { month: "Jun", churn: 21, admission: 78, satisfaction: 71 },
];

const Overview: React.FC = () => (
  <div className="page">
    <div className="page-header">
      <h2 className="page-title">Executive Overview</h2>
      <SystemHealthBadge />
    </div>
    <p className="page-desc">
      Live KPIs from the hospital database · ML model status · API health.
    </p>

    <KpiSection />

    <div className="charts-row">
      <div className="chart-card chart-card--wide">
        <h3 className="chart-title">
          Indicative Monthly Trend
          <span className="chart-note-inline">
            {" "}— illustrative shape; live time-series requires an analytics endpoint
          </span>
        </h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={INSIGHT_DATA}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis yAxisId="left" domain={[0, 100]} tickFormatter={(v) => `${v}`} />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" dataKey="churn" stroke={PALETTE[4]} name="Churn %" dot={false} strokeWidth={2} />
            <Line yAxisId="left" dataKey="admission" stroke={PALETTE[2]} name="Admission %" dot={false} strokeWidth={2} />
            <Line yAxisId="left" dataKey="satisfaction" stroke={PALETTE[1]} name="Satisfaction" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ModelHealthBars />
    </div>

    <div className="insight-box">
      <strong>📌 Data Note:</strong> KPIs are computed from the full PostgreSQL hospital table
      via <code>/business-kpi</code>. The trend chart uses illustrative shape data. For
      date-filtered analytics, add a <code>/analytics/summary</code> endpoint.
    </div>
  </div>
);

export default Overview;
