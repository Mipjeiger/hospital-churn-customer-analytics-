import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { fmtNum, fmtFloat, fmtPct } from "../lib/fmt";
import { SERVICE_COLORS, CHURN_COLORS } from "../lib/constants";

const SERVICE_PIE_DATA = [
  { name: "Emergency", value: 28 },
  { name: "Surgery", value: 22 },
  { name: "General Med", value: 35 },
  { name: "ICU", value: 15 },
];
const SERVICE_COLOURS = Object.values(SERVICE_COLORS);

const ServicePie: React.FC = () => (
  <div className="chart-card">
    <h3 className="chart-title">Patient Distribution by Service</h3>
    <p className="chart-note">
      Approximate split — per-service data requires <code>/analytics/summary</code>
    </p>
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={SERVICE_PIE_DATA}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={90}
          label={({ name, percent }: { name: string; percent: number }) =>
            `${name} ${(percent * 100).toFixed(0)}%`
          }
          labelLine={false}
        >
          {SERVICE_PIE_DATA.map((_, i) => (
            <Cell key={i} fill={SERVICE_COLOURS[i % SERVICE_COLOURS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(v: number) => [`${v}%`, "Share"]} />
      </PieChart>
    </ResponsiveContainer>
  </div>
);

const ChurnBar: React.FC<{ churn_rate: number }> = ({ churn_rate }) => {
  const data = [
    { name: "Retained", value: Math.round((1 - churn_rate) * 100), fill: CHURN_COLORS["Retained"] },
    { name: "Churned", value: Math.round(churn_rate * 100), fill: CHURN_COLORS["Churned"] },
  ];
  return (
    <div className="chart-card">
      <h3 className="chart-title">Churn Overview</h3>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis tickFormatter={(v) => `${v}%`} />
          <Tooltip formatter={(v: number) => [`${v}%`, ""]} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const SatisfactionBar: React.FC<{ avg_satisfaction: number }> = ({ avg_satisfaction }) => {
  const bins = [
    { bin: "0–20", share: 5 },
    { bin: "21–40", share: 10 },
    { bin: "41–60", share: 20 },
    { bin: "61–80", share: 35 },
    { bin: "81–100", share: 30 },
  ];
  return (
    <div className="chart-card">
      <h3 className="chart-title">
        Satisfaction Distribution
        <span className="chart-note-inline"> (illustrative bins)</span>
      </h3>
      <p className="chart-note">
        Avg satisfaction: <strong>{fmtFloat(avg_satisfaction, 1)}/100</strong>
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={bins}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="bin" />
          <YAxis tickFormatter={(v) => `${v}%`} />
          <Tooltip formatter={(v: number) => [`${v}%`, "Share"]} />
          <Bar dataKey="share" fill="#2A9D8F" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const PatientAnalytics: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading patient analytics…" />;
  if (error)
    return (
      <ErrorState
        message="Could not load patient KPIs from backend."
        retry={refetch}
      />
    );
  if (!data) return null;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Patient Analytics</h2>
      </div>
      <p className="page-desc">
        Population metrics sourced from <code>/business-kpi</code>.
        Detailed per-patient breakdowns require a listing endpoint.
      </p>

      <KpiGrid>
        <KpiCard label="Total Patients" value={fmtNum(data.total_patients)} sub="in database" accent="#0F2A44" />
        <KpiCard label="Avg Satisfaction" value={fmtFloat(data.average_satisfaction, 1)} sub="/ 100" accent="#0E7C7B" />
        <KpiCard label="Churn Rate" value={fmtPct(data.churn_rate)} sub="historical" accent="#D64545" />
        <KpiCard label="Admission Rate" value={fmtPct(data.admission_rate)} sub="admitted / requested" accent="#2A9D8F" />
      </KpiGrid>

      <div className="charts-row">
        <ChurnBar churn_rate={data.churn_rate} />
        <SatisfactionBar avg_satisfaction={data.average_satisfaction} />
        <ServicePie />
      </div>

      <UnavailableState
        feature="Age / Length-of-Stay / Per-patient breakdowns"
        reason="The current API does not expose a patient listing or aggregated analytics endpoint. These charts require a GET /analytics/patients endpoint backed by PostgreSQL aggregation queries."
      />
    </div>
  );
};

export default PatientAnalytics;
