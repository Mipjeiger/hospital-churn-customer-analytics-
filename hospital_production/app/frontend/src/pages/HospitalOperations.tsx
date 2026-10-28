import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { fmtPct } from "../lib/fmt";
import { PALETTE } from "../lib/constants";

const CAPACITY_DATA = [
  { label: "Low", value: 30, fill: "#0E7C7B" },
  { label: "Moderate", value: 40, fill: "#E6A817" },
  { label: "High", value: 20, fill: "#E67E22" },
  { label: "Critical", value: 10, fill: "#D64545" },
];

const RADAR_DATA = [
  { metric: "Admission", value: 82 },
  { metric: "Beds", value: 65 },
  { metric: "Staff", value: 78 },
  { metric: "Satisfaction", value: 72 },
  { metric: "Refusal", value: 45 },
];

const HospitalOperations: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading operations data…" />;
  if (error)
    return <ErrorState message="Could not load operations data." retry={refetch} />;
  if (!data) return null;

  const requestData = [
    { name: "Admitted", value: Math.round(data.admission_rate * 100), fill: PALETTE[2] },
    { name: "Refused", value: Math.round((1 - data.admission_rate) * 100), fill: PALETTE[4] },
  ];

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Hospital Operations</h2>
      </div>
      <p className="page-desc">
        Capacity, admission, refusal, and staff metrics from <code>/business-kpi</code>.
      </p>

      <KpiGrid>
        <KpiCard label="Admission Rate" value={fmtPct(data.admission_rate)} sub="admitted / requested" accent="#2A9D8F" />
        <KpiCard label="Refusal Rate" value={fmtPct(1 - data.admission_rate)} sub="refused / requested" accent="#D64545" />
        <KpiCard label="Staff Attendance" value={fmtPct(data.staff_attendance_rate)} sub="present / total" accent="#E6A817" />
        <KpiCard label="Avg Satisfaction" value={data.average_satisfaction.toFixed(1)} sub="/ 100" accent="#0E7C7B" />
      </KpiGrid>

      <div className="charts-row">
        {/* Admitted vs Refused */}
        <div className="chart-card">
          <h3 className="chart-title">Admitted vs Refused (%)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={requestData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v: number) => [`${v}%`, ""]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {requestData.map((d, i) => (
                  <Cell key={i} fill={d.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Capacity pressure distribution */}
        <div className="chart-card">
          <h3 className="chart-title">Capacity Pressure Distribution</h3>
          <p className="chart-note">Illustrative — requires analytics endpoint</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={CAPACITY_DATA}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(v: number) => [`${v}%`, "Share"]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {CAPACITY_DATA.map((d, i) => (
                  <Cell key={i} fill={d.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar overview */}
        <div className="chart-card">
          <h3 className="chart-title">Operations Radar</h3>
          <p className="chart-note">Indicative shape</p>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={RADAR_DATA}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <Radar
                name="Operations"
                dataKey="value"
                stroke="#0F2A44"
                fill="#0F2A44"
                fillOpacity={0.3}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <UnavailableState
        feature="Time-series capacity trends / Per-service operations"
        reason="The API does not expose a time-series or per-service aggregation endpoint. Add GET /analytics/operations?from=&to=&service= to enable these charts."
      />
    </div>
  );
};

export default HospitalOperations;
