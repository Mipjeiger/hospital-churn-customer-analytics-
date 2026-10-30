import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { fmtPct, fmtFloat } from "../lib/fmt";
import { PALETTE } from "../lib/constants";

const SAMPLE_IDS = ["PAT-12345678"]; // replace with real IDs from your data

const RiskOverview: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["risk-overview", SAMPLE_IDS],
    queryFn: () => api.predictBatch("risk_level", SAMPLE_IDS),
    staleTime: 60_000,
  });

  if (isLoading) return <LoadingSpinner label="Loading risk predictions…" />;
  if (error) return <ErrorState message="Could not load risk predictions." retry={refetch} />;
  if (!data) return null;

  const highRiskCount = data.predictions.filter(
    (item) => String(item.prediction).toLowerCase() === "high"
  ).length;

  const avgProb =
    data.predictions.reduce((sum, item) => sum + (item.probability ?? 0), 0) /
    Math.max(data.predictions.length, 1);

  return (
    <KpiGrid>
      <KpiCard label="Predicted High-Risk" value={String(highRiskCount)} sub={`of ${data.count}`} accent="#D64545" />
      <KpiCard label="Avg Positive Prob" value={fmtPct(avgProb)} sub="model" accent="#E6A817" />
    </KpiGrid>
  );
};

const OperationsCharts: React.FC<{ admission_rate: number; avg_satisfaction: number; staff_attendance_rate: number }> = ({
  admission_rate, avg_satisfaction, staff_attendance_rate,
}) => {
  const requestData = [
    { name: "Admitted", value: Math.round(admission_rate * 100), fill: PALETTE[2] },
    { name: "Refused", value: Math.round((1 - admission_rate) * 100), fill: PALETTE[4] },
  ];

  const radarData = [
    { metric: "Admission", value: Math.round(admission_rate * 100) },
    { metric: "Satisfaction", value: Math.round(avg_satisfaction) },
    { metric: "Staff", value: Math.round(staff_attendance_rate * 100) },
  ];

  return (
    <div className="charts-row">
      <div className="chart-card">
        <h3 className="chart-title">Admitted vs Refused (%)</h3>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={requestData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis tickFormatter={(v) => `${v}%`} />
            <Tooltip formatter={(v: number) => [`${v}%`, ""]} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {requestData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-card">
        <h3 className="chart-title">Operations Radar</h3>
        <ResponsiveContainer width="100%" height={240}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <Radar name="Ops" dataKey="value" stroke="#0F2A44" fill="#0F2A44" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const HospitalOperations: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading operations data…" />;
  if (error) return <ErrorState message="Could not load operations data." retry={refetch} />;
  if (!data) return null;

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
        <KpiCard label="Avg Satisfaction" value={fmtFloat(data.average_satisfaction, 1)} sub="/ 100" accent="#0E7C7B" />
      </KpiGrid>

      <RiskOverview />

      <OperationsCharts
        admission_rate={data.admission_rate}
        avg_satisfaction={data.average_satisfaction}
        staff_attendance_rate={data.staff_attendance_rate}
      />

      <UnavailableState
        feature="Time-series capacity trends / Per-service operations"
        reason="Add GET /analytics/operations?from=&to=&service= to enable these charts."
      />
    </div>
  );
};

export default HospitalOperations;