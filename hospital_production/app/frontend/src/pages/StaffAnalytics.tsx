import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { fmtPct } from "../lib/fmt";
import { PALETTE } from "../lib/constants";

const SAMPLE_IDS = ["PAT-12345678"];

const MORALE_TREND = [
  { month: "Jan", morale: 72 }, { month: "Feb", morale: 69 },
  { month: "Mar", morale: 74 }, { month: "Apr", morale: 70 },
  { month: "May", morale: 75 }, { month: "Jun", morale: 73 },
];

const DEPT_DATA = [
  { dept: "Emergency", attendance: 88, morale: 68 },
  { dept: "Surgery", attendance: 92, morale: 75 },
  { dept: "General Med", attendance: 85, morale: 72 },
  { dept: "ICU", attendance: 90, morale: 65 },
];

const StaffPredictions: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["staff-predictions", SAMPLE_IDS],
    queryFn: async () => {
      const [morale, attendance] = await Promise.all([
        api.predictBatch("staff_staff_morale", SAMPLE_IDS),
        api.predictBatch("staff_staff_attendance_rate", SAMPLE_IDS),
      ]);
      return { morale, attendance };
    },
    staleTime: 60_000,
  });

  if (isLoading) return <LoadingSpinner label="Loading staff predictions…" />;
  if (error) return <ErrorState message="Could not load staff predictions." retry={refetch} />;
  if (!data) return null;

  const atRisk = data.morale.predictions.filter(
    (item) => typeof item.prediction === "number" && item.prediction < 65
  );

  return (
    <div className="card">
      <h3 className="card-title">Predicted Staff Health</h3>
      <p className="card-desc">
        Morale and attendance predictions from{" "}
        <code>POST /predict/staff_staff_morale/batch</code> and{" "}
        <code>POST /predict/staff_staff_attendance_rate/batch</code>.
      </p>
      <div className="result-card">
        <div className="result-row">
          <span>Patients evaluated</span>
          <strong>{data.morale.count}</strong>
        </div>
        <div className="result-row">
          <span>Predicted low-morale (&lt; 65)</span>
          <strong>{atRisk.length}</strong>
        </div>
      </div>
    </div>
  );
};

const StaffAnalytics: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading staff analytics…" />;
  if (error) return <ErrorState message="Could not load staff data." retry={refetch} />;
  if (!data) return null;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Staff Analytics</h2>
      </div>
      <p className="page-desc">
        Staff attendance from <code>/business-kpi</code>. Morale and per-department breakdowns are indicative.
      </p>

      <KpiGrid>
        <KpiCard label="Staff Attendance Rate" value={fmtPct(data.staff_attendance_rate)} sub="present / total" accent="#E6A817" />
        <KpiCard
          label="Attendance Health"
          value={data.staff_attendance_rate >= 0.85 ? "Good" : data.staff_attendance_rate >= 0.70 ? "Moderate" : "Low"}
          accent={data.staff_attendance_rate >= 0.85 ? "#0E7C7B" : "#D64545"}
        />
      </KpiGrid>

      <StaffPredictions />

      <div className="charts-row">
        <div className="chart-card chart-card--wide">
          <h3 className="chart-title">Staff Morale Trend (Illustrative)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={MORALE_TREND}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis domain={[60, 80]} />
              <Tooltip />
              <Line dataKey="morale" stroke={PALETTE[3]} strokeWidth={2} dot={false} name="Avg Morale" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card chart-card--wide">
          <h3 className="chart-title">Attendance & Morale by Department (Illustrative)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={DEPT_DATA}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="dept" />
              <YAxis domain={[60, 100]} />
              <Tooltip />
              <Bar dataKey="attendance" fill={PALETTE[0]} name="Attendance %" radius={[4, 4, 0, 0]} />
              <Bar dataKey="morale" fill={PALETTE[3]} name="Morale" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <UnavailableState
        feature="Real-time per-department staff metrics"
        reason="Add GET /analytics/staff for per-department morale and absenteeism."
      />
    </div>
  );
};

export default StaffAnalytics;