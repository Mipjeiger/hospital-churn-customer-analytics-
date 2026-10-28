import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";
import { api } from "../lib/api";
import { KpiCard, KpiGrid } from "../components/ui/KpiCard";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { fmtPct, fmtFloat } from "../lib/fmt";
import { PALETTE } from "../lib/constants";

const MORALE_TREND = [
  { month: "Jan", morale: 72 },
  { month: "Feb", morale: 69 },
  { month: "Mar", morale: 74 },
  { month: "Apr", morale: 70 },
  { month: "May", morale: 75 },
  { month: "Jun", morale: 73 },
];

const DEPT_DATA = [
  { dept: "Emergency", attendance: 88, morale: 68 },
  { dept: "Surgery", attendance: 92, morale: 75 },
  { dept: "General Med", attendance: 85, morale: 72 },
  { dept: "ICU", attendance: 90, morale: 65 },
];

const StaffAnalytics: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["business-kpi"],
    queryFn: () => api.businessKpi(),
  });

  if (isLoading) return <LoadingSpinner label="Loading staff analytics…" />;
  if (error)
    return <ErrorState message="Could not load staff data." retry={refetch} />;
  if (!data) return null;

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Staff Analytics</h2>
      </div>
      <p className="page-desc">
        Staff attendance from <code>/business-kpi</code>. Morale and
        per-department breakdowns are indicative.
      </p>

      <KpiGrid>
        <KpiCard
          label="Staff Attendance Rate"
          value={fmtPct(data.staff_attendance_rate)}
          sub="present / total"
          accent="#E6A817"
        />
        <KpiCard
          label="Attendance Health"
          value={data.staff_attendance_rate >= 0.85 ? "Good" : data.staff_attendance_rate >= 0.70 ? "Moderate" : "Low"}
          sub={data.staff_attendance_rate >= 0.85 ? "≥ 85%" : "< 85%"}
          accent={data.staff_attendance_rate >= 0.85 ? "#0E7C7B" : "#D64545"}
        />
      </KpiGrid>

      <div className="charts-row">
        <div className="chart-card chart-card--wide">
          <h3 className="chart-title">Staff Morale Trend (Illustrative)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={MORALE_TREND}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis domain={[60, 80]} />
              <Tooltip />
              <Line
                dataKey="morale"
                stroke={PALETTE[3]}
                strokeWidth={2}
                dot={false}
                name="Avg Morale"
              />
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
        reason="The current API exposes only an aggregate staff_attendance_rate via /business-kpi. Per-department morale and detailed absenteeism require a /analytics/staff endpoint."
      />
    </div>
  );
};

export default StaffAnalytics;

