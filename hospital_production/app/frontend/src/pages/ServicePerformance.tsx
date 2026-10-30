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
} from "recharts";
import { api } from "../lib/api";
import { LoadingSpinner, ErrorState, UnavailableState } from "../components/ui/States";
import { PALETTE } from "../lib/constants";

const SAMPLE_IDS = ["PAT-12345678"]; // replace with real IDs from your data

const ServicePredictionChart: React.FC = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["service-prediction", SAMPLE_IDS],
    queryFn: () => api.predictBatch("service_performance", SAMPLE_IDS),
    staleTime: 60_000,
  });

  if (isLoading) return <LoadingSpinner label="Loading service-performance predictions…" />;
  if (error)
    return <ErrorState message="Could not load service-performance predictions." retry={refetch} />;
  if (!data) return null;

  const chartData = data.predictions.map((item) => ({
    patientId: item.patient_id,
    predictedHealthScore: typeof item.prediction === "number" ? item.prediction : 0,
  }));

  return (
    <div className="card">
      <h3 className="card-title">Predicted Service Health</h3>
      <p className="card-desc">
        Calls <code>POST /predict/service_performance/batch</code>.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="patientId" tick={{ fontSize: 11 }} />
          <YAxis />
          <Tooltip />
          <Bar dataKey="predictedHealthScore" fill={PALETTE[2]} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const ServicePerformance: React.FC = () => (
  <div className="page">
    <div className="page-header">
      <h2 className="page-title">Service Performance</h2>
    </div>
    <p className="page-desc">
      Predicted health scores from the trained service_performance model.
    </p>
    <ServicePredictionChart />
    <UnavailableState
      feature="Aggregated per-service comparison"
      reason="Add GET /analytics/services for per-service aggregates of the trained predictions."
    />
  </div>
);

export default ServicePerformance;