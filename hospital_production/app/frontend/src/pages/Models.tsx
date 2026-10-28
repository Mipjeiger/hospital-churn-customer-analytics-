import React from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { LoadingSpinner, ErrorState } from "../components/ui/States";
import { StatusDot } from "../components/ui/Badge";

export const ModelsPage: React.FC = () => {
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["models"],
    queryFn: () => api.listModels(),
    refetchInterval: 60_000,
  });

  if (isLoading) return <LoadingSpinner label="Loading models…" />;
  if (error)
    return (
      <ErrorState
        message="Failed to fetch model list. Is the FastAPI backend running?"
        retry={refetch}
      />
    );

  return (
    <div className="page">
      <div className="page-header">
        <h2 className="page-title">Model Artifacts</h2>
      </div>
      <p className="page-desc">
        Shows what model bundles are present in MinIO and whether they are loaded by the service.
      </p>

      <div className="card">
        <h3 className="card-title">Available Models</h3>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Stored Name</th>
                <th>Status</th>
                <th>Size (bytes)</th>
                <th>Last Modified</th>
              </tr>
            </thead>
            <tbody>
              {data?.models.map((m) => (
                <tr key={m.model} className="table-row">
                  <td className="table-cell text-mono">{m.model}</td>
                  <td className="table-cell text-mono">{m.object ?? "—"}</td>
                  <td className="table-cell">
                    <span className="flex-row-center gap-6">
                      <StatusDot ok={m.loaded} />
                      {m.loaded ? "Loaded" : "Not loaded"}
                    </span>
                  </td>
                  <td className="table-cell text-mono">{m.size_bytes?.toLocaleString() ?? "—"}</td>
                  <td className="table-cell text-mono">{m.last_modified ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelsPage;