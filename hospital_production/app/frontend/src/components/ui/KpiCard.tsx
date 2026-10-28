import React from "react";
import { clsx } from "../../lib/clsx";

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
  accent?: string; // CSS colour string
  icon?: React.ReactNode;
}

export const KpiCard: React.FC<KpiCardProps> = ({
  label,
  value,
  sub,
  accent = "#0F2A44",
  icon,
}) => (
  <div
    className="kpi-card"
    style={{ borderTop: `3px solid ${accent}` }}
  >
    <div className="kpi-header">
      <span className="kpi-label">{label}</span>
      {icon && <span className="kpi-icon">{icon}</span>}
    </div>
    <div className="kpi-value">{value}</div>
    {sub && <div className="kpi-sub">{sub}</div>}
  </div>
);

export const KpiGrid: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <div className="kpi-grid">{children}</div>;

