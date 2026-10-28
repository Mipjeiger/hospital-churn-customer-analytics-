import React from "react";

interface BadgeProps {
  label: string;
  colour?: string;
  bg?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  label,
  colour = "#fff",
  bg = "#0F2A44",
}) => (
  <span
    className="badge"
    style={{ color: colour, background: bg }}
  >
    {label}
  </span>
);

export const StatusDot: React.FC<{ ok: boolean }> = ({ ok }) => (
  <span
    className="status-dot"
    style={{ background: ok ? "#0E7C7B" : "#D64545" }}
    title={ok ? "Online" : "Offline"}
  />
);

