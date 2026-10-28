import React from "react";

export const LoadingSpinner: React.FC<{ label?: string }> = ({
  label = "Loading…",
}) => (
  <div className="loading-state">
    <div className="spinner" aria-hidden="true" />
    <p className="loading-label">{label}</p>
  </div>
);

export const ErrorState: React.FC<{ message: string; retry?: () => void }> = ({
  message,
  retry,
}) => (
  <div className="error-state">
    <span className="error-icon">⚠️</span>
    <p className="error-message">{message}</p>
    {retry && (
      <button className="btn btn-sm" onClick={retry}>
        Retry
      </button>
    )}
  </div>
);

export const EmptyState: React.FC<{
  title?: string;
  description?: string;
}> = ({ title = "No data available", description }) => (
  <div className="empty-state">
    <span className="empty-icon">📭</span>
    <p className="empty-title">{title}</p>
    {description && <p className="empty-desc">{description}</p>}
  </div>
);

export const UnavailableState: React.FC<{
  feature: string;
  reason: string;
}> = ({ feature, reason }) => (
  <div className="unavailable-state">
    <span className="unavailable-icon">🚧</span>
    <p className="unavailable-title">{feature} — Unavailable</p>
    <p className="unavailable-reason">{reason}</p>
  </div>
);

