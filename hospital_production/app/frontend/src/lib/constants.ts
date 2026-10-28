/** Healthcare design tokens */
export const PALETTE = [
  "#0F2A44", "#0E7C7B", "#2A9D8F",
  "#E6A817", "#D64545", "#6B7C93",
  "#8AB4B5", "#C4D7D1",
];

export const SERVICE_COLORS: Record<string, string> = {
  emergency: "#0F2A44",
  surgery: "#0E7C7B",
  general_medicine: "#2A9D8F",
  ICU: "#D64545",
};

export const RISK_COLORS: Record<string, string> = {
  Low: "#0E7C7B",
  Medium: "#E6A817",
  High: "#D64545",
  Critical: "#9B1C1C",
};

export const CHURN_COLORS: Record<string, string> = {
  Retained: "#0E7C7B",
  Churned: "#D64545",
};

export const MODEL_CONCEPTS = [
  "patient_churn",
  "risk_score",
  "risk_level",
  "service_performance",
  "integrity_audit",
  "staff_staff_morale",
  "staff_staff_attendance_rate",
] as const;

export type ModelConcept = (typeof MODEL_CONCEPTS)[number];

