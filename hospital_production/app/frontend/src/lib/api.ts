/**
 * Centralized typed API client for the Hospital Operations FastAPI backend.
 * All requests go through /api, which:
 *   - in dev  → Vite proxies /api/* to http://fastapi:8000/*
 *   - in prod → nginx proxies /api/* to http://fastapi:8000/*
 */

const BASE =
  ((import.meta as any).env?.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ||
  "/api";

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? JSON.stringify(body);
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, detail);
  }

  // Some endpoints (e.g. a 204) return no body. Guard against JSON.parse("").
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  message: string;
}

export interface ModelEntry {
  model: string;
  loaded: boolean;
  object: string;
  size_bytes: number;
  last_modified: string;
}

export interface ModelListResponse {
  bucket: string;
  count: number;
  models: ModelEntry[];
}

export interface ModelHealthResponse {
  model: string;
  stored_name: string;
  loaded: boolean;
  object: string;
}

export interface BusinessKpiResponse {
  total_patients: number;
  average_satisfaction: number;
  churn_rate: number;
  admission_rate: number;
  staff_attendance_rate: number;
}

export interface ChurnResponse {
  patient_id: string;
  churn_probability: number;
  churn_prediction: number;
  best_submodel: string;
  cached: boolean;
}

export interface PredictionItem {
  patient_id: string;
  prediction: string | number;
  probability: number | null;
  probabilities: number[] | null;
  model_user: string;
  submodel: string;
}

export interface PredictionResponse {
  concept: string;
  task: string;
  count: number;
  predictions: PredictionItem[];
}

export interface FeatureSchemaResponse {
  features: string[];
}

export interface LLMMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface LLMRequest {
  messages: LLMMessage[];
}

export interface LLMResponse {
  text: string;
  role: string;
  provider: string;
}

export interface LLMAgentRequest {
  question: string;
  role: "executive" | "clinical" | "ops" | "risk";
  history?: Array<Record<string, string>> | null;
}

export interface LLMAgentResponse {
  answer: string;
  role: string;
}

export interface PromptResponse {
  name: string;
  template: string;
}

export interface BatchPredictionRequest {
  patient_ids: string[];
}

export interface ChurnBatchResult {
  patient_id: string;
  churn_probability: number;
  churn_prediction: number;
  submodel: string;
}

export interface ChurnBatchResponse {
  model: string;
  count: number;
  results: ChurnBatchResult[];
}

export interface RiskBatchResult {
  patient_id: string;
  risk_score: number;
  risk_level: string;
  risk_score_submodel: string;
  risk_level_submodel: string;
}

export interface RiskBatchResponse {
  count: number;
  results: RiskBatchResult[];
}

export interface ServicePerformanceResult {
  patient_id: string;
  health_score: number;
}

export interface ServicePerformanceResponse {
  model: string;
  count: number;
  results: ServicePerformanceResult[];
}

export interface StaffMoraleResult {
  patient_id: string;
  morale: number;
}

export interface StaffMoraleResponse {
  model: string;
  count: number;
  results: StaffMoraleResult[];
}

export interface StaffAttendanceResult {
  patient_id: string;
  attendance: number;
}

export interface StaffAttendanceResponse {
  model: string;
  count: number;
  results: StaffAttendanceResult[];
}

export interface IntegrityResult {
  patient_id: string;
  audit_flag: number;
  audit_probability: number;
}

export interface IntegrityResponse {
  model: string;
  count: number;
  results: IntegrityResult[];
}

export interface OverviewPredictionResponse {
  predicted_churn_rate: number;
  high_risk_count: number;
  sample_size: number;
  average_health_score: number;
}

// ─── API Service ──────────────────────────────────────────────────────────────

export const api = {
  /** GET /health */
  health: () => request<HealthResponse>("/health"),

  /** GET /model */
  listModels: () => request<ModelListResponse>("/model"),

  /** GET /model/{name}/health */
  modelHealth: (name: string) =>
    request<ModelHealthResponse>(`/model/${encodeURIComponent(name)}/health`),

  /** GET /business-kpi */
  businessKpi: () => request<BusinessKpiResponse>("/business-kpi"),

  /** POST /predict/churn */
  predictChurn: (patient_id: string) =>
    request<ChurnResponse>("/predict/churn", {
      method: "POST",
      body: JSON.stringify({ patient_id }),
    }),

  /** POST /predict/{concept}/batch */
  predictBatch: (
    concept: string,
    patient_ids: string[],
    submodel?: string
  ) =>
    request<PredictionResponse>(
      `/predict/${encodeURIComponent(concept)}/batch`,
      {
        method: "POST",
        body: JSON.stringify({ patient_ids, submodel }),
      }
    ),

  /** GET /predict/features/schema */
  featureSchema: () =>
    request<FeatureSchemaResponse>("/predict/features/schema"),

  /** GET /predict/features/{patient_id} */
  patientFeatures: (patient_id: string) =>
    request<Record<string, number | string | null>>(
      `/predict/features/${encodeURIComponent(patient_id)}`
    ),

  /** POST /llm */
  llm: (payload: LLMRequest) =>
    request<LLMResponse>("/llm", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  /** POST /llm/agent */
  llmAgent: (payload: LLMAgentRequest) =>
    request<LLMAgentResponse>("/llm/agent", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  /** GET /prompt/{name} */
  prompt: (name: string, version?: number) => {
    const qs = version != null ? `?version=${version}` : "";
    return request<PromptResponse>(
      `/prompt/${encodeURIComponent(name)}${qs}`
    );
  },

  // ─── Optional dedicated batch endpoints ───────────────────────────────────
  // These will 404 until you add the matching FastAPI routes. They're kept
  // so callers can be written against a stable shape. Remove them if unused.

  predictBatchChurn: (patient_ids: string[]) =>
    request<ChurnBatchResponse>("/predict/batch/churn", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictBatchRisk: (patient_ids: string[]) =>
    request<RiskBatchResponse>("/predict/batch/risk", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictServicePerformance: (patient_ids: string[]) =>
    request<ServicePerformanceResponse>("/predict/batch/service-performance", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictStaffMorale: (patient_ids: string[]) =>
    request<StaffMoraleResponse>("/predict/batch/staff-morale", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictStaffAttendance: (patient_ids: string[]) =>
    request<StaffAttendanceResponse>("/predict/batch/staff-attendance", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictIntegrityAudit: (patient_ids: string[]) =>
    request<IntegrityResponse>("/predict/batch/integrity-audit", {
      method: "POST",
      body: JSON.stringify({ patient_ids }),
    }),

  predictionOverview: () =>
    request<OverviewPredictionResponse>("/predict/overview"),
};