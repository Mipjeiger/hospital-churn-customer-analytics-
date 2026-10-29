/**
 * Centralized typed API client for the Hospital Operations FastAPI backend.
 * All requests go through /api which Vite proxies to VITE_API_BASE_URL (default: http://localhost:8000).
 */

/**
 * Resolves the backend base URL for browser requests.
 * - If VITE_API_BASE_URL is explicitly set to a valid URL, use it.
 * - If VITE_API_BASE_URL is omitted or set to the Docker-internal hostname 'http://fastapi:8000'
 *   (which the host browser cannot resolve via DNS), automatically resolve to 'http://localhost:8000'
 *   where the FastAPI container's port 8000 is mapped.
 */
function resolveApiBase(): string {
  const envUrl = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
  if (!envUrl) {
    return "http://localhost:8000";
  }
  const trimmed = envUrl.trim().replace(/\/+$/, "");
  if (typeof window !== "undefined" && trimmed.includes("fastapi:")) {
    return "http://localhost:8000";
  }
  return trimmed;
}

const BASE = resolveApiBase();

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
  return res.json() as Promise<T>;
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
    request<PredictionResponse>(`/predict/${encodeURIComponent(concept)}/batch`, {
      method: "POST",
      body: JSON.stringify({ patient_ids, submodel }),
    }),

  /** GET /predict/features/schema */
  featureSchema: () => request<FeatureSchemaResponse>("/predict/features/schema"),

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
    return request<PromptResponse>(`/prompt/${encodeURIComponent(name)}${qs}`);
  },
};

