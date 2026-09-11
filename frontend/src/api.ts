import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  withCredentials: true,
});

export interface User {
  email: string;
  name: string;
  picture: string;
}

export interface DatasetInfo {
  id: string;
  filename: string;
  uploaded_at: string;
  uploaded_by: string;
  configured: boolean;
  trained: boolean;
}

export interface CategoricalColumnConfig {
  order: string[];
  ordinal: boolean;
}

export interface DatasetMetadata {
  id: string;
  filename: string;
  ignore_columns: string[];
  nullable_columns: string[];
  target_column: string | null;
  positive_values: string[];
  categorical_columns: Record<string, CategoricalColumnConfig>;
  multi_value_columns: string[];
  uploaded_at: string;
  uploaded_by: string;
}

export interface DatasetMetadataUpdate {
  ignore_columns?: string[];
  nullable_columns?: string[];
  target_column?: string | null;
  positive_values?: string[];
  categorical_columns?: Record<string, CategoricalColumnConfig>;
  multi_value_columns?: string[];
}

export interface DatasetColumnsResponse {
  columns: string[];
  sample: string[][];
  unique_value_counts: number[];
  null_counts: number[];
  row_count: number;
}

export interface ModelConfig {
  constructor_params: Record<string, unknown>;
  param_grid: Record<string, unknown> | unknown[];
}

export interface TrainingSettings {
  tune: boolean;
  cross_validation_folds: number;
  cross_validation_tune_folds: number;
  scaling_type: string;
  oversampling_threshold: number | null;
  turbo: boolean;
  random_seed: number;
  tune_iterations: number;
  timeout: number;
  row_acceptance_threshold: number;
  column_acceptance_threshold: number;
  max_ohe_unique_values: number;
  max_sortable_values: number;
  correlation_acceptance_threshold: number;
  selected_models: string[];
  models: Record<string, ModelConfig>;
}

export interface TrainingSettingsUpdate {
  tune?: boolean;
  cross_validation_folds?: number;
  cross_validation_tune_folds?: number;
  scaling_type?: string;
  oversampling_threshold?: number | null;
  turbo?: boolean;
  random_seed?: number;
  tune_iterations?: number;
  timeout?: number;
  row_acceptance_threshold?: number;
  column_acceptance_threshold?: number;
  max_ohe_unique_values?: number;
  max_sortable_values?: number;
  correlation_acceptance_threshold?: number;
  selected_models?: string[];
  models?: Record<string, ModelConfig>;
}

export interface JobStatus {
  dataset_id: string;
  status: string;
  progress: number;
  current_step: string;
  started_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface ModelResult {
  model_name: string;
  model_path?: string;
  f1?: number | null;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  auc?: number | null;
  is_best: boolean;
  roc_chart?: string | null;
  confusion_matrix_chart?: string | null;
  error?: string | null;
}

export interface SavedModelInfo {
  dataset_id: string;
  filename: string;
  model_name: string;
  metrics: {
    f1: number;
    accuracy: number;
    precision: number;
    recall: number;
    auc: number;
  };
  has_explanation: boolean;
}

export interface ModelSchemaField {
  name: string;
  type: string;
  options: string[];
  required: boolean;
  min?: number | null;
  max?: number | null;
}

export interface PredictionResponse {
  prediction: number;
  probability: number | null;
  label: string;
}

export interface ExplanationData {
  feature_names: string[];
  base_value: number;
  shap_values: number[][];
  feature_values: number[][];
  sample_indices: number[];
  predicted_labels: number[];
  proba_positive: (number | null)[];
  charts: string[];
  error?: string;
}

export interface ColumnValuesResponse {
  column: string;
  unique_values: string[];
}

export interface SampleRowsResponse {
  columns: string[];
  rows: Record<string, string | number>[];
}

export const authApi = {
  loginWithToken: (token: string) =>
    api.post<User>("/auth/login-token", { token }),

  logout: () => api.post("/auth/logout"),

  me: () => api.get<User>("/auth/me"),
};

export const datasetsApi = {
  list: () => api.get<DatasetInfo[]>("/datasets"),

  get: (id: string) => api.get<DatasetMetadata>(`/datasets/${id}`),

  getColumns: (id: string) =>
    api.get<DatasetColumnsResponse>(`/datasets/${id}/columns`),

  getColumnValues: (id: string, column: string) =>
    api.get<ColumnValuesResponse>(`/datasets/${id}/columns/${encodeURIComponent(column)}/values`),

  upload: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return api.post<DatasetInfo>("/datasets", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },

  update: (id: string, data: DatasetMetadataUpdate) =>
    api.patch<DatasetMetadata>(`/datasets/${id}`, data),

  delete: (id: string) => api.delete(`/datasets/${id}`),

  getTraining: (id: string) =>
    api.get<TrainingSettings>(`/datasets/${id}/training`),

  getAvailableModels: (id: string) =>
    api.get<string[]>(`/datasets/${id}/training/models`),

  updateTraining: (id: string, data: TrainingSettingsUpdate) =>
    api.patch<TrainingSettings>(`/datasets/${id}/training`, data),

  startTraining: (id: string) =>
    api.post<JobStatus>(`/datasets/${id}/train`),

  getTrainingStatus: (id: string) =>
    api.get<JobStatus>(`/datasets/${id}/train/status`),

  getTrainingLogs: (id: string) =>
    api.get<{ logs: string[] }>(`/datasets/${id}/train/logs`),

  getTrainingResults: (id: string) =>
    api.get<{ results: ModelResult[] }>(`/datasets/${id}/train/results`),

  getTrainingCharts: (id: string) =>
    api.get<{ charts: string[] }>(`/datasets/${id}/train/charts`),
};

export const modelsApi = {
  list: () => api.get<SavedModelInfo[]>("/models"),

  get: (id: string) => api.get<SavedModelInfo>(`/models/${id}`),

  getSchema: (id: string) => api.get<{ fields: ModelSchemaField[]; target_column: string | null }>(`/models/${id}/schema`),

  predict: (id: string, values: Record<string, string>) =>
    api.post<PredictionResponse>(`/models/${id}/predict`, { values }),

  getExplanation: (id: string) =>
    api.get<ExplanationData>(`/models/${id}/explanation`),

  getPredictionLogs: (id: string) =>
    api.get<{ logs: Record<string, unknown>[] }>(`/models/${id}/prediction-logs`),

  getSampleRows: (
    id: string,
    params: { n?: number; random?: boolean } = {}
  ) =>
    api.get<SampleRowsResponse>(`/models/${id}/sample-rows`, {
      params: { n: params.n ?? 50, random: params.random ?? false },
    }),

  delete: (id: string) => api.delete(`/models/${id}`),
};

export default api;