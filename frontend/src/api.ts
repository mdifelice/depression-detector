import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
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
}

export interface CategoricalColumnConfig {
  order: string[];
}

export interface DatasetMetadata {
  id: string;
  filename: string;
  ignore_columns: string[];
  target_column: string | null;
  positive_values: string[];
  categorical_columns: Record<string, CategoricalColumnConfig>;
  multi_value_columns: string[];
  uploaded_at: string;
  uploaded_by: string;
}

export interface DatasetMetadataUpdate {
  ignore_columns?: string[];
  target_column?: string | null;
  positive_values?: string[];
  categorical_columns?: Record<string, CategoricalColumnConfig>;
  multi_value_columns?: string[];
}

export interface DatasetColumnsResponse {
  columns: string[];
  sample: Record<string, unknown>[];
  row_count: number;
}

export interface ModelConfig {
  constructor_params: Record<string, unknown>;
  param_grid: Record<string, unknown>;
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
  row_acceptance_threshold: number;
  column_acceptance_threshold: number;
  max_ohe_unique_values: number;
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
  row_acceptance_threshold?: number;
  column_acceptance_threshold?: number;
  max_ohe_unique_values?: number;
  correlation_acceptance_threshold?: number;
  selected_models?: string[];
  models?: Record<string, ModelConfig>;
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
};

export default api;
