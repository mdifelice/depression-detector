import { useState, useEffect } from "react";
import axios from "axios";
import { useParams, useNavigate } from "react-router-dom";
import {
  datasetsApi,
  type DatasetMetadata,
  type TrainingSettings,
} from "../api";

const SCALING_TYPES = ["standard", "minmax", "robust"];

function modelDisplayName(fullName: string): string {
  const parts = fullName.split(".");
  return parts[parts.length - 1];
}

export default function TrainDataset() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [settings, setSettings] = useState<TrainingSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [editingModel, setEditingModel] = useState<string | null>(null);
  const [modelParams, setModelParams] = useState<Record<string, unknown>>({});
  const [modelParamGrid, setModelParamGrid] = useState<
    Record<string, unknown> | unknown[]
  >({});

  useEffect(() => {
    if (!id) return;
    Promise.all([datasetsApi.get(id), datasetsApi.getTraining(id)]).then(
      ([metaRes, trainRes]) => {
        setMetadata(metaRes.data);
        setSettings(trainRes.data);
      }
    );
    datasetsApi
      .getTrainingStatus(id)
      .then((res) => {
        if (res.data.status === "running") {
          navigate(`/results/${id}`);
        }
      })
      .catch(() => {
        // no training job for this dataset yet
      });
  }, [id]);

  const updateField = <K extends keyof TrainingSettings>(
    field: K,
    value: TrainingSettings[K]
  ) => {
    setSettings((prev) => (prev ? { ...prev, [field]: value } : null));
  };

  const toggleModel = (modelKey: string) => {
    if (!settings) return;
    const current = settings.selected_models;
    const next = current.includes(modelKey)
      ? current.filter((m) => m !== modelKey)
      : [...current, modelKey];
    updateField("selected_models", next);
  };

  const selectAllModels = () => {
    if (!settings) return;
    const allKeys = Object.keys(settings.models);
    updateField("selected_models", allKeys);
  };

  const deselectAllModels = () => {
    updateField("selected_models", []);
  };

  const openModelEditor = (modelKey: string) => {
    if (!settings) return;
    const cfg = settings.models[modelKey] || {
      constructor_params: {},
      param_grid: {},
    };
    setEditingModel(modelKey);
    setModelParams({ ...cfg.constructor_params });
    setModelParamGrid(
      Array.isArray(cfg.param_grid)
        ? [...cfg.param_grid]
        : { ...(cfg.param_grid as Record<string, unknown>) }
    );
  };

  const saveModelConfig = () => {
    if (!settings || !editingModel) return;
    const updatedModels = { ...settings.models };
    updatedModels[editingModel] = {
      constructor_params: { ...modelParams },
      param_grid: Array.isArray(modelParamGrid)
        ? [...modelParamGrid]
        : { ...(modelParamGrid as Record<string, unknown>) },
    };
    updateField("models", updatedModels);
    setEditingModel(null);
  };

  const handleSave = async (startAfter: boolean) => {
    if (!id || !settings) return;
    setSaving(true);
    try {
      await datasetsApi.updateTraining(id, settings);
      if (startAfter) {
        await datasetsApi.startTraining(id);
        navigate(`/results/${id}`);
      } else {
        navigate("/dashboard");
      }
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : String(err);
      console.error("Save failed", err);
      alert(msg);
    } finally {
      setSaving(false);
    }
  };

  const handleJsonChange = (
    value: string,
    setter: (v: unknown) => void
  ) => {
    try {
      setter(JSON.parse(value));
    } catch {
      // ignore invalid JSON while typing
    }
  };

  if (!metadata || !settings) return <p>Loading...</p>;

  return (
    <div className="configure-dataset">
      <header className="sticky-header">
        <h1>Train: {metadata.filename}</h1>
        <button onClick={() => navigate("/dashboard")}>Back</button>
      </header>

      <div className="table-scroll-container">
        <section>
          <h2>General Settings</h2>
        <div className="form-grid">
          <label>
            <span>Scaling Type</span>
            <select
              value={settings.scaling_type}
              onChange={(e) => updateField("scaling_type", e.target.value)}
            >
              {SCALING_TYPES.map((st) => (
                <option key={st} value={st}>
                  {st}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Random Seed</span>
            <input
              type="number"
              value={settings.random_seed}
              onChange={(e) => updateField("random_seed", Number(e.target.value))}
            />
          </label>

          <label>
            <span>CV Folds</span>
            <input
              type="number"
              min={2}
              value={settings.cross_validation_folds}
              onChange={(e) =>
                updateField("cross_validation_folds", Number(e.target.value))
              }
            />
          </label>

          <label>
            <span>CV Tune Folds</span>
            <input
              type="number"
              min={2}
              value={settings.cross_validation_tune_folds}
              onChange={(e) =>
                updateField(
                  "cross_validation_tune_folds",
                  Number(e.target.value)
                )
              }
            />
          </label>

          <label>
            <span>Tune Iterations</span>
            <input
              type="number"
              min={1}
              value={settings.tune_iterations}
              onChange={(e) =>
                updateField("tune_iterations", Number(e.target.value))
              }
            />
          </label>

          <label>
            <span>Timeout per Algorithm (seconds, 0 = none)</span>
            <input
              type="number"
              min={0}
              value={settings.timeout}
              onChange={(e) => updateField("timeout", Number(e.target.value))}
            />
          </label>

          <label>
            <span>Oversampling Threshold</span>
            <input
              type="number"
              min={0}
              placeholder="None"
              value={settings.oversampling_threshold ?? ""}
              onChange={(e) =>
                updateField(
                  "oversampling_threshold",
                  e.target.value === "" ? null : Number(e.target.value)
                )
              }
            />
          </label>

          <label>
            <span>Row Acceptance Threshold</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={settings.row_acceptance_threshold}
              onChange={(e) =>
                updateField(
                  "row_acceptance_threshold",
                  Number(e.target.value)
                )
              }
            />
          </label>

          <label>
            <span>Column Acceptance Threshold</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={settings.column_acceptance_threshold}
              onChange={(e) =>
                updateField(
                  "column_acceptance_threshold",
                  Number(e.target.value)
                )
              }
            />
          </label>

          <label>
            <span>Max OHE Unique Values</span>
            <input
              type="number"
              min={2}
              value={settings.max_ohe_unique_values}
              onChange={(e) =>
                updateField("max_ohe_unique_values", Number(e.target.value))
              }
            />
          </label>

          <label>
            <span>Max Sortable Values</span>
            <input
              type="number"
              min={2}
              value={settings.max_sortable_values}
              onChange={(e) =>
                updateField("max_sortable_values", Number(e.target.value))
              }
            />
          </label>

          <label>
            <span>Correlation Acceptance Threshold</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={settings.correlation_acceptance_threshold}
              onChange={(e) =>
                updateField(
                  "correlation_acceptance_threshold",
                  Number(e.target.value)
                )
              }
            />
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={settings.tune}
              onChange={(e) => updateField("tune", e.target.checked)}
            />
            <span>Tune Hyperparameters</span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={settings.turbo}
              onChange={(e) => updateField("turbo", e.target.checked)}
            />
            <span>Turbo Mode</span>
          </label>
        </div>
      </section>

      <section>
        <h2>Models</h2>
        <div className="model-actions">
          <button onClick={selectAllModels}>Select All</button>
          <button onClick={deselectAllModels}>Deselect All</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>Enabled</th>
              <th>Model</th>
              <th>Constructor Params</th>
              <th>Param Grid</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {Object.keys(settings.models).map((modelKey) => {
              const isSelected = settings.selected_models.includes(modelKey);
              const cfg = settings.models[modelKey];
              return (
                <tr key={modelKey}>
                  <td>
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleModel(modelKey)}
                    />
                  </td>
                  <td>{modelDisplayName(modelKey)}</td>
                  <td>
                    <code>{JSON.stringify(cfg.constructor_params)}</code>
                  </td>
                  <td>
                    <code>{JSON.stringify(cfg.param_grid)}</code>
                  </td>
                  <td>
                    <button onClick={() => openModelEditor(modelKey)}>
                      Edit
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
      </div>

      {editingModel && (
        <div className="modal-overlay" onClick={() => setEditingModel(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>{modelDisplayName(editingModel)}</h2>
            <label>
              <span>Constructor Params (JSON)</span>
              <textarea
                rows={6}
                value={JSON.stringify(modelParams, null, 2)}
                onChange={(e) =>
                  handleJsonChange(
                    e.target.value,
                    (v) => setModelParams(v as Record<string, unknown>)
                  )
                }
              />
            </label>
            <label>
              <span>Param Grid (JSON)</span>
              <textarea
                rows={6}
                value={JSON.stringify(modelParamGrid, null, 2)}
                onChange={(e) =>
                  handleJsonChange(
                    e.target.value,
                    (v) =>
                      setModelParamGrid(
                        v as Record<string, unknown> | unknown[]
                      )
                  )
                }
              />
            </label>
            <div className="modal-actions">
              <button onClick={() => setEditingModel(null)}>Cancel</button>
              <button onClick={saveModelConfig}>Save</button>
            </div>
          </div>
        </div>
      )}

      <section className="sticky-footer">
        <div className="actions">
          <button onClick={() => handleSave(false)} disabled={saving}>
            {saving ? "Saving..." : "Save Training Settings"}
          </button>
          <button onClick={() => handleSave(true)} disabled={saving}>
            {saving ? "Starting..." : "Save & Start Training"}
          </button>
        </div>
      </section>
    </div>
  );
}
