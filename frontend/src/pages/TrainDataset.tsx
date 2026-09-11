import { useState, useEffect } from "react";
import axios from "axios";
import { useParams, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
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

  if (!metadata || !settings) return <p>{t("common.loading")}</p>;

  return (
    <div className="configure-dataset">
      <header className="sticky-header">
        <h1>{t("train.title", { filename: metadata.filename })}</h1>
        <button onClick={() => navigate("/dashboard")}>{t("common.back")}</button>
      </header>

      <div className="table-scroll-container">
        <section>
          <h2>{t("train.generalSettings")}</h2>
        <div className="form-grid">
          <label>
            <span>{t("train.scalingType")}</span>
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
            <span>{t("train.randomSeed")}</span>
            <input
              type="number"
              value={settings.random_seed}
              onChange={(e) => updateField("random_seed", Number(e.target.value))}
            />
          </label>

          <label>
            <span>{t("train.cvFolds")}</span>
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
            <span>{t("train.cvTuneFolds")}</span>
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
            <span>{t("train.tuneIterations")}</span>
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
            <span>{t("train.timeout")}</span>
            <input
              type="number"
              min={0}
              value={settings.timeout}
              onChange={(e) => updateField("timeout", Number(e.target.value))}
            />
          </label>

          <label>
            <span>{t("train.oversamplingThreshold")}</span>
            <input
              type="number"
              min={0}
              placeholder={t("train.none")}
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
            <span>{t("train.rowAcceptanceThreshold")}</span>
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
            <span>{t("train.columnAcceptanceThreshold")}</span>
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
            <span>{t("train.maxOhe")}</span>
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
            <span>{t("train.maxSortable")}</span>
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
            <span>{t("train.corrAcceptance")}</span>
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
            <span>{t("train.tuneHyperparams")}</span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={settings.turbo}
              onChange={(e) => updateField("turbo", e.target.checked)}
            />
            <span>{t("train.turboMode")}</span>
          </label>
        </div>
      </section>

      <section>
        <h2>{t("train.models")}</h2>
        <div className="model-actions">
          <button onClick={selectAllModels}>{t("train.selectAll")}</button>
          <button onClick={deselectAllModels}>{t("train.deselectAll")}</button>
        </div>
        <table>
          <thead>
            <tr>
              <th>{t("train.enabled")}</th>
              <th>{t("common.model")}</th>
              <th>{t("train.constructorParams")}</th>
              <th>{t("train.paramGrid")}</th>
              <th className="actions-header">{t("common.actions")}</th>
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
                    <div className="actions-row">
                      <button onClick={() => openModelEditor(modelKey)}>
                        {t("train.edit")}
                      </button>
                    </div>
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
            <h2>{t("train.editTitle", { model: modelDisplayName(editingModel) })}</h2>
            <label>
              <span>{t("train.constructorParamsJson")}</span>
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
              <span>{t("train.paramGridJson")}</span>
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
              <button onClick={() => setEditingModel(null)}>{t("common.cancel")}</button>
              <button onClick={saveModelConfig}>{t("train.save")}</button>
            </div>
          </div>
        </div>
      )}

      <section className="sticky-footer">
        <div className="actions">
          <button onClick={() => handleSave(false)} disabled={saving}>
            {saving ? t("common.saving") : t("train.saveTrainingSettings")}
          </button>
          <button onClick={() => handleSave(true)} disabled={saving}>
            {saving ? t("train.starting") : t("train.saveStartTraining")}
          </button>
        </div>
      </section>
    </div>
  );
}
