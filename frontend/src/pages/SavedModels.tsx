import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { modelsApi, type SavedModelInfo } from "../api";

export default function SavedModels() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [models, setModels] = useState<SavedModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    modelsApi
      .list()
      .then((res) => setModels(res.data))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (dataset_id: string) => {
    if (!confirm(t("models.deleteConfirm"))) return;
    await modelsApi.delete(dataset_id);
    setModels((prev) => prev.filter((m) => m.dataset_id !== dataset_id));
  };

  return (
    <div className="saved-models">
      <header>
        <h1>{t("models.title")}</h1>
        <button onClick={() => navigate("/dashboard")}>{t("common.back")}</button>
      </header>

      <main>
        {loading ? (
          <p>{t("common.loading")}</p>
        ) : models.length === 0 ? (
          <p>{t("models.empty")}</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>{t("models.dataset")}</th>
                <th>{t("models.model")}</th>
                <th>{t("metrics.f1")}</th>
                <th>{t("metrics.accuracy")}</th>
                <th>{t("metrics.precision")}</th>
                <th>{t("metrics.recall")}</th>
                <th>{t("metrics.auc")}</th>
                <th className="actions-header">{t("common.actions")}</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.dataset_id}>
                  <td>{m.filename}</td>
                  <td>{m.model_name}</td>
                  <td>{m.metrics.f1.toFixed(4)}</td>
                  <td>{m.metrics.accuracy.toFixed(4)}</td>
                  <td>{m.metrics.precision.toFixed(4)}</td>
                  <td>{m.metrics.recall.toFixed(4)}</td>
                  <td>{m.metrics.auc.toFixed(4)}</td>
                  <td>
                    <div className="actions-row">
                      <button onClick={() => navigate(`/predict/${m.dataset_id}`)}>
                        {t("models.predict")}
                      </button>
                      <button
                        onClick={() => navigate(`/explain/${m.dataset_id}`)}
                        disabled={!m.has_explanation}
                        title={m.has_explanation ? "" : t("models.explainHint")}
                      >
                        {t("models.explain")}
                      </button>
                      <button onClick={() => handleDelete(m.dataset_id)}>
                        {t("common.delete")}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </main>
    </div>
  );
}