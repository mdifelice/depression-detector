import { useState, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { modelsApi, type ExplanationData, type GlobalImportance } from "../api";

function formatValue(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(4);
}

function signed(v: number, digits = 4): string {
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export default function Explain() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [data, setData] = useState<ExplanationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    if (!id) return;
    modelsApi
      .getExplanation(id)
      .then((res) => {
        setData(res.data);
        if (res.data.error) setError(res.data.error);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || t("explain.fail"));
      })
      .finally(() => setLoading(false));
  }, [id]);

  // Every feature, ranked by mean |SHAP| over all explained test rows. Explanations
  // saved by older versions don't carry it, so fall back to the sample rows.
  const globalRows: GlobalImportance[] = useMemo(() => {
    if (!data) return [];
    if (data.global_importance?.length) return data.global_importance;
    return data.feature_names
      .map((feature, j) => {
        const column = data.shap_values.map((row) => row[j] ?? 0);
        const n = Math.max(column.length, 1);
        return {
          feature,
          mean_abs_shap: column.reduce((a, b) => a + Math.abs(b), 0) / n,
          mean_shap: column.reduce((a, b) => a + b, 0) / n,
        };
      })
      .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
  }, [data]);

  const header = (
    <header>
      <h1>{t("explain.title")}</h1>
      <div className="explanation-actions">
        <button onClick={() => navigate(-1)}>{t("common.back")}</button>
      </div>
    </header>
  );

  if (loading) return <div className="explanation"><p>{t("explain.loading")}</p></div>;
  if (error && !data) return (
    <div className="explanation">
      {header}
      <p className="error-text">{error}</p>
    </div>
  );

  if (!data || !data.feature_names.length) return (
    <div className="explanation">
      {header}
      <p>{t("explain.noData")}</p>
    </div>
  );

  const {
    feature_names,
    shap_values,
    feature_values,
    sample_indices,
    predicted_labels,
    proba_positive,
    charts,
  } = data;

  const version = data.generated_at ? `?v=${encodeURIComponent(data.generated_at)}` : "";
  const chartUrl = (name: string) => `/static/${id}/charts/${name}${version}`;

  const nSamples = sample_indices.length;
  const current = selected < nSamples ? selected : 0;

  const hasChart = (name: string) => charts.includes(name);
  const waterfallCharts = charts
    .filter((c) => c.startsWith("shap_waterfall_"))
    .sort((a, b) => {
      const na = parseInt(a.split("_").pop()!.replace(".png", ""), 10);
      const nb = parseInt(b.split("_").pop()!.replace(".png", ""), 10);
      return na - nb;
    });

  const maxGlobal = Math.max(...globalRows.map((r) => r.mean_abs_shap), 1e-9);

  const rowContributions = feature_names
    .map((name, j) => ({
      name,
      shap: shap_values[current]?.[j] ?? 0,
      value: feature_values[current]?.[j] ?? 0,
    }))
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  const maxAbs = Math.max(...rowContributions.map((c) => Math.abs(c.shap)), 1e-9);

  const subtitleParts = [
    data.model_name,
    data.explainer,
    data.n_explained != null ? t("explain.testRows", { count: data.n_explained }) : null,
  ].filter(Boolean);

  return (
    <div className="explanation">
      {header}

      {subtitleParts.length > 0 && (
        <p className="explanation-subtitle">{subtitleParts.join(" · ")}</p>
      )}

      {error && <p className="error-text">{error}</p>}

      {/* GLOBAL */}
      <section className="explanation-global">
        <h2>{t("explain.global")}</h2>
        <div className="chart-row">
          {hasChart("shap_beeswarm.png") && (
            <div className="chart-card">
              <img src={chartUrl("shap_beeswarm.png")} alt={t("explain.beeswarmAlt")} />
            </div>
          )}
          {hasChart("shap_importance.png") && (
            <div className="chart-card">
              <img src={chartUrl("shap_importance.png")} alt={t("explain.importanceAlt")} />
            </div>
          )}
          {hasChart("shap_dependence.png") && (
            <div className="chart-card">
              <img
                src={chartUrl("shap_dependence.png")}
                alt={t("explain.dependenceAlt", { feature: data.top_feature ?? "" })}
              />
            </div>
          )}
        </div>

        <h3>{t("explain.allFeatures", { count: globalRows.length })}</h3>
        <p className="explanation-hint">{t("explain.allFeaturesHint")}</p>
        <table className="shap-table shap-table-wide">
          <thead>
            <tr>
              <th>#</th>
              <th>{t("explain.feature")}</th>
              <th>{t("explain.meanAbs")}</th>
              <th>{t("explain.importance")}</th>
              <th>{t("explain.meanSigned")}</th>
            </tr>
          </thead>
          <tbody>
            {globalRows.map((r, i) => (
              <tr key={r.feature}>
                <td>{i + 1}</td>
                <td className="feature-name" title={r.feature}>{r.feature}</td>
                <td>{r.mean_abs_shap.toFixed(4)}</td>
                <td className="bar-cell">
                  <span
                    className="bar neutral"
                    style={{ width: `${(r.mean_abs_shap / maxGlobal) * 100}%` }}
                  />
                </td>
                <td>{signed(r.mean_shap)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* LOCAL */}
      {nSamples > 0 && (
        <section className="explanation-local">
          <h2>{t("explain.local")}</h2>

          <div className="local-controls">
            <label>
              {t("explain.sampleRow")}
              <select
                value={current}
                onChange={(e) => setSelected(Number(e.target.value))}
              >
                {sample_indices.map((idx, i) => {
                  const prob = proba_positive[i];
                  const probStr = prob != null ? ` (${(prob * 100).toFixed(1)}%)` : "";
                  return (
                    <option key={i} value={i}>
                      {t("explain.rowLabel", {
                        idx,
                        label: predicted_labels[i] === 1 ? t("predict.positive") : t("predict.negative"),
                        prob: probStr,
                      })}
                    </option>
                  );
                })}
              </select>
            </label>
          </div>

          <div className="local-charts">
            {waterfallCharts[current] && (
              <div className="chart-card waterfall-chart">
                <img
                  src={chartUrl(waterfallCharts[current])}
                  alt={t("explain.waterfallAlt", { index: sample_indices[current] })}
                />
              </div>
            )}
          </div>

          <h3>{t("explain.perFeature")}</h3>
          <table className="shap-table">
            <thead>
              <tr>
                <th>{t("explain.feature")}</th>
                <th>{t("explain.value")}</th>
                <th>{t("explain.shap")}</th>
                <th>{t("explain.effect")}</th>
              </tr>
            </thead>
            <tbody>
              {rowContributions.map((c) => {
                const barPct = (Math.abs(c.shap) / maxAbs) * 100;
                const cls = c.shap > 0 ? "positive" : "negative";
                return (
                  <tr key={c.name}>
                    <td className="feature-name" title={c.name}>{c.name}</td>
                    <td>{formatValue(c.value)}</td>
                    <td>{signed(c.shap)}</td>
                    <td className="bar-cell">
                      <span className={`bar ${cls}`} style={{ width: `${barPct}%` }} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
