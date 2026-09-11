import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { modelsApi, type ExplanationData } from "../api";
import { abbreviateName } from "../utils";

export default function Explain() {
  const { id } = useParams<{ id: string }>();
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
        setError(err.response?.data?.detail || "Failed to load explanation");
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="explanation"><p>Loading explanation...</p></div>;
  if (error && !data) return (
    <div className="explanation">
      <header>
        <h1>Model Explanation</h1>
        <button onClick={() => navigate(-1)}>Back</button>
      </header>
      <p className="error-text">{error}</p>
    </div>
  );

  if (!data || !data.feature_names.length) return (
    <div className="explanation">
      <header>
        <h1>Model Explanation</h1>
        <button onClick={() => navigate(-1)}>Back</button>
      </header>
      <p>No explanation data available.</p>
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

  const nSamples = sample_indices.length;
  if (selected >= nSamples && nSamples > 0) setSelected(0);

  const importanceChart = charts.find((c) => c === "shap_importance.png");
  const beeswarmChart  = charts.find((c) => c === "shap_beeswarm.png");
  const waterfallCharts = charts
    .filter((c) => c.startsWith("shap_waterfall_"))
    .sort((a, b) => {
      const na = parseInt(a.split("_").pop()!.replace(".png", ""), 10);
      const nb = parseInt(b.split("_").pop()!.replace(".png", ""), 10);
      return na - nb;
    });

  const rowContributions = feature_names
    .map((name, j) => ({
      name,
      shap: shap_values[selected]?.[j] ?? 0,
      value: feature_values[selected]?.[j] ?? 0,
    }))
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  const maxAbs = Math.max(...rowContributions.map((c) => Math.abs(c.shap)), 1e-9);

  return (
    <div className="explanation">
      <header>
        <h1>Model Explanation</h1>
        <div className="explanation-actions">
          <button onClick={() => navigate(-1)}>Back</button>
        </div>
      </header>

      {error && <p className="error-text">{error}</p>}

      {/* GLOBAL */}
      <section className="explanation-global">
        <h2>Global Feature Importance</h2>
        <div className="chart-row">
          {importanceChart && (
            <div className="chart-card">
              <img
                src={`/static/${id}/charts/${importanceChart}`}
                alt="Feature importance"
              />
            </div>
          )}
          {beeswarmChart && (
            <div className="chart-card">
              <img
                src={`/static/${id}/charts/${beeswarmChart}`}
                alt="SHAP beeswarm summary"
              />
            </div>
          )}
        </div>
      </section>

      {/* LOCAL */}
      {nSamples > 0 && (
        <section className="explanation-local">
          <h2>Local Explanation</h2>

          <div className="local-controls">
            <label>
              Sample row
              <select
                value={selected}
                onChange={(e) => setSelected(Number(e.target.value))}
              >
                {sample_indices.map((idx, i) => {
                  const prob = proba_positive[i];
                  const probStr = prob != null ? ` (${(prob * 100).toFixed(1)}%)` : "";
                  return (
                    <option key={i} value={i}>
                      Row {idx} — {predicted_labels[i] === 1 ? "positive" : "negative"}{probStr}
                    </option>
                  );
                })}
              </select>
            </label>
          </div>

          <div className="local-charts">
            {waterfallCharts[selected] && (
              <div className="chart-card waterfall-chart">
                <img
                  src={`/static/${id}/charts/${waterfallCharts[selected]}`}
                  alt={`Waterfall for sample row ${sample_indices[selected]}`}
                />
              </div>
            )}
          </div>

          <h3>Per-feature contributions</h3>
          <table className="shap-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>SHAP</th>
                <th>Effect</th>
              </tr>
            </thead>
            <tbody>
              {rowContributions.map((c) => {
                const barPct = (Math.abs(c.shap) / maxAbs) * 100;
                const cls = c.shap > 0 ? "positive" : "negative";
                return (
                  <tr key={c.name}>
                    <td className="feature-name" title={c.name}>{abbreviateName(c.name)}</td>
                    <td>{c.value.toFixed(4)}</td>
                    <td>{c.shap > 0 ? "+" : ""}{c.shap.toFixed(4)}</td>
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