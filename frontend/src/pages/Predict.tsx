import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { modelsApi, type ModelSchemaField, type PredictionResponse } from "../api";
import { abbreviateName } from "../utils";

export default function Predict() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [fields, setFields] = useState<ModelSchemaField[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [browseOpen, setBrowseOpen] = useState(false);
  const [browseRows, setBrowseRows] = useState<Record<string, string | number>[]>([]);
  const [browseLoading, setBrowseLoading] = useState(false);
  const [browseColumns, setBrowseColumns] = useState<string[]>([]);
  const [prefillingRandom, setPrefillingRandom] = useState(false);

  useEffect(() => {
    if (!id) return;
    modelsApi
      .getSchema(id)
      .then((res) => {
        setFields(res.data.fields);
        const initial: Record<string, string> = {};
        res.data.fields.forEach((f) => {
          initial[f.name] = "";
        });
        setValues(initial);
      })
      .finally(() => setLoading(false));
  }, [id]);

  const applyRow = (record: Record<string, string | number>) => {
    setValues((prev) => {
      const next = { ...prev };
      for (const key of Object.keys(record)) {
        if (key in next) next[key] = String(record[key]);
      }
      return next;
    });
    setPrediction(null);
  };

  const handleRandom = async () => {
    if (!id) return;
    setPrefillingRandom(true);
    try {
      const res = await modelsApi.getSampleRows(id, { n: 1, random: true });
      if (res.data.rows.length > 0) applyRow(res.data.rows[0]);
    } catch {
      setError("Could not load a random sample row");
    } finally {
      setPrefillingRandom(false);
    }
  };

  const openBrowse = async () => {
    if (!id) return;
    setBrowseOpen(true);
    setBrowseLoading(true);
    try {
      const res = await modelsApi.getSampleRows(id, { n: 100 });
      setBrowseColumns(res.data.columns);
      setBrowseRows(res.data.rows);
    } catch {
      setBrowseRows([]);
    } finally {
      setBrowseLoading(false);
    }
  };

  const handleChange = (name: string, value: string) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    setPrediction(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!id) return;
    setPredicting(true);
    setError("");
    try {
      const res = await modelsApi.predict(id, values);
      setPrediction(res.data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  if (loading) return <p>Loading...</p>;

  return (
    <div className="predict-page">
      <header>
        <h1>Make a Prediction</h1>
        <button onClick={() => navigate("/trained-models")}>Back</button>
      </header>

      <main>
        <form onSubmit={handleSubmit}>
          <div className="prefill-actions">
            <span className="prefill-label">Prefill from a real record:</span>
            <button
              type="button"
              onClick={handleRandom}
              disabled={prefillingRandom}
            >
              {prefillingRandom ? "Loading..." : "Random record"}
            </button>
            <button type="button" onClick={openBrowse}>
              Browse records...
            </button>
          </div>

          <div className="form-grid">
            {fields.map((field) => (
              <label key={field.name}>
                <span title={field.name}>
                  {abbreviateName(field.name)} {field.required ? "*" : ""}
                </span>
                {field.type === "select" ? (
                  <select
                    value={values[field.name] || ""}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                  >
                    <option value="">Select...</option>
                    {field.options.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>
                ) : field.type === "number" ? (
                  <>
                    <input
                      type="number"
                      step="any"
                      value={values[field.name] || ""}
                      onChange={(e) => handleChange(field.name, e.target.value)}
                    />
                    {field.min != null && field.max != null && (
                      <span className="field-guide">
                        Range: {field.min} – {field.max}
                      </span>
                    )}
                  </>
                ) : field.type === "date" ? (
                  <input
                    type="date"
                    value={values[field.name] || ""}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                  />
                ) : field.type === "multi" ? (
                  <>
                    <input
                      type="text"
                      value={values[field.name] || ""}
                      placeholder="Comma-separated values"
                      onChange={(e) => handleChange(field.name, e.target.value)}
                    />
                    {field.options.length > 0 && (
                      <span className="field-guide">
                        Possible values: {field.options.join(", ")}
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    <input
                      type="text"
                      value={values[field.name] || ""}
                      placeholder="Enter value"
                      onChange={(e) => handleChange(field.name, e.target.value)}
                    />
                    {field.options.length > 0 && (
                      <span className="field-guide">
                        Possible values: {field.options.join(", ")}
                      </span>
                    )}
                  </>
                )}
              </label>
            ))}
          </div>

          <section className="actions">
            <button type="submit" disabled={predicting}>
              {predicting ? "Predicting..." : "Predict"}
            </button>
          </section>
        </form>

        {error && <p className="error-text">{error}</p>}

        {prediction && (
          <section className="prediction-result">
            <h2>Prediction Result</h2>
            <div className="prediction-card">
              <span className={`prediction-badge ${prediction.label.includes("positive") ? "badge-positive" : "badge-negative"}`}>
                {prediction.label.toUpperCase()}
              </span>
              <p>
                Prediction: {prediction.prediction === 1 ? "Positive" : "Negative"}
              </p>
              {prediction.probability != null && (
                <p>Probability: {prediction.probability.toFixed(4)}</p>
              )}
            </div>
          </section>
        )}
      </main>

      {browseOpen && (
        <div className="modal-overlay" onClick={() => setBrowseOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Select a record to prefill</h2>
            {browseLoading ? (
              <p>Loading records...</p>
            ) : browseRows.length === 0 ? (
              <p>No records available.</p>
            ) : (
              <div className="browse-table-scroll">
                <table className="browse-table">
                  <thead>
                    <tr>
                      {browseColumns.map((c) => (
                        <th key={c} title={c}>{abbreviateName(c)}</th>
                      ))}
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {browseRows.map((row, i) => (
                      <tr key={i}>
                        {browseColumns.map((c) => (
                          <td key={c} title={String(row[c] ?? "")}>
                            {String(row[c] ?? "")}
                          </td>
                        ))}
                        <td>
                          <button
                            type="button"
                            onClick={() => {
                              applyRow(row);
                              setBrowseOpen(false);
                            }}
                          >
                            Use
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div className="modal-actions">
              <button type="button" onClick={() => setBrowseOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}