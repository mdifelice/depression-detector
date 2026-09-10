import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { modelsApi, type ModelSchemaField, type PredictionResponse } from "../api";

export default function Predict() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [fields, setFields] = useState<ModelSchemaField[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");

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
          <div className="form-grid">
            {fields.map((field) => (
              <label key={field.name}>
                <span>
                  {field.name} {field.required ? "*" : ""}
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
                  <input
                    type="number"
                    step="any"
                    value={values[field.name] || ""}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                  />
                ) : field.type === "date" ? (
                  <input
                    type="date"
                    value={values[field.name] || ""}
                    onChange={(e) => handleChange(field.name, e.target.value)}
                  />
                ) : (
                  <input
                    type="text"
                    value={values[field.name] || ""}
                    placeholder={
                      field.type === "multi"
                        ? "Comma-separated values"
                        : "Enter value"
                    }
                    onChange={(e) => handleChange(field.name, e.target.value)}
                  />
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
    </div>
  );
}