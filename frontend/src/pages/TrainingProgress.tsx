import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { datasetsApi, type JobStatus, type ModelResult } from "../api";

const CHART_BASE = "http://localhost:8000";

export default function TrainingProgress() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [status, setStatus] = useState<JobStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<ModelResult[] | null>(null);
  const [charts, setCharts] = useState<string[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const loadData = async () => {
      if (!id) return;
      try {
        const statusRes = await datasetsApi.getTrainingStatus(id);
        setStatus(statusRes.data);
        if (statusRes.data.status === "completed") {
          const [resultsRes, chartsRes] = await Promise.all([
            datasetsApi.getTrainingResults(id),
            datasetsApi.getTrainingCharts(id),
          ]);
          setResults(resultsRes.data.results);
          setCharts(chartsRes.data.charts);
        }
      } catch {
        setStatus(null);
      }

      try {
        const logsRes = await datasetsApi.getTrainingLogs(id);
        setLogs(logsRes.data.logs);
      } catch {
        // no logs yet
      }
    };

    loadData();
    pollRef.current = setInterval(loadData, 2000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [id]);

  const progressPct = status ? Math.round(status.progress * 100) : 0;
  const showResults = status?.status === "completed";

  return (
    <div className="training-progress">
      <header>
        <h1>Training Progress</h1>
        <div>
          <button onClick={() => navigate("/dashboard")}>Back</button>
        </div>
      </header>

      {status && (
        <section className="status-section">
          <div className="status-row">
            <span className={`status-badge status-${status.status}`}>
              {status.status.toUpperCase()}
            </span>
            <span>{status.current_step}</span>
          </div>
          <div className="progress-bar">
            <div
              className={`progress-fill progress-${status.status}`}
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <p className="progress-text">{progressPct}%</p>
        </section>
      )}

      {showResults && results && (
        <>
          <section>
            <h2>Results</h2>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>F1</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>AUC</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr
                    key={r.model_name}
                    className={r.is_best ? "best-model-row" : ""}
                  >
                    <td>
                      {r.model_name}
                      {r.is_best && <span className="best-label">BEST</span>}
                    </td>
                    <td>{r.f1 != null ? r.f1.toFixed(4) : "-"}</td>
                    <td>{r.accuracy != null ? r.accuracy.toFixed(4) : "-"}</td>
                    <td>{r.precision != null ? r.precision.toFixed(4) : "-"}</td>
                    <td>{r.recall != null ? r.recall.toFixed(4) : "-"}</td>
                    <td>{r.auc != null ? r.auc.toFixed(4) : "-"}</td>
                    <td>
                      {r.error ? (
                        <span className="error-text">{r.error}</span>
                      ) : (
                        <span className="success-text">OK</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {charts.length > 0 && (
            <section>
              <h2>Charts</h2>
              <div className="charts-grid">
                {charts.map((chart) => (
                  <div key={chart} className="chart-card">
                    <img
                      src={`${CHART_BASE}${chart}`}
                      alt={chart}
                      loading="lazy"
                    />
                    <p>{chart}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="model-actions">
            <button onClick={() => navigate(`/predict/${id}`)}>
              Go to Prediction
            </button>
          </section>
        </>
      )}

      {status?.status === "failed" && (
        <section>
          <h2>Training Failed</h2>
          <p className="error-text">{status.error}</p>
        </section>
      )}

      <section>
        <h2>Logs</h2>
        <pre className="log-viewer">{logs.join("\n") || "No logs yet..."}</pre>
      </section>
    </div>
  );
}