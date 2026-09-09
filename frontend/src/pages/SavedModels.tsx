import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { modelsApi, type SavedModelInfo } from "../api";

export default function SavedModels() {
  const navigate = useNavigate();
  const [models, setModels] = useState<SavedModelInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    modelsApi
      .list()
      .then((res) => setModels(res.data))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (dataset_id: string) => {
    if (!confirm("Delete this trained model?")) return;
    await modelsApi.delete(dataset_id);
    setModels((prev) => prev.filter((m) => m.dataset_id !== dataset_id));
  };

  return (
    <div className="saved-models">
      <header>
        <h1>Trained Models</h1>
        <button onClick={() => navigate("/dashboard")}>Back</button>
      </header>

      <main>
        {loading ? (
          <p>Loading...</p>
        ) : models.length === 0 ? (
          <p>No trained models yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Model</th>
                <th>F1</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>AUC</th>
                <th>Actions</th>
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
                    <button onClick={() => navigate(`/predict/${m.dataset_id}`)}>
                      Predict
                    </button>
                    <button onClick={() => handleDelete(m.dataset_id)}>
                      Delete
                    </button>
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