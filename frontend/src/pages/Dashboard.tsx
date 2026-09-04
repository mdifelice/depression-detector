import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { datasetsApi, DatasetInfo } from "../api";
import { useAuth } from "../contexts/AuthContext";

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    const res = await datasetsApi.list();
    setDatasets(res.data);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      await datasetsApi.upload(file);
      await loadDatasets();
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      console.error("Upload failed", err);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this dataset?")) return;
    await datasetsApi.delete(id);
    await loadDatasets();
  };

  return (
    <div className="dashboard">
      <header>
        <h1>Depression Detector</h1>
        <div className="user-info">
          {user?.picture && <img src={user.picture} alt="" className="avatar" />}
          <span>{user?.name}</span>
          <button onClick={logout}>Logout</button>
        </div>
      </header>

      <main>
        <div className="actions">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx"
            onChange={handleUpload}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Upload Dataset"}
          </button>
        </div>

        <div className="dataset-list">
          {datasets.length === 0 ? (
            <p>No datasets uploaded yet.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Uploaded</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr key={ds.id}>
                    <td>{ds.filename}</td>
                    <td>{new Date(ds.uploaded_at).toLocaleString()}</td>
                    <td>
                      <button onClick={() => navigate(`/configure/${ds.id}`)}>
                        Configure
                      </button>
                      <button onClick={() => navigate(`/train/${ds.id}`)}>
                        Train
                      </button>
                      <button onClick={() => handleDelete(ds.id)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
