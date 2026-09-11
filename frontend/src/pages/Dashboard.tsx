import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { datasetsApi, type DatasetInfo } from "../api";
import { useAuth } from "../contexts/AuthContext";

export default function Dashboard() {
  const { user, logout } = useAuth();
  const { t } = useTranslation();
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
    if (!confirm(t("dashboard.deleteConfirm"))) return;
    await datasetsApi.delete(id);
    await loadDatasets();
  };

  return (
    <div className="dashboard">
      <header>
        <div className="brand">
          <img src="/gemis.png" alt="Gemis" className="gemis-logo" />
          <h1>{t("dashboard.title")}</h1>
        </div>
        <div className="user-info">
          <button onClick={() => navigate("/trained-models")}>{t("dashboard.models")}</button>
          {user?.picture && <img src={user.picture} alt="" className="avatar" />}
          <span>{user?.name}</span>
          <button onClick={logout}>{t("dashboard.logout")}</button>
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
            {uploading ? t("dashboard.uploading") : t("dashboard.upload")}
          </button>
        </div>

        <div className="dataset-list">
          {datasets.length === 0 ? (
            <p>{t("dashboard.empty")}</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>{t("dashboard.filename")}</th>
                  <th>{t("dashboard.uploaded")}</th>
                  <th className="actions-header">{t("common.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr key={ds.id}>
                    <td>{ds.filename}</td>
                    <td>{new Date(ds.uploaded_at).toLocaleString()}</td>
                    <td>
                      <div className="actions-row">
                        <button onClick={() => navigate(`/configure/${ds.id}`)}>
                          {t("dashboard.configure")}
                        </button>
                        <button
                          onClick={() => navigate(`/train/${ds.id}`)}
                          disabled={!ds.configured}
                          title={ds.configured ? "" : t("dashboard.configureHint")}
                        >
                          {t("dashboard.train")}
                        </button>
                        <button
                          onClick={() => navigate(`/results/${ds.id}`)}
                          disabled={!ds.trained}
                          title={ds.trained ? "" : t("dashboard.trainHint")}
                        >
                          {t("dashboard.results")}
                        </button>
                        <button onClick={() => handleDelete(ds.id)}>
                          {t("common.delete")}
                        </button>
                      </div>
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
