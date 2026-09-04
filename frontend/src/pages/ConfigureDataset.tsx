import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  datasetsApi,
  DatasetMetadata,
  DatasetColumnsResponse,
  CategoricalColumnConfig,
} from "../api";

export default function ConfigureDataset() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [columns, setColumns] = useState<DatasetColumnsResponse | null>(null);
  const [ignoreColumns, setIgnoreColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [positiveValues, setPositiveValues] = useState<string[]>([]);
  const [positiveValueInput, setPositiveValueInput] = useState("");
  const [categoricalColumns, setCategoricalColumns] = useState<
    Record<string, CategoricalColumnConfig>
  >({});
  const [multiValueColumns, setMultiValueColumns] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!id) return;
    Promise.all([datasetsApi.get(id), datasetsApi.getColumns(id)]).then(
      ([metaRes, colRes]) => {
        const meta = metaRes.data;
        setMetadata(meta);
        setColumns(colRes.data);
        setIgnoreColumns(meta.ignore_columns);
        setTargetColumn(meta.target_column || "");
        setPositiveValues(meta.positive_values);
        setCategoricalColumns(meta.categorical_columns);
        setMultiValueColumns(meta.multi_value_columns);
      }
    );
  }, [id]);

  const toggleIgnore = (col: string) => {
    setIgnoreColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const toggleCategorical = (col: string) => {
    setCategoricalColumns((prev) => {
      const next = { ...prev };
      if (next[col]) {
        delete next[col];
      } else {
        next[col] = { order: [] };
      }
      return next;
    });
  };

  const toggleMultiValue = (col: string) => {
    setMultiValueColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const updateCategoricalOrder = (col: string, orderStr: string) => {
    const order = orderStr
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    setCategoricalColumns((prev) => ({
      ...prev,
      [col]: { order },
    }));
  };

  const addPositiveValue = () => {
    const val = positiveValueInput.trim();
    if (val && !positiveValues.includes(val)) {
      setPositiveValues((prev) => [...prev, val]);
      setPositiveValueInput("");
    }
  };

  const removePositiveValue = (val: string) => {
    setPositiveValues((prev) => prev.filter((v) => v !== val));
  };

  const handleSave = async () => {
    if (!id) return;
    setSaving(true);
    try {
      await datasetsApi.update(id, {
        ignore_columns: ignoreColumns,
        target_column: targetColumn || null,
        positive_values: positiveValues,
        categorical_columns: categoricalColumns,
        multi_value_columns: multiValueColumns,
      });
      navigate("/dashboard");
    } catch (err) {
      console.error("Save failed", err);
    } finally {
      setSaving(false);
    }
  };

  if (!metadata || !columns) return <p>Loading...</p>;

  const availableColumns = columns.columns.filter(
    (col) => !ignoreColumns.includes(col)
  );

  return (
    <div className="configure-dataset">
      <header>
        <h1>Configure: {metadata.filename}</h1>
        <button onClick={() => navigate("/dashboard")}>Back</button>
      </header>

      <section>
        <h2>Dataset Info</h2>
        <p>Rows: {columns.row_count}</p>
        <p>Columns: {columns.columns.length}</p>
      </section>

      <section>
        <h2>Columns</h2>
        <table>
          <thead>
            <tr>
              <th>Column</th>
              <th>Sample Values</th>
              <th>Ignore</th>
              <th>Target</th>
              <th>Categorical</th>
              <th>Multi-value</th>
            </tr>
          </thead>
          <tbody>
            {columns.columns.map((col) => (
              <tr key={col}>
                <td>{col}</td>
                <td>
                  {columns.sample
                    .slice(0, 3)
                    .map((row) => String(row[col]))
                    .join(", ")}
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={ignoreColumns.includes(col)}
                    onChange={() => toggleIgnore(col)}
                  />
                </td>
                <td>
                  <input
                    type="radio"
                    name="target"
                    checked={targetColumn === col}
                    onChange={() => setTargetColumn(col)}
                    disabled={ignoreColumns.includes(col)}
                  />
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={!!categoricalColumns[col]}
                    onChange={() => toggleCategorical(col)}
                    disabled={ignoreColumns.includes(col)}
                  />
                  {categoricalColumns[col] && (
                    <input
                      type="text"
                      placeholder="Comma-separated order"
                      value={categoricalColumns[col].order.join(", ")}
                      onChange={(e) =>
                        updateCategoricalOrder(col, e.target.value)
                      }
                    />
                  )}
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={multiValueColumns.includes(col)}
                    onChange={() => toggleMultiValue(col)}
                    disabled={ignoreColumns.includes(col)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {targetColumn && (
        <section>
          <h2>Positive Values for "{targetColumn}"</h2>
          <div>
            <input
              type="text"
              value={positiveValueInput}
              onChange={(e) => setPositiveValueInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addPositiveValue()}
              placeholder="Enter value"
            />
            <button onClick={addPositiveValue}>Add</button>
          </div>
          <div className="tag-list">
            {positiveValues.map((val) => (
              <span key={val} className="tag">
                {val}
                <button onClick={() => removePositiveValue(val)}>&times;</button>
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="actions">
        <button onClick={handleSave} disabled={saving}>
          {saving ? "Saving..." : "Save Configuration"}
        </button>
      </section>
    </div>
  );
}
