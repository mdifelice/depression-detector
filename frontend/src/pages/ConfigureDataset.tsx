import { useState, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  datasetsApi,
  type DatasetMetadata,
  type DatasetColumnsResponse,
  type CategoricalColumnConfig,
  type TrainingSettings,
} from "../api";
import SortableList from "../components/SortableList";
import MultiSelect from "../components/MultiSelect";

const DEFAULT_MAX_SORTABLE_VALUES = 25;
const MAX_TARGET_VALUES = 25;

type CategoricalMode = "none" | "categorical" | "sortable";

function isNumericSample(vals: string[]): boolean {
  return (
    vals.length > 0 &&
    vals.every((v) => v.trim() !== "" && !isNaN(Number(v)))
  );
}

export default function ConfigureDataset() {
  const { id } = useParams<{ id: string }>();
  const { t } = useTranslation();
  const navigate = useNavigate();

  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [columns, setColumns] = useState<DatasetColumnsResponse | null>(null);
  const [training, setTraining] = useState<TrainingSettings | null>(null);
  const [ignoreColumns, setIgnoreColumns] = useState<string[]>([]);
  const [nullableColumns, setNullableColumns] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [positiveValues, setPositiveValues] = useState<string[]>([]);
  const [categoricalColumns, setCategoricalColumns] = useState<
    Record<string, CategoricalColumnConfig>
  >({});
  const [multiValueColumns, setMultiValueColumns] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [uniqueValuesCache, setUniqueValuesCache] = useState<
    Record<string, string[]>
  >({});

  useEffect(() => {
    if (!id) return;
    Promise.all([
      datasetsApi.get(id),
      datasetsApi.getColumns(id),
      datasetsApi.getTraining(id),
    ]).then(([metaRes, colRes, trainRes]) => {
      const meta = metaRes.data;
      setMetadata(meta);
      setColumns(colRes.data);
      setTraining(trainRes.data);
      setIgnoreColumns(meta.ignore_columns);
      setNullableColumns(meta.nullable_columns);
      setTargetColumn(meta.target_column || "");
      setPositiveValues(meta.positive_values);
      setCategoricalColumns(meta.categorical_columns);
      setMultiValueColumns(meta.multi_value_columns);
      autoGuessCategorical(meta.categorical_columns, colRes.data);
    });
  }, [id]);

  const autoGuessCategorical = (
    saved: Record<string, CategoricalColumnConfig>,
    colRes: DatasetColumnsResponse
  ) => {
    const guessed: Record<string, CategoricalColumnConfig> = {};
    let changed = false;
    for (let i = 0; i < colRes.columns.length; i++) {
      const col = colRes.columns[i];
      if (saved[col]) continue;
      const sample = colRes.sample[i] || [];
      const count = colRes.unique_value_counts[i] || 0;
      if (sample.length === 0) continue;
      if (isNumericSample(sample)) continue;

      const lowerSample = sample.map((v) => v.trim().toLowerCase());
      const isYesNo =
        count === 2 &&
        lowerSample.includes("yes") &&
        lowerSample.includes("no");
      if (isYesNo) {
        const noVal =
          sample.find((v) => v.trim().toLowerCase() === "no") || "no";
        const yesVal =
          sample.find((v) => v.trim().toLowerCase() === "yes") || "yes";
        guessed[col] = { order: [noVal, yesVal], ordinal: true };
      } else {
        guessed[col] = { order: [], ordinal: false };
      }
      changed = true;
    }
    if (changed) {
      setCategoricalColumns((prev) => ({ ...prev, ...guessed }));
    }
  };

  const fetchUniqueValues = async (col: string) => {
    if (uniqueValuesCache[col]) return uniqueValuesCache[col];
    if (!id) return [];
    try {
      const res = await datasetsApi.getColumnValues(id, col);
      const vals = res.data.unique_values;
      setUniqueValuesCache((prev) => ({ ...prev, [col]: vals }));
      return vals;
    } catch {
      return [];
    }
  };

  const uniqueValuesByCol = useMemo(() => {
    if (!columns) return {};
    const map: Record<string, string[]> = {};
    for (let i = 0; i < columns.columns.length; i++) {
      const col = columns.columns[i];
      map[col] = columns.sample[i] || [];
    }
    return map;
  }, [columns]);

  const countsByCol = useMemo(() => {
    if (!columns) return {};
    const map: Record<string, number> = {};
    for (let i = 0; i < columns.columns.length; i++) {
      map[columns.columns[i]] = columns.unique_value_counts[i] || 0;
    }
    return map;
  }, [columns]);

  const nullCountsByCol = useMemo(() => {
    if (!columns) return {};
    const map: Record<string, number> = {};
    for (let i = 0; i < columns.columns.length; i++) {
      map[columns.columns[i]] = columns.null_counts[i] || 0;
    }
    return map;
  }, [columns]);

  const maxSortableValues =
    training?.max_sortable_values ?? DEFAULT_MAX_SORTABLE_VALUES;

  const toggleIgnore = (col: string) => {
    setIgnoreColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const toggleNullable = (col: string) => {
    setNullableColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const setCategoricalMode = async (
    col: string,
    mode: CategoricalMode
  ) => {
    if (mode === "none") {
      setCategoricalColumns((prev) => {
        const next = { ...prev };
        delete next[col];
        return next;
      });
    } else if (mode === "categorical") {
      setCategoricalColumns((prev) => ({
        ...prev,
        [col]: { order: [], ordinal: false },
      }));
    } else {
      const vals = await fetchUniqueValues(col);
      if (vals.length > maxSortableValues) return;
      const order = vals.length > 0 ? vals : uniqueValuesByCol[col] || [];
      setCategoricalColumns((prev) => ({
        ...prev,
        [col]: { order, ordinal: true },
      }));
    }
  };

  const toggleMultiValue = (col: string) => {
    setMultiValueColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const updateCategoricalOrder = (col: string, order: string[]) => {
    setCategoricalColumns((prev) => ({
      ...prev,
      [col]: { ...prev[col], order },
    }));
  };

  const handleTargetChange = async (col: string) => {
    if (targetColumn === col) {
      setTargetColumn("");
      setPositiveValues([]);
      return;
    }
    const uniqueCount = countsByCol[col] || 0;
    if (uniqueCount > MAX_TARGET_VALUES) return;
    await fetchUniqueValues(col);
    setTargetColumn(col);
    setPositiveValues([]);
  };

  const handleSave = async () => {
    if (!id) return;
    setSaving(true);
    try {
      await datasetsApi.update(id, {
        ignore_columns: ignoreColumns,
        nullable_columns: nullableColumns,
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

  if (!metadata || !columns) return <p>{t("common.loading")}</p>;

  return (
    <div className="configure-dataset">
      <header className="sticky-header">
        <h1>{t("configure.title", { filename: metadata.filename })}</h1>
        <button onClick={() => navigate("/dashboard")}>{t("common.back")}</button>
      </header>

      <section>
        <h2>{t("configure.datasetInfo")}</h2>
        <p>{t("configure.rows", { count: columns.row_count })}</p>
        <p>{t("configure.columns", { count: columns.columns.length })}</p>
      </section>

      <div className="table-scroll-container">
        <table>
          <thead>
            <tr>
              <th>{t("configure.column")}</th>
              <th>{t("configure.sampleValues")}</th>
              <th>{t("configure.ignore")}</th>
              <th>{t("configure.allowNull")}</th>
              <th>{t("configure.target")}</th>
              <th>{t("configure.categorical")}</th>
              <th>{t("configure.multiValue")}</th>
            </tr>
          </thead>
          <tbody>
            {columns.columns.map((col, i) => {
              const sampleVals = columns.sample[i] || [];
              const nullCount = nullCountsByCol[col] || 0;
              const isIgnored = ignoreColumns.includes(col);
              const isTarget = targetColumn === col;
              const cfg = categoricalColumns[col];
              const isCategorical = !!cfg;
              const isOrdinal = cfg?.ordinal ?? false;
              const uniqueCount = countsByCol[col] || 0;
              const isTargetEligible = !isIgnored && uniqueCount <= MAX_TARGET_VALUES;
              const isSortableEligible =
                !isIgnored && uniqueCount <= maxSortableValues;
              const categoricalMode: CategoricalMode = !isCategorical
                ? "none"
                : isOrdinal
                ? "sortable"
                : "categorical";

              return (
                <tr key={col} className={isTarget ? "target-row" : ""}>
                  <td className="col-name" title={col}>{col}</td>
                  <td className="col-sample">
                    {sampleVals.length > 0 ? sampleVals.join(", ") : "—"}
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={isIgnored}
                      onChange={() => toggleIgnore(col)}
                    />
                  </td>
                  <td>
                    <div className="target-cell">
                      <input
                        type="checkbox"
                        checked={nullableColumns.includes(col)}
                        onChange={() => toggleNullable(col)}
                        disabled={isIgnored}
                      />
                      {nullCount > 0 && (
                        <span className="categorical-hint">
                          {t("configure.nullCount", { count: nullCount })}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="col-target">
                    <div className="target-cell">
                      <input
                        type="radio"
                        name="target"
                        checked={isTarget}
                        onChange={() => handleTargetChange(col)}
                        disabled={isIgnored || !isTargetEligible}
                        title={
                          !isTargetEligible && !isIgnored
                            ? t("configure.tooManyValues", { count: uniqueCount, max: MAX_TARGET_VALUES })
                            : ""
                        }
                      />
                      {isTarget && (
                        <div className="target-positive-values">
                          <span className="target-label">{t("configure.positive")}</span>
                          <MultiSelect
                            options={
                              uniqueValuesCache[col] ||
                              uniqueValuesByCol[col] ||
                              []
                            }
                            selected={positiveValues}
                            onChange={setPositiveValues}
                            placeholder={t("configure.selectPositive")}
                          />
                        </div>
                      )}
                      {isIgnored && (
                        <span className="categorical-hint">
                          {t("configure.columnIgnored")}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="col-categorical">
                    <div className="categorical-cell">
                      <div className="categorical-options">
                        <label className="categorical-option">
                          <input
                            type="radio"
                            name={`categorical-${col}`}
                            checked={categoricalMode === "none"}
                            onChange={() => setCategoricalMode(col, "none")}
                            disabled={isIgnored}
                          />
                          <span>{t("configure.no")}</span>
                        </label>
                        <label className="categorical-option">
                          <input
                            type="radio"
                            name={`categorical-${col}`}
                            checked={categoricalMode === "categorical"}
                            onChange={() =>
                              setCategoricalMode(col, "categorical")
                            }
                            disabled={isIgnored}
                          />
                          <span>{t("configure.categorical")}</span>
                        </label>
                        <label
                          className={`categorical-option ${
                            isSortableEligible ? "" : "disabled"
                          }`}
                          title={
                            !isSortableEligible && !isIgnored
                              ? t("configure.tooManyValues", { count: uniqueCount, max: maxSortableValues })
                              : ""
                          }
                        >
                          <input
                            type="radio"
                            name={`categorical-${col}`}
                            checked={categoricalMode === "sortable"}
                            onChange={() =>
                              setCategoricalMode(col, "sortable")
                            }
                            disabled={isIgnored || !isSortableEligible}
                          />
                          <span>
                            {t(`configure.sortable${uniqueCount > 0 ? "" : "_zero"}`, { count: uniqueCount })}
                          </span>
                        </label>
                      </div>
                      {isCategorical && isOrdinal && (
                        <>
                          <SortableList
                            values={categoricalColumns[col].order}
                            onChange={(order) =>
                              updateCategoricalOrder(col, order)
                            }
                          />
                          <span className="categorical-hint">
                            {t("configure.orderHint")}
                          </span>
                        </>
                      )}
                      {categoricalMode === "categorical" && (
                        <span className="categorical-hint">
                          {t("configure.oheHint", { count: training?.max_ohe_unique_values ?? 10 })}
                        </span>
                      )}
                    </div>
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={multiValueColumns.includes(col)}
                      onChange={() => toggleMultiValue(col)}
                      disabled={isIgnored}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section className="sticky-footer">
        <button onClick={handleSave} disabled={saving}>
          {saving ? t("common.saving") : t("configure.save")}
        </button>
      </section>
    </div>
  );
}