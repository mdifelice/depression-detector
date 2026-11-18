import { useTranslation } from "react-i18next";
import type { PredictionExplanation } from "../api";

type Legend = Record<string, Record<string, string>>;

const NUM = "(-?\\d+(?:\\.\\d+)?)";

/**
 * Rules from LIME/Anchor are written on the encoded values, e.g.
 * "Health_condition <= 1.00" where 1 is a category code. If the feature is an
 * ordinal-encoded column, translate the interval back to the original categories.
 */
function decodeRule(rule: string, legend: Legend): string | null {
  let lower = -Infinity;
  let upper = Infinity;
  let name: string;

  let m = rule.match(new RegExp(`^${NUM} < (.+) <= ${NUM}$`));
  if (m) {
    lower = Number(m[1]);
    name = m[2];
    upper = Number(m[3]);
  } else if ((m = rule.match(new RegExp(`^(.+) <= ${NUM}$`)))) {
    name = m[1];
    upper = Number(m[2]);
  } else if ((m = rule.match(new RegExp(`^(.+) > ${NUM}$`)))) {
    name = m[1];
    lower = Number(m[2]);
  } else {
    return null;
  }

  const entries = legend[name];
  if (!entries) return null;
  const labels = Object.entries(entries)
    .filter(([code]) => Number(code) > lower && Number(code) <= upper)
    .map(([, label]) => label);
  return labels.length ? labels.join(", ") : null;
}

interface Props {
  explanation: PredictionExplanation | null;
  loading: boolean;
  error: string;
}

export default function InstanceExplanation({ explanation, loading, error }: Props) {
  const { t } = useTranslation();

  if (loading) {
    return (
      <section className="instance-explanation">
        <h2>{t("predict.explainTitle")}</h2>
        <p className="explanation-hint">{t("predict.explaining")}</p>
      </section>
    );
  }
  if (error) {
    return (
      <section className="instance-explanation">
        <h2>{t("predict.explainTitle")}</h2>
        <p className="error-text">{error}</p>
      </section>
    );
  }
  if (!explanation) return null;

  const { lime, anchor, legend } = explanation;
  const limeFeatures = lime?.features ?? [];
  const maxWeight = Math.max(...limeFeatures.map((f) => Math.abs(f.weight)), 1e-9);
  const anchorPrediction = anchor?.prediction === 1 ? t("predict.positive") : t("predict.negative");
  const belowThreshold =
    anchor?.precision != null && anchor?.threshold != null && anchor.precision < anchor.threshold;

  return (
    <section className="instance-explanation">
      <h2>{t("predict.explainTitle")}</h2>

      <div className="xai-grid">
        {lime && (
          <div className="xai-card">
            <h3>{t("predict.limeTitle")}</h3>
            {lime.error ? (
              <p className="error-text">{lime.error}</p>
            ) : (
              <>
                <p className="explanation-hint">
                  {lime.mode === "classification"
                    ? t("predict.limeHint")
                    : t("predict.limeHintRegression")}
                </p>
                <table className="lime-table">
                  <thead>
                    <tr>
                      <th>{t("predict.limeRule")}</th>
                      <th>{t("predict.limeWeight")}</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {limeFeatures.map((f) => {
                      const decoded = decodeRule(f.rule, legend);
                      return (
                        <tr key={f.rule}>
                          <td className="feature-name">
                            {f.rule}
                            {decoded && <span className="decoded-rule">= {decoded}</span>}
                          </td>
                          <td>{f.weight > 0 ? "+" : ""}{f.weight.toFixed(4)}</td>
                          <td className="bar-cell">
                            <span
                              className={`bar ${f.weight > 0 ? "positive" : "negative"}`}
                              style={{ width: `${(Math.abs(f.weight) / maxWeight) * 100}%` }}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="xai-footnote">
                  {lime.predict_proba &&
                    t("predict.limeProba", {
                      negative: (lime.predict_proba[0] * 100).toFixed(1),
                      positive: (lime.predict_proba[1] * 100).toFixed(1),
                    })}{" "}
                  {lime.score != null && t("predict.limeScore", { value: lime.score.toFixed(3) })}
                </p>
              </>
            )}
          </div>
        )}

        {anchor && (
          <div className="xai-card">
            <h3>{t("predict.anchorTitle")}</h3>
            {anchor.error ? (
              <p className="notice notice-info">
                {t("predict.anchorUnavailable")} {anchor.error}
              </p>
            ) : (
              <>
                <p className="explanation-hint">{t("predict.anchorHint")}</p>
                {anchor.anchor && anchor.anchor.length > 0 ? (
                  <>
                    <p className="anchor-if">{t("predict.anchorIf")}</p>
                    <ul className="anchor-rule">
                      {anchor.anchor.map((condition) => {
                        const decoded = decodeRule(condition, legend);
                        return (
                          <li key={condition}>
                            {condition}
                            {decoded && <span className="decoded-rule">= {decoded}</span>}
                          </li>
                        );
                      })}
                    </ul>
                    <p className="anchor-then">
                      {t("predict.anchorThen", { value: anchorPrediction })}
                    </p>
                  </>
                ) : (
                  <p>{t("predict.anchorNone", { value: anchorPrediction })}</p>
                )}
                <p className="xai-footnote">
                  {t("predict.anchorPrecision", { value: ((anchor.precision ?? 0) * 100).toFixed(1) })}
                  {" · "}
                  {t("predict.anchorCoverage", { value: ((anchor.coverage ?? 0) * 100).toFixed(1) })}
                </p>
                {belowThreshold && (
                  <p className="notice notice-warning">
                    {t("predict.anchorBelow", {
                      threshold: ((anchor.threshold ?? 0) * 100).toFixed(0),
                    })}
                  </p>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {Object.keys(legend).length > 0 && (
        <details className="encoding-legend">
          <summary>{t("predict.legendTitle")}</summary>
          <ul>
            {Object.entries(legend).map(([feature, codes]) => (
              <li key={feature}>
                <strong>{feature}:</strong>{" "}
                {Object.entries(codes)
                  .map(([code, label]) => `${code} = ${label}`)
                  .join(", ")}
              </li>
            ))}
          </ul>
        </details>
      )}

      <div className="definitions">
        <h3>{t("predict.definitionsTitle")}</h3>
        <dl>
          <dt>{t("predict.defLimeScoreTerm")}</dt>
          <dd>{t("predict.defLimeScore")}</dd>
          <dt>{t("predict.defPrecisionTerm")}</dt>
          <dd>{t("predict.defPrecision")}</dd>
          <dt>{t("predict.defCoverageTerm")}</dt>
          <dd>{t("predict.defCoverage")}</dd>
        </dl>
      </div>
    </section>
  );
}
