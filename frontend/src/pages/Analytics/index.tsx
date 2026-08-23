import { useMemo, useState, type FormEvent } from "react";
import { SalesChart } from "@/components/analytics/SalesChart";
import { ModelPerformance } from "@/components/analytics/ModelPerformance";
import { RegionalMap } from "@/components/analytics/RegionalMap";
import { useAvailableModels, usePredict } from "@/hooks/usePredictions";
import { QUANTITY_FEATURE_FIELDS, SALES_FEATURE_FIELDS } from "@/utils/constants";
import { useAppSelector } from "@/store/hooks";

const DEFAULTS: Record<string, string> = {
  day_of_week: "5",
  week_of_year: "32",
  season: "summer",
  cost: "18000",
  gross_sales: "24000",
  profit: "4000",
  rolling_mean_7: "21000",
  rolling_std_7: "1200",
  rolling_max_7: "26000",
  quantity: "12",
  model: "Innova",
  price_band: "mid",
  gender: "M",
  income_customer: "85000000",
  dealer_name: "Central Motors",
  dealer_region: "West",
  company: "Toyota",
  color: "White",
  body_style: "SUV",
  price: "28000",
  discount: "0.05",
  sales: "24000",
};

export default function AnalyticsPage() {
  const { data: models } = useAvailableModels();
  const predict = usePredict();
  const history = useAppSelector((s) => s.analytics.predictions);
  const [modelType, setModelType] = useState<"sales" | "quantity" | "both">("both");
  const [salesModel, setSalesModel] = useState("");
  const [qtyModel, setQtyModel] = useState("");
  const [features, setFeatures] = useState(DEFAULTS);

  const fields = useMemo(() => {
    if (modelType === "sales") return SALES_FEATURE_FIELDS;
    if (modelType === "quantity") return QUANTITY_FEATURE_FIELDS;
    return Array.from(new Set([...SALES_FEATURE_FIELDS, ...QUANTITY_FEATURE_FIELDS]));
  }, [modelType]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const parsed: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(features)) {
      const n = Number(v);
      parsed[k] = Number.isFinite(n) && v.trim() !== "" && !Number.isNaN(n) && v.match(/^-?\d/) ? n : v;
    }
    await predict.mutateAsync({
      model_type: modelType,
      sales_model_name: salesModel || undefined,
      quantity_model_name: qtyModel || undefined,
      features: parsed,
    });
  }

  return (
    <div className="space-y-6">
      <h1 className="page-title">Sales analytics</h1>
      <SalesChart />
      <RegionalMap />
      <ModelPerformance />
      <section className="glass-card p-5">
        <h2 className="font-semibold">Run a prediction</h2>
        <p className="subtle mb-4 text-sm">POST /predict/ using registered sales and quantity models.</p>
        <form className="space-y-4" onSubmit={onSubmit}>
          <div className="grid gap-3 md:grid-cols-3">
            <select className="rounded-xl border bg-transparent px-3 py-2" style={{ borderColor: "var(--border)" }} value={modelType} onChange={(e) => setModelType(e.target.value as typeof modelType)}>
              <option value="both">Both</option>
              <option value="sales">Sales</option>
              <option value="quantity">Quantity</option>
            </select>
            <select className="rounded-xl border bg-transparent px-3 py-2" style={{ borderColor: "var(--border)" }} value={salesModel} onChange={(e) => setSalesModel(e.target.value)}>
              <option value="">Default sales model</option>
              {(models?.sales_models ?? []).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <select className="rounded-xl border bg-transparent px-3 py-2" style={{ borderColor: "var(--border)" }} value={qtyModel} onChange={(e) => setQtyModel(e.target.value)}>
              <option value="">Default quantity model</option>
              {(models?.quantity_models ?? []).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {fields.map((field) => (
              <label key={field} className="text-sm">
                {field}
                <input
                  className="mt-1 w-full rounded-xl border bg-transparent px-3 py-2"
                  style={{ borderColor: "var(--border)" }}
                  value={features[field] ?? ""}
                  onChange={(e) => setFeatures((prev) => ({ ...prev, [field]: e.target.value }))}
                />
              </label>
            ))}
          </div>
          <button className="btn-primary" disabled={predict.isLoading}>
            {predict.isLoading ? "Predicting…" : "Predict"}
          </button>
          {predict.error instanceof Error ? <p className="text-sm text-rose-400">{predict.error.message}</p> : null}
          {predict.data ? (
            <p className="text-sm">
              Sales {predict.data.predicted_sales ?? "—"} (conf {predict.data.sales_confidence ?? "—"}) · Qty{" "}
              {predict.data.predicted_quantity ?? "—"} (conf {predict.data.quantity_confidence ?? "—"})
            </p>
          ) : null}
        </form>
        {history.length > 0 ? (
          <p className="subtle mt-4 text-xs">{history.length} predictions stored in this session.</p>
        ) : null}
      </section>
    </div>
  );
}
