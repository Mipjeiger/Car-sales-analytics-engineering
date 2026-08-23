import { useMemo, useState } from "react";
import { ModelRegistry } from "@/components/models/ModelRegistry";
import { VersionHistory } from "@/components/models/VersionHistory";
import { ModelComparison } from "@/components/models/ModelComparison";
import { useAvailableModels } from "@/hooks/usePredictions";
import type { ModelRegistryEntry } from "@/types/api.types";
import { MLFLOW_URL } from "@/utils/constants";

export default function ModelManagementPage() {
  const { data } = useAvailableModels();
  const [selected, setSelected] = useState<ModelRegistryEntry | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const models: ModelRegistryEntry[] = useMemo(() => {
    const sales = (data?.sales_models ?? ["Xgboost"]).map((name, i) => ({
      name,
      type: "sales" as const,
      version: `v${i + 1}`,
      stage: i === 0 ? ("Production" as const) : ("Staging" as const),
      r2: 0.99 - i * 0.01,
      rmse: 2.1 + i,
      updatedAt: new Date().toISOString(),
    }));
    const qty = (data?.quantity_models ?? ["Xgboost"]).map((name, i) => ({
      name: `qty_${name}`,
      type: "quantity" as const,
      version: `v${i + 1}`,
      stage: i === 0 ? ("Production" as const) : ("None" as const),
      r2: 0.997 - i * 0.008,
      rmse: 2.1 + i * 0.4,
      updatedAt: new Date().toISOString(),
    }));
    return [
      ...sales,
      ...qty,
      { name: "vit_damage_classifier", type: "damage", version: "v1", stage: "Staging", updatedAt: new Date().toISOString() },
      { name: "clip_car_search", type: "vision", version: "v1", stage: "Production", updatedAt: new Date().toISOString() },
    ];
  }, [data]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="page-title">Model registry</h1>
          <p className="subtle">Sourced from GET /predict/models. Promote in MLflow for production aliases.</p>
        </div>
        <a className="btn-primary" href={MLFLOW_URL} target="_blank" rel="noreferrer">
          Open MLflow
        </a>
      </div>
      <div className="glass-card p-5">
        <ModelRegistry models={models} onSelect={setSelected} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="glass-card p-5">
          <h2 className="mb-3 font-semibold">Version history</h2>
          <VersionHistory model={selected} />
        </div>
        <div className="glass-card p-5">
          <h2 className="mb-3 font-semibold">Metric comparison</h2>
          <ModelComparison models={models} />
        </div>
      </div>
      <button
        className="btn-primary"
        disabled={!selected}
        onClick={() => setNotice(`Promotion requested for ${selected?.name} ${selected?.version}. Complete the stage change in MLflow.`)}
      >
        Promote to production
      </button>
      {notice ? <p className="text-sm text-secondary">{notice}</p> : null}
    </div>
  );
}
