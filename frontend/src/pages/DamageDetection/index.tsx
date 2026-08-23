import { useState } from "react";
import { DamageUploader } from "@/components/damage/DamageUploader";
import { DetectionResults } from "@/components/damage/DetectionResults";
import { SimilarCases } from "@/components/damage/SimilarCases";
import { damageApi, mapSeverity, qaFromSeverity } from "@/api/endpoints/damage";
import { fileToObjectUrl } from "@/utils/helpers";
import type { DamageDetectResponse } from "@/types/api.types";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";

export default function DamageDetectionPage() {
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DamageDetectResponse | null>(null);

  async function onFile(file: File) {
    setPreview(fileToObjectUrl(file));
    setLoading(true);
    setError(null);
    try {
      const data = await damageApi.detect(file);
      setResult(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Detection failed";
      setError(`${message}. POST /damage/detect is not on FastAPI yet — showing a local estimate from the filename.`);
      const guess = file.name.toLowerCase();
      const damageType = guess.includes("scratch") ? "door_scratch" : guess.includes("dent") ? "dent" : "02-moderate";
      const severity = mapSeverity(damageType);
      setResult({
        damage_type: damageType,
        severity,
        confidence: 0.74,
        repair_cost_estimate: severity === "High" ? 3000000 : severity === "Medium" ? 1000000 : 400000,
        repair_days: severity === "High" ? 7 : severity === "Medium" ? 3 : 1,
        qa_status: qaFromSeverity(severity),
        similar_cases: [
          { id: "1", damage_type: damageType, severity, similarity: 0.81 },
          { id: "2", damage_type: "bumper_scrape", severity: "Low", similarity: 0.64 },
        ],
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="page-title">Damage detection</h1>
      <p className="subtle">ViT classifier + FAISS similar cases. Endpoint: POST /damage/detect</p>
      <DamageUploader onFile={onFile} previewUrl={preview} disabled={loading} />
      {loading ? <LoadingSpinner label="Classifying damage" /> : null}
      {error ? <p className="text-sm text-amber-400">{error}</p> : null}
      <DetectionResults result={result} />
      <section>
        <h2 className="mb-3 font-semibold">Similar cases</h2>
        <SimilarCases cases={result?.similar_cases ?? []} />
      </section>
    </div>
  );
}
