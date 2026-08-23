import { ImageUploader } from "@/components/search/ImageUploader";

export function DamageUploader({
  onFile,
  previewUrl,
  disabled,
}: {
  onFile: (file: File) => void;
  previewUrl?: string | null;
  disabled?: boolean;
}) {
  return (
    <ImageUploader
      onFile={onFile}
      previewUrl={previewUrl}
      disabled={disabled}
      title="Upload damage photo"
      subtitle="Bumper, door, lamp, or body panel"
    />
  );
}
