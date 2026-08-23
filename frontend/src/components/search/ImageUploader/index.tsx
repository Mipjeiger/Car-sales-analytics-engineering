import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import type { ImageUploaderProps } from "@/types/component.types";
import { isValidImage } from "@/utils/validators";
import { cn } from "@/utils/helpers";

export function ImageUploader({ onFile, title = "Upload a car image", subtitle = "Drag and drop or click to browse", disabled, previewUrl }: ImageUploaderProps) {
  const onDrop = useCallback(
    (files: File[]) => {
      const file = files[0];
      if (file && isValidImage(file)) onFile(file);
    },
    [onFile],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled,
    multiple: false,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"] },
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "glass-card cursor-pointer border-dashed p-8 text-center transition",
        isDragActive && "border-primary ring-2 ring-primary/40",
        disabled && "pointer-events-none opacity-60",
      )}
    >
      <input {...getInputProps()} aria-label={title} />
      {previewUrl ? (
        <img src={previewUrl} alt="Upload preview" className="mx-auto mb-4 max-h-56 rounded-xl object-contain" />
      ) : null}
      <p className="font-medium">{title}</p>
      <p className="subtle mt-1 text-sm">{subtitle}</p>
    </div>
  );
}
