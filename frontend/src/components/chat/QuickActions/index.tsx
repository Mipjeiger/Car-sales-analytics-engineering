const ACTIONS = [
  { label: "Recommend a family SUV", message: "Recommend a family SUV under 400 million IDR" },
  { label: "Compare two models", message: "Compare Toyota Innova and Honda CR-V for a family of 5" },
  { label: "Ask price", message: "What is a fair price for a 2022 Audi sedan?" },
];

export function QuickActions({ onPick, disabled }: { onPick: (message: string) => void; disabled?: boolean }) {
  return (
    <div className="flex flex-wrap gap-2">
      {ACTIONS.map((a) => (
        <button key={a.label} className="btn-ghost text-xs" disabled={disabled} onClick={() => onPick(a.message)}>
          {a.label}
        </button>
      ))}
    </div>
  );
}
