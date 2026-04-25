import { Monitor, Glasses, Bot, Circle } from "lucide-react";
import { cn } from "@/lib/utils";
import { InferenceStatus } from "@/lib/inference-types";

const MODES = [
  { icon: Monitor, label: "PC", han: "板", active: true },
  { icon: Glasses, label: "AR", han: "目", active: false },
  { icon: Bot, label: "Robot", han: "械", active: false },
];

const STATUS_LABEL: Record<InferenceStatus, { text: string; cls: string; han: string }> = {
  idle: { text: "STAND BY", cls: "text-ink/55", han: "待" },
  running: { text: "INFERRING", cls: "text-cinnabar", han: "推" },
  complete: { text: "COMPLETE", cls: "text-[#3f4b29]", han: "證" },
};

export function ConsoleHeader({ status }: { status: InferenceStatus }) {
  const s = STATUS_LABEL[status];
  return (
    <div className="flex items-center justify-between border-b-2 border-ink bg-background-2 px-4 py-2.5">
      <div className="flex items-center gap-3">
        <span className="seal h-8 w-8 text-[15px]">證</span>
        <div className="border-l border-ink/30 pl-3">
          <h1 className="font-han text-sm font-bold text-ink leading-tight">
            CDSS 콘솔 · 臨床推論
          </h1>
          <p className="label-doc">
            KH-MFM · 환자 ID <span className="text-ink/80">P-2026-04-261</span>
          </p>
        </div>
      </div>

      <div className="flex items-center gap-0 border-2 border-ink bg-surface">
        {MODES.map((m) => {
          const Icon = m.icon;
          return (
            <button
              key={m.label}
              disabled={!m.active}
              className={cn(
                "flex items-center gap-1.5 border-r border-ink/30 last:border-r-0 px-3 py-1.5 text-xs font-bold uppercase tracking-wider transition-colors",
                m.active
                  ? "bg-ink text-background"
                  : "text-ink/35 cursor-not-allowed"
              )}
            >
              <span className="font-han text-sm leading-none">{m.han}</span>
              <Icon className="h-3 w-3" />
              {m.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2">
        <span className="font-han text-base font-bold text-cinnabar">{s.han}</span>
        <Circle className={cn("h-2 w-2 fill-current", s.cls, status === "running" && "animate-pulse")} />
        <span className={cn("label-doc", s.cls, "font-bold tracking-widest")}>
          {s.text}
        </span>
      </div>
    </div>
  );
}
