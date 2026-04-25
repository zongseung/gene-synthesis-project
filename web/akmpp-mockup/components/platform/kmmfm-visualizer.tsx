"use client";

import { Check, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { INFERENCE_STEPS, InferenceStatus } from "@/lib/inference-types";

const MODALITIES = [
  { han: "聞", label: "문진" },
  { han: "視", label: "초음파" },
  { han: "切", label: "생체신호" },
  { han: "基", label: "유전체" },
  { han: "望", label: "망진" },
];

const OUTPUTS = [
  { han: "證", label: "변증추론" },
  { han: "藥", label: "치료반응" },
  { han: "險", label: "위험도" },
  { han: "據", label: "약리근거" },
];

interface Props {
  status: InferenceStatus;
  activeStep: number;
  completedSteps: string[];
}

export function KMMfmVisualizer({ status, activeStep, completedSteps }: Props) {
  const running = status === "running";
  const progress =
    status === "complete"
      ? 100
      : (completedSteps.length / INFERENCE_STEPS.length) * 100;

  return (
    <Card className="flex flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div className="flex items-center gap-2">
          <span className="font-han text-base font-bold text-cinnabar leading-none">智</span>
          <CardTitle className="font-han">KH-MFM Multimodal Foundation Model</CardTitle>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="muted">latent dim 2048</Badge>
          <Badge variant="muted">DDIM × 4</Badge>
        </div>
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px] flex-1 min-h-0">
        <div className="relative aspect-[16/9] lg:aspect-auto lg:h-full min-h-[200px] border border-ink/30 bg-background overflow-hidden">
          <FlowDiagram running={running} complete={status === "complete"} />
        </div>

        <div className="flex flex-col border border-ink/30 bg-background-2 p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="label-doc text-ink/80">處理 STEPS</span>
            <span className="font-mono text-[10px] font-bold text-ink">
              {completedSteps.length}/{INFERENCE_STEPS.length}
            </span>
          </div>
          <Progress value={progress} className="mb-3" />
          <ol className="space-y-1.5 overflow-y-auto pr-1 flex-1 max-h-[180px] lg:max-h-none">
            {INFERENCE_STEPS.map((step, idx) => {
              const done = completedSteps.includes(step.id);
              const active = activeStep === idx;
              return (
                <li
                  key={step.id}
                  className={cn(
                    "flex items-start gap-2 px-1.5 py-1 text-[11px] transition-colors border-l-2",
                    done && "text-ink border-l-[#5b6b3d]",
                    active && "bg-cinnabar-soft text-ink border-l-cinnabar",
                    !done && !active && "text-ink/45 border-l-transparent"
                  )}
                >
                  <span className="mt-0.5 h-3 w-3 flex-shrink-0">
                    {done ? (
                      <Check className="h-3 w-3 text-[#5b6b3d]" />
                    ) : active ? (
                      <Loader2 className="h-3 w-3 animate-spin text-cinnabar" />
                    ) : (
                      <span className="block h-3 w-3 border border-ink/30" />
                    )}
                  </span>
                  <span className="leading-snug">{step.label}</span>
                </li>
              );
            })}
          </ol>
        </div>
      </CardContent>
    </Card>
  );
}

function FlowDiagram({ running, complete }: { running: boolean; complete: boolean }) {
  const W = 600;
  const H = 320;
  const leftX = 80;
  const coreX = W / 2;
  const rightX = W - 80;
  const coreY = H / 2;

  const modalityYs = MODALITIES.map(
    (_, i) => 40 + (i * (H - 80)) / (MODALITIES.length - 1)
  );
  const outputYs = OUTPUTS.map(
    (_, i) => 60 + (i * (H - 120)) / (OUTPUTS.length - 1)
  );

  const lineDefault = "rgba(110, 99, 83, 0.35)";
  const lineActive = "#b83025";
  const lineComplete = "#5b6b3d";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full">
      {/* corner crosshairs */}
      {[[20, 20], [W - 20, 20], [20, H - 20], [W - 20, H - 20]].map(([cx, cy], i) => (
        <g key={i} stroke="#1a1612" strokeWidth={0.8} fill="none">
          <line x1={cx - 6} y1={cy} x2={cx + 6} y2={cy} />
          <line x1={cx} y1={cy - 6} x2={cx} y2={cy + 6} />
        </g>
      ))}

      {/* input flow lines */}
      {modalityYs.map((y, i) => (
        <g key={`in-${i}`}>
          <path
            d={`M ${leftX + 22} ${y} Q ${(leftX + coreX) / 2} ${y}, ${coreX - 60} ${coreY}`}
            fill="none"
            stroke={running || complete ? lineActive : lineDefault}
            strokeWidth={running ? 1.4 : 0.9}
            strokeDasharray={running ? "4 4" : undefined}
            opacity={running ? 0.85 : 0.55}
            style={running ? { animation: `pulse-flow 2s linear ${i * 0.2}s infinite` } : undefined}
          />
          {running && (
            <circle r={2.5} fill={lineActive}>
              <animateMotion
                dur={`${1.5 + i * 0.1}s`}
                repeatCount="indefinite"
                path={`M ${leftX + 22} ${y} Q ${(leftX + coreX) / 2} ${y}, ${coreX - 60} ${coreY}`}
              />
            </circle>
          )}
        </g>
      ))}

      {/* output flow lines */}
      {outputYs.map((y, i) => (
        <g key={`out-${i}`}>
          <path
            d={`M ${coreX + 60} ${coreY} Q ${(coreX + rightX) / 2} ${y}, ${rightX - 22} ${y}`}
            fill="none"
            stroke={complete ? lineComplete : lineDefault}
            strokeWidth={complete ? 1.4 : 0.9}
            opacity={complete ? 0.85 : 0.4}
            style={
              complete
                ? { animation: `pulse-flow 2.5s linear ${i * 0.15}s infinite` }
                : undefined
            }
          />
          {complete && (
            <circle r={2.5} fill={lineComplete}>
              <animateMotion
                dur={`${1.8 + i * 0.1}s`}
                repeatCount="indefinite"
                path={`M ${coreX + 60} ${coreY} Q ${(coreX + rightX) / 2} ${y}, ${rightX - 22} ${y}`}
              />
            </circle>
          )}
        </g>
      ))}

      {/* core 印 (cinnabar square) */}
      <rect
        x={coreX - 50}
        y={coreY - 38}
        width={100}
        height={76}
        fill={running || complete ? "#b83025" : "#c3a397"}
        stroke="#8e2118"
        strokeWidth={1.5}
      />
      <rect
        x={coreX - 44}
        y={coreY - 32}
        width={88}
        height={64}
        fill="none"
        stroke="rgba(255,255,255,0.35)"
        strokeWidth={0.8}
      />
      <text
        x={coreX}
        y={coreY - 6}
        textAnchor="middle"
        fontSize={18}
        fontFamily="Noto Serif KR, serif"
        fontWeight={700}
        fill="#faf6ea"
      >
        韓醫
      </text>
      <text
        x={coreX}
        y={coreY + 18}
        textAnchor="middle"
        fontSize={9}
        fontFamily="ui-monospace, monospace"
        fontWeight={700}
        fill="rgba(250, 246, 234, 0.85)"
        letterSpacing="2"
      >
        KH-MFM
      </text>

      {/* modality plates */}
      {MODALITIES.map((m, i) => (
        <g key={m.label} transform={`translate(${leftX} ${modalityYs[i]})`}>
          <rect x={-58} y={-13} width={80} height={26} fill="#faf6ea" stroke="#1a1612" strokeWidth={1} />
          <text x={-46} y={5} fontSize={13} fontWeight={700} fill="#b83025" fontFamily="Noto Serif KR, serif">
            {m.han}
          </text>
          <text x={-30} y={4} fontSize={10} fontWeight={600} fill="#1a1612">
            {m.label}
          </text>
        </g>
      ))}

      {/* output plates */}
      {OUTPUTS.map((o, i) => (
        <g key={o.label} transform={`translate(${rightX} ${outputYs[i]})`}>
          <rect x={-22} y={-13} width={80} height={26} fill="#faf6ea" stroke="#1a1612" strokeWidth={1} />
          <text x={-12} y={5} fontSize={13} fontWeight={700} fill="#b83025" fontFamily="Noto Serif KR, serif">
            {o.han}
          </text>
          <text x={5} y={4} fontSize={10} fontWeight={600} fill="#1a1612">
            {o.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
