"use client";

import { useEffect, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { InferenceStatus, PatientInput } from "@/lib/inference-types";

interface Props {
  status: InferenceStatus;
  patient: PatientInput;
}

export function InferenceCards({ status, patient }: Props) {
  const isComplete = status === "complete";

  return (
    <div className="flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-130px)]">
      <DiagnosisCard show={isComplete} patient={patient} delay={0} />
      <TreatmentCard show={isComplete} delay={120} />
      <RiskGaugeCard show={isComplete} delay={240} />
      <NetworkPharmacologyCard show={isComplete} delay={360} />
    </div>
  );
}

function CardWrap({
  show,
  delay,
  children,
}: {
  show: boolean;
  delay: number;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "transition-all",
        show ? "opacity-100 translate-y-0" : "opacity-50 translate-y-1"
      )}
      style={{ transitionDelay: show ? `${delay}ms` : "0ms" }}
    >
      {children}
    </div>
  );
}

function CardLabel({ han, idx, label }: { han: string; idx: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-han text-base font-bold text-cinnabar leading-none">
        {han}
      </span>
      <CardTitle className="font-han">
        <span className="font-mono text-cinnabar mr-1.5">({idx})</span>
        {label}
      </CardTitle>
    </div>
  );
}

function DiagnosisCard({
  show,
  patient,
  delay,
}: {
  show: boolean;
  patient: PatientInput;
  delay: number;
}) {
  return (
    <CardWrap show={show} delay={delay}>
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardLabel han="證" idx="a" label="증상 → 변증 추론" />
          <Badge variant="default">XAI</Badge>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <div className="label-doc">예측 변증 · 辨證</div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="font-han text-2xl font-bold text-ink">
                {show ? patient.syndrome : "—"}
              </span>
              {show && <Badge variant="success">conf 0.87</Badge>}
            </div>
          </div>

          {show && (
            <div className="space-y-1.5 border-t border-ink/15 pt-2">
              <div className="label-doc">기여도 · SHAP</div>
              {[
                { label: "두통+어지러움 (문진)", v: 0.34 },
                { label: "수축기 혈압 148", v: 0.22 },
                { label: "HRV 42ms (낮음)", v: 0.18 },
                { label: "ALDH2*2 변이", v: 0.14 },
                { label: "IMT 1.12mm (초음파)", v: 0.12 },
              ].map((c) => (
                <div key={c.label} className="space-y-0.5">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-ink/85">{c.label}</span>
                    <span className="font-mono text-ink/60">
                      {(c.v * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Progress value={c.v * 100} />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </CardWrap>
  );
}

function TreatmentCard({ show, delay }: { show: boolean; delay: number }) {
  const treatments = [
    { name: "천마구등음(天麻鉤藤飮)", han: "方", score: 0.91, type: "處方" },
    { name: "용담사간탕(龍膽瀉肝湯)", han: "方", score: 0.74, type: "處方" },
    { name: "조구등(鉤藤) 12g", han: "草", score: 0.66, type: "本草" },
    { name: "백질려(白蒺藜) 8g", han: "草", score: 0.58, type: "本草" },
  ];
  return (
    <CardWrap show={show} delay={delay}>
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardLabel han="藥" idx="b" label="치료 반응 · 예후" />
          <Badge variant="success">RWE 매칭</Badge>
        </CardHeader>
        <CardContent className="space-y-2.5">
          {show
            ? treatments.map((t) => (
                <div key={t.name}>
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="flex items-center gap-1.5">
                      <span className="font-han text-cinnabar">{t.han}</span>
                      <span className="font-bold text-ink">{t.name}</span>
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[9px] uppercase text-ink/60 border border-ink/25 px-1">
                        {t.type}
                      </span>
                      <span className="font-mono text-ink">
                        {(t.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <Progress value={t.score * 100} />
                </div>
              ))
            : <Skeleton lines={4} />}
        </CardContent>
      </Card>
    </CardWrap>
  );
}

function RiskGaugeCard({ show, delay }: { show: boolean; delay: number }) {
  const target = 87;
  const [v, setV] = useState(0);

  useEffect(() => {
    if (!show) {
      setV(0);
      return;
    }
    const start = performance.now();
    const duration = 1100;
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / duration);
      const ease = 1 - Math.pow(1 - p, 3);
      setV(Math.round(ease * target));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [show]);

  const R = 56;
  const C = 2 * Math.PI * R;
  const offset = C - (v / 100) * C;

  return (
    <CardWrap show={show} delay={delay}>
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardLabel han="險" idx="c" label="위험도 · 조기경보" />
          <Badge variant="warning">10년 ASCVD</Badge>
        </CardHeader>
        <CardContent className="grid grid-cols-[140px_1fr] gap-3 items-center">
          <div className="relative h-[140px] w-[140px]">
            <svg viewBox="0 0 140 140" className="h-full w-full -rotate-90">
              <circle
                cx={70}
                cy={70}
                r={R}
                fill="none"
                stroke="rgba(110, 99, 83, 0.25)"
                strokeWidth={10}
              />
              <circle
                cx={70}
                cy={70}
                r={R}
                fill="none"
                stroke="#b83025"
                strokeWidth={10}
                strokeLinecap="butt"
                strokeDasharray={C}
                strokeDashoffset={offset}
                style={{ transition: "stroke-dashoffset 0.1s linear" }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="font-serif text-3xl font-bold text-ink tabular-nums leading-none">
                {v}
                <span className="text-base text-ink/55">%</span>
              </div>
              <div className="label-doc text-cinnabar mt-1">HIGH RISK</div>
            </div>
          </div>
          <div className="space-y-1.5 text-[11px]">
            {show ? (
              <>
                <RiskRow label="뇌졸중 · 中風" value={64} />
                <RiskRow label="심근경색" value={42} />
                <RiskRow label="간양상항 → 中風 전조" value={87} strong />
                <div className="mt-2 border border-cinnabar/40 bg-cinnabar-soft p-2 text-[10px] text-cinnabar-strong">
                  <span className="font-han mr-1">注</span>
                  1주 내 추적 검사 권고 · BP holter 처방
                </div>
              </>
            ) : (
              <Skeleton lines={4} />
            )}
          </div>
        </CardContent>
      </Card>
    </CardWrap>
  );
}

function RiskRow({ label, value, strong }: { label: string; value: number; strong?: boolean }) {
  return (
    <div>
      <div className="flex justify-between text-ink/85">
        <span>{label}</span>
        <span className="font-mono">{value}%</span>
      </div>
      <div className="mt-0.5 h-1 w-full border border-ink/20 bg-surface-2 overflow-hidden">
        <div
          className={strong ? "h-full bg-cinnabar" : "h-full bg-ink/55"}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function NetworkPharmacologyCard({ show, delay }: { show: boolean; delay: number }) {
  return (
    <CardWrap show={show} delay={delay}>
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardLabel han="據" idx="d" label="네트워크 약리학" />
          <Badge variant="muted">ER stress</Badge>
        </CardHeader>
        <CardContent className="space-y-3">
          <NetworkGraph show={show} />
          {show && (
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {[
                { l: "Targets", v: "28" },
                { l: "Pathways", v: "12" },
                { l: "유사환자", v: "1,284명" },
                { l: "반응률", v: "76.4%", success: true },
              ].map((s) => (
                <div
                  key={s.l}
                  className="border border-ink/25 bg-surface-2/40 p-2"
                >
                  <div className="label-doc">{s.l}</div>
                  <div
                    className={cn(
                      "font-mono font-bold",
                      s.success ? "text-[#3f4b29]" : "text-ink"
                    )}
                  >
                    {s.v}
                  </div>
                </div>
              ))}
            </div>
          )}
          {show && (
            <button className="w-full flex items-center justify-center gap-1 border border-ink/40 bg-surface py-1.5 text-[11px] font-mono uppercase tracking-wider text-ink/70 hover:bg-ink hover:text-background transition-colors">
              근거 패널 펼치기
              <ArrowUpRight className="h-3 w-3" />
            </button>
          )}
        </CardContent>
      </Card>
    </CardWrap>
  );
}

function NetworkGraph({ show }: { show: boolean }) {
  const nodes = [
    { id: "rx", label: "천마구등음", x: 30, y: 60, type: "rx" },
    { id: "h1", label: "천마", x: 100, y: 30, type: "herb" },
    { id: "h2", label: "구등", x: 100, y: 60, type: "herb" },
    { id: "h3", label: "두충", x: 100, y: 90, type: "herb" },
    { id: "t1", label: "ATF6", x: 175, y: 25, type: "target" },
    { id: "t2", label: "CHOP", x: 175, y: 55, type: "target" },
    { id: "t3", label: "GRP78", x: 175, y: 85, type: "target" },
    { id: "t4", label: "XBP1", x: 175, y: 110, type: "target" },
    { id: "p1", label: "ER stress", x: 245, y: 45, type: "pathway" },
    { id: "p2", label: "Apoptosis", x: 245, y: 90, type: "pathway" },
  ];
  const edges = [
    ["rx", "h1"], ["rx", "h2"], ["rx", "h3"],
    ["h1", "t1"], ["h1", "t2"],
    ["h2", "t2"], ["h2", "t3"],
    ["h3", "t3"], ["h3", "t4"],
    ["t1", "p1"], ["t2", "p1"], ["t3", "p1"],
    ["t2", "p2"], ["t4", "p2"],
  ];
  const colors: Record<string, string> = {
    rx: "#b83025",      // 朱
    herb: "#5b6b3d",    // 草 moss
    target: "#a07b22",  // 金 gold
    pathway: "#1a1612", // 墨 ink
  };

  return (
    <div className="relative h-[140px] w-full border border-ink/30 bg-background">
      <svg viewBox="0 0 280 140" className="h-full w-full">
        {edges.map(([a, b], i) => {
          const na = nodes.find((n) => n.id === a)!;
          const nb = nodes.find((n) => n.id === b)!;
          return (
            <line
              key={`${a}-${b}`}
              x1={na.x}
              y1={na.y}
              x2={nb.x}
              y2={nb.y}
              stroke="rgba(26, 22, 18, 0.45)"
              strokeWidth={0.5}
              strokeDasharray={show ? undefined : "2 2"}
              opacity={show ? 1 : 0.4}
              style={{
                animation: show ? `fade-up 0.6s ease-out ${i * 40}ms both` : undefined,
              }}
            />
          );
        })}
        {nodes.map((n) => (
          <g key={n.id}>
            <circle
              cx={n.x}
              cy={n.y}
              r={n.type === "rx" ? 8 : n.type === "pathway" ? 6 : 4.5}
              fill={colors[n.type]}
              opacity={show ? 0.95 : 0.4}
            />
            <text
              x={n.x}
              y={n.y - (n.type === "rx" || n.type === "pathway" ? 11 : 9)}
              textAnchor="middle"
              fontSize={n.type === "rx" || n.type === "pathway" ? 7.5 : 6.5}
              fill="#1a1612"
              fontWeight={n.type === "rx" ? 700 : 500}
            >
              {n.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

const SKELETON_WIDTHS = [82, 64, 91, 73, 86, 68, 79, 88];

function Skeleton({ lines }: { lines: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3 border border-ink/15 bg-surface-2/60"
          style={{ width: `${SKELETON_WIDTHS[i % SKELETON_WIDTHS.length]}%` }}
        />
      ))}
    </div>
  );
}
