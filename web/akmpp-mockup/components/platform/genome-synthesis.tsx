"use client";

import { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { InferenceStatus } from "@/lib/inference-types";

// HANJI/INK palette — five regions in warm earthy tones
const POPULATIONS = [
  { code: "CHB", color: "#b83025", region: "EAS" },
  { code: "JPT", color: "#c54a2c", region: "EAS" },
  { code: "CHS", color: "#a02418", region: "EAS" },
  { code: "KHV", color: "#8e2118", region: "EAS" },
  { code: "CDX", color: "#7a1b13", region: "EAS" },
  { code: "CEU", color: "#5b6b3d", region: "EUR" },
  { code: "TSI", color: "#6c7d4a", region: "EUR" },
  { code: "FIN", color: "#7a8c58", region: "EUR" },
  { code: "GBR", color: "#48562d", region: "EUR" },
  { code: "IBS", color: "#8a9b66", region: "EUR" },
  { code: "YRI", color: "#a07b22", region: "AFR" },
  { code: "LWK", color: "#b8902c", region: "AFR" },
  { code: "MAG", color: "#c9a447", region: "AFR" },
  { code: "ESN", color: "#856518", region: "AFR" },
  { code: "MXL", color: "#6e6353", region: "AMR" },
  { code: "PUR", color: "#83786a", region: "AMR" },
  { code: "CLM", color: "#9a8e78", region: "AMR" },
  { code: "PEL", color: "#574d40", region: "AMR" },
  { code: "GIH", color: "#6a8a83", region: "SAS" },
  { code: "PJL", color: "#7e9a93", region: "SAS" },
];

interface Point {
  x: number;
  y: number;
  vx: number;
  vy: number;
  tx: number;
  ty: number;
  color: string;
  size: number;
}

export function GenomeSynthesisCanvas({ status }: { status: InferenceStatus }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const phaseRef = useRef<"noise" | "denoising" | "settled">("noise");
  const [stats, setStats] = useState({ epoch: 0, loss: 1.0, sampled: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const initPoints = () => {
      const rect = canvas.getBoundingClientRect();
      const W = rect.width;
      const H = rect.height;
      const NUM_POINTS = 220;
      const clusterCenters = POPULATIONS.map((p, i) => {
        const region = p.region;
        const regionAngle: Record<string, number> = {
          EAS: -Math.PI / 2,
          EUR: 0,
          AFR: Math.PI / 2,
          AMR: Math.PI,
          SAS: -Math.PI / 4,
        };
        const baseAngle = regionAngle[region] ?? 0;
        const within = (i % 5) * 0.18 - 0.36;
        const angle = baseAngle + within;
        const r = 0.32 + (i % 3) * 0.04;
        return {
          x: W / 2 + Math.cos(angle) * r * Math.min(W, H),
          y: H / 2 + Math.sin(angle) * r * Math.min(W, H) * 0.7,
          color: p.color,
        };
      });

      const points: Point[] = [];
      for (let i = 0; i < NUM_POINTS; i++) {
        const cluster = clusterCenters[i % clusterCenters.length];
        points.push({
          x: Math.random() * W,
          y: Math.random() * H,
          vx: 0,
          vy: 0,
          tx: cluster.x + (Math.random() - 0.5) * 28,
          ty: cluster.y + (Math.random() - 0.5) * 28,
          color: cluster.color,
          size: 1.4 + Math.random() * 1.4,
        });
      }
      return { points, W, H };
    };

    let scene = initPoints();

    const draw = () => {
      const rect = canvas.getBoundingClientRect();

      // hanji wash background fade
      ctx.fillStyle = "rgba(240, 233, 216, 0.22)";
      ctx.fillRect(0, 0, rect.width, rect.height);

      // ink grid
      ctx.strokeStyle = "rgba(110, 99, 83, 0.22)";
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (let i = 0; i <= 8; i++) {
        const x = (rect.width / 8) * i;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, rect.height);
      }
      for (let j = 0; j <= 6; j++) {
        const y = (rect.height / 6) * j;
        ctx.moveTo(0, y);
        ctx.lineTo(rect.width, y);
      }
      ctx.stroke();

      // axis labels
      ctx.fillStyle = "rgba(26, 22, 18, 0.55)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText("PC1", rect.width - 30, rect.height - 8);
      ctx.fillText("PC2", 8, 14);

      const phase = phaseRef.current;
      const lerp = phase === "noise" ? 0 : phase === "denoising" ? 0.06 : 0.04;

      scene.points.forEach((p) => {
        if (phase !== "noise") {
          p.vx += (p.tx - p.x) * lerp - p.vx * 0.18;
          p.vy += (p.ty - p.y) * lerp - p.vy * 0.18;
          p.x += p.vx;
          p.y += p.vy;
          if (phase === "settled") {
            p.x += (Math.random() - 0.5) * 0.25;
            p.y += (Math.random() - 0.5) * 0.25;
          }
        } else {
          p.x += (Math.random() - 0.5) * 1.0;
          p.y += (Math.random() - 0.5) * 1.0;
          if (p.x < 0) p.x = rect.width;
          if (p.x > rect.width) p.x = 0;
          if (p.y < 0) p.y = rect.height;
          if (p.y > rect.height) p.y = 0;
        }

        // ink dot — flat fill, no glow, with slight ring
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
        if (phase === "settled") {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size + 1.2, 0, Math.PI * 2);
          ctx.strokeStyle = p.color + "55";
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    draw();

    const onResize = () => {
      scene = initPoints();
    };
    const ro2 = new ResizeObserver(onResize);
    ro2.observe(canvas);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ro.disconnect();
      ro2.disconnect();
    };
  }, []);

  useEffect(() => {
    if (status === "idle") {
      phaseRef.current = "noise";
      setStats({ epoch: 0, loss: 1.0, sampled: 0 });
    } else if (status === "running") {
      phaseRef.current = "denoising";
      let epoch = 0;
      let loss = 1.0;
      let sampled = 0;
      const id = setInterval(() => {
        epoch += 1;
        loss = Math.max(0.012, loss * 0.78 + Math.random() * 0.01);
        sampled += Math.floor(8 + Math.random() * 6);
        setStats({ epoch, loss, sampled });
      }, 350);
      return () => clearInterval(id);
    } else if (status === "complete") {
      phaseRef.current = "settled";
      setStats((s) => ({ ...s, loss: 0.014 }));
    }
  }, [status]);

  return (
    <Card className="flex flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <div className="flex items-center gap-2">
          <span className="font-han text-base font-bold text-cinnabar leading-none">合</span>
          <CardTitle className="font-han">
            HybridGenoDiT — 合成 유전체 시뮬레이션
          </CardTitle>
        </div>
        <Badge variant="muted">DDIM 디노이징</Badge>
      </CardHeader>
      <CardContent className="flex-1 p-0 relative min-h-[180px]">
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

        {/* stats overlay */}
        <div className="absolute left-3 top-3 flex flex-col gap-1 border border-ink/40 bg-surface/95 backdrop-blur p-2 text-[10px] font-mono">
          <Stat
            label="phase"
            value={
              status === "complete"
                ? "settled"
                : status === "running"
                ? "denoising"
                : "noise"
            }
          />
          <Stat label="step" value={stats.epoch.toString()} />
          <Stat label="loss" value={stats.loss.toFixed(4)} />
          <Stat label="sampled" value={stats.sampled.toString()} />
        </div>

        {/* legend */}
        <div className="absolute bottom-3 right-3 flex flex-wrap gap-1.5 border border-ink/40 bg-surface/95 backdrop-blur p-2 max-w-[260px]">
          {["EAS", "EUR", "AFR", "AMR", "SAS"].map((r) => {
            const sample = POPULATIONS.find((p) => p.region === r);
            return (
              <div key={r} className="flex items-center gap-1 text-[10px] font-mono">
                <span
                  className="inline-block h-2 w-2"
                  style={{ background: sample?.color }}
                />
                <span className="text-ink/75">{r}</span>
              </div>
            );
          })}
        </div>

        {status === "idle" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="border border-ink/40 bg-surface/95 backdrop-blur px-4 py-2 text-xs text-ink/65">
              <span className="font-han text-cinnabar mr-1">▶</span>
              추론 시작 시 26 populations 잠재공간으로 합성됩니다
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-ink/55 uppercase tracking-wider">{label}</span>
      <span className="font-bold text-ink">{value}</span>
    </div>
  );
}
