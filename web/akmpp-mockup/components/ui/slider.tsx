"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface SliderProps {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  label?: string;
  unit?: string;
}

export function Slider({
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  className,
  label,
  unit,
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className={cn("space-y-1.5", className)}>
      {label && (
        <div className="flex items-baseline justify-between text-xs">
          <span className="text-ink/65">{label}</span>
          <span className="font-mono font-bold text-ink">
            {value}
            {unit && <span className="text-ink/55 ml-0.5">{unit}</span>}
          </span>
        </div>
      )}
      <div className="relative h-1.5 rounded-none border border-ink/30 bg-surface-2">
        <div
          className="absolute h-full bg-cinnabar"
          style={{ width: `${pct}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div
          className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-none border-2 border-ink bg-cinnabar pointer-events-none"
          style={{ left: `${pct}%` }}
        />
      </div>
    </div>
  );
}
