"use client";

import {
  HeartPulse,
  Image as ImageIcon,
  Play,
  RotateCcw,
  Upload,
  Check,
  Loader2,
  ScrollText,
  MessagesSquare,
  Dna,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  CONSTITUTIONS,
  InferenceStatus,
  PatientInput,
  SYNDROMES,
} from "@/lib/inference-types";

interface Props {
  patient: PatientInput;
  onChange: <K extends keyof PatientInput>(key: K, value: PatientInput[K]) => void;
  onRun: () => void;
  onReset: () => void;
  status: InferenceStatus;
}

export function PatientInputPanel({ patient, onChange, onRun, onReset, status }: Props) {
  const running = status === "running";
  const complete = status === "complete";

  return (
    <Card className="flex flex-col overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
        <div className="flex items-center gap-2">
          <span className="font-han text-lg font-bold text-cinnabar leading-none">入</span>
          <CardTitle className="font-han">환자 데이터 입력</CardTitle>
        </div>
        <Badge variant="muted">5 modalities</Badge>
      </CardHeader>

      <CardContent className="space-y-5 overflow-y-auto flex-1 max-h-[calc(100vh-220px)]">
        {/* 문진 */}
        <Section han="聞" title="문진 (Text)" subtitle="HanMed-LLM 토큰화" icon={MessagesSquare}>
          <Textarea
            value={patient.symptoms}
            onChange={(e) => onChange("symptoms", e.target.value)}
            rows={4}
            placeholder="환자 호소 증상을 입력하세요…"
          />
          <div className="grid grid-cols-2 gap-2">
            <div className="space-y-1">
              <label className="label-doc">변증 · 辨證</label>
              <Select
                value={patient.syndrome}
                onChange={(e) => onChange("syndrome", e.target.value)}
              >
                {SYNDROMES.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </Select>
            </div>
            <div className="space-y-1">
              <label className="label-doc">체질 · 體質</label>
              <Select
                value={patient.constitution}
                onChange={(e) => onChange("constitution", e.target.value)}
              >
                {CONSTITUTIONS.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </Select>
            </div>
          </div>
        </Section>

        <Divider />

        {/* 생체신호 */}
        <Section han="切" title="절·맥진 (Signal)" subtitle="웨어러블 + EHR" icon={HeartPulse}>
          <div className="space-y-3">
            <Slider
              label="HRV (RMSSD)"
              unit=" ms"
              value={patient.hrv}
              onChange={(v) => onChange("hrv", v)}
              min={10}
              max={120}
            />
            <div className="grid grid-cols-2 gap-3">
              <Slider
                label="수축기 SBP"
                unit=" mmHg"
                value={patient.sbp}
                onChange={(v) => onChange("sbp", v)}
                min={90}
                max={200}
              />
              <Slider
                label="이완기 DBP"
                unit=" mmHg"
                value={patient.dbp}
                onChange={(v) => onChange("dbp", v)}
                min={50}
                max={120}
              />
            </div>
            <Slider
              label="맥박 · 脈"
              unit=" bpm"
              value={patient.pulse}
              onChange={(v) => onChange("pulse", v)}
              min={40}
              max={140}
            />
          </div>
        </Section>

        <Divider />

        {/* 초음파 */}
        <Section han="視" title="망진 (Vision)" subtitle="초음파 · 설진" icon={ImageIcon}>
          <button
            onClick={() => onChange("ultrasoundLoaded", !patient.ultrasoundLoaded)}
            className="group flex w-full items-center gap-3 border border-dashed border-ink/40 bg-surface p-3 hover:border-cinnabar transition-colors text-left"
          >
            <div className="flex h-10 w-10 items-center justify-center border border-ink/30 bg-surface-2">
              {patient.ultrasoundLoaded ? (
                <Check className="h-4 w-4 text-[#3f4b29]" />
              ) : (
                <Upload className="h-4 w-4 text-ink/55" />
              )}
            </div>
            <div className="flex-1">
              <div className="text-xs font-bold text-ink font-mono">
                {patient.ultrasoundLoaded ? "carotid_R_20260426.dcm" : "DICOM 업로드"}
              </div>
              <div className="text-[10px] text-ink/60">
                {patient.ultrasoundLoaded
                  ? "IMT 1.12mm · plaque score 2"
                  : "JPG · PNG · DICOM (≤50MB)"}
              </div>
            </div>
          </button>
        </Section>

        <Divider />

        {/* 유전체 */}
        <Section han="基" title="오믹스 (Gene)" subtitle="HybridGenoDiT 합성" icon={Dna}>
          <div className="border border-ink/30 bg-surface p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-ink/65">로드된 변이 수</span>
              <span className="font-mono text-sm font-bold text-ink">
                {patient.genomeVariants.toLocaleString()}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1 text-center text-[10px]">
              {[
                { c: "EAS", v: "98.2%", strong: true },
                { c: "EUR", v: "1.1%" },
                { c: "AFR", v: "0.7%" },
              ].map((r) => (
                <div
                  key={r.c}
                  className={`border py-1.5 ${
                    r.strong
                      ? "border-cinnabar bg-cinnabar-soft"
                      : "border-ink/20 bg-surface-2/50"
                  }`}
                >
                  <div
                    className={`font-mono font-bold ${
                      r.strong ? "text-cinnabar" : "text-ink/55"
                    }`}
                  >
                    {r.c}
                  </div>
                  <div className="text-ink/65">{r.v}</div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap gap-1 pt-1">
              {["CYP2D6*10", "ALDH2*2", "MTHFR C677T", "ApoE ε3/ε4"].map((g) => (
                <span
                  key={g}
                  className="border border-ink/30 bg-surface-2/40 px-1.5 py-0.5 text-[10px] font-mono text-ink/75"
                >
                  {g}
                </span>
              ))}
            </div>
          </div>
        </Section>
      </CardContent>

      <div className="border-t-2 border-ink p-3 space-y-2 bg-surface-2/40">
        <Button
          onClick={onRun}
          disabled={running}
          size="lg"
          className="w-full"
        >
          {running ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="font-han mr-1">推</span>
              KH-MFM 추론 중…
            </>
          ) : complete ? (
            <>
              <Play className="h-4 w-4" />
              <span className="font-han mr-1">再</span>
              다시 추론
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              <span className="font-han mr-1">推</span>
              KH-MFM 추론 시작
            </>
          )}
        </Button>
        {complete && (
          <Button onClick={onReset} variant="ghost" size="sm" className="w-full">
            <RotateCcw className="h-3 w-3" />
            결과 초기화
          </Button>
        )}
      </div>
    </Card>
  );
}

function Divider() {
  return <div className="h-px w-full bg-ink/15" />;
}

function Section({
  han,
  title,
  subtitle,
  icon: Icon,
  children,
}: {
  han: string;
  title: string;
  subtitle: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="font-han text-base font-bold text-cinnabar leading-none">
          {han}
        </span>
        <Icon className="h-3.5 w-3.5 text-ink/55" />
        <span className="text-xs font-bold text-ink">{title}</span>
        <span className="label-doc">· {subtitle}</span>
      </div>
      {children}
    </div>
  );
}
