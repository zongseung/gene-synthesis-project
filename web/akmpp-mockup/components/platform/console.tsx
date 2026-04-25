"use client";

import { useCallback, useState } from "react";
import { DEFAULT_PATIENT, INFERENCE_STEPS, PatientInput, InferenceStatus } from "@/lib/inference-types";
import { PatientInputPanel } from "./patient-input-panel";
import { KMMfmVisualizer } from "./kmmfm-visualizer";
import { GenomeSynthesisCanvas } from "./genome-synthesis";
import { InferenceCards } from "./inference-cards";
import { DiagnosisChatbot } from "./diagnosis-chatbot";
import { ConsoleHeader } from "./console-header";

export function CdssConsole() {
  const [patient, setPatient] = useState<PatientInput>(DEFAULT_PATIENT);
  const [status, setStatus] = useState<InferenceStatus>("idle");
  const [activeStep, setActiveStep] = useState<number>(-1);
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);

  const updatePatient = useCallback(<K extends keyof PatientInput>(key: K, value: PatientInput[K]) => {
    setPatient((p) => ({ ...p, [key]: value }));
  }, []);

  const startInference = useCallback(() => {
    if (status === "running") return;
    setStatus("running");
    setActiveStep(0);
    setCompletedSteps([]);

    let cumulative = 0;
    INFERENCE_STEPS.forEach((step, idx) => {
      cumulative += step.duration;
      setTimeout(() => {
        setActiveStep(idx);
      }, cumulative - step.duration);
      setTimeout(() => {
        setCompletedSteps((prev) => [...prev, step.id]);
        if (idx === INFERENCE_STEPS.length - 1) {
          setStatus("complete");
          setActiveStep(-1);
        }
      }, cumulative);
    });
  }, [status]);

  const reset = useCallback(() => {
    setStatus("idle");
    setActiveStep(-1);
    setCompletedSteps([]);
  }, []);

  return (
    <>
      <ConsoleHeader status={status} />
      <div className="flex-1 grid gap-3 p-3 lg:grid-cols-[340px_1fr_380px] xl:grid-cols-[360px_1fr_420px]">
        <PatientInputPanel
          patient={patient}
          onChange={updatePatient}
          onRun={startInference}
          onReset={reset}
          status={status}
        />
        <div className="grid gap-3 grid-rows-[minmax(280px,_auto)_1fr] min-h-0">
          <KMMfmVisualizer
            status={status}
            activeStep={activeStep}
            completedSteps={completedSteps}
          />
          <GenomeSynthesisCanvas status={status} />
        </div>
        <InferenceCards status={status} patient={patient} />
      </div>
      <DiagnosisChatbot />
    </>
  );
}
