export type InferenceStatus = "idle" | "running" | "complete";

export interface PatientInput {
  symptoms: string;
  syndrome: string;
  constitution: string;
  hrv: number;
  sbp: number;
  dbp: number;
  pulse: number;
  ultrasoundLoaded: boolean;
  genomeVariants: number;
}

export interface InferenceStep {
  id: string;
  label: string;
  duration: number;
}

export const INFERENCE_STEPS: InferenceStep[] = [
  { id: "tokenize", label: "모달리티 토큰화 (텍스트·신호·영상)", duration: 600 },
  { id: "embed", label: "잠재공간(Latent Space) 임베딩", duration: 700 },
  { id: "denoise-1", label: "디노이징 step 1/4 (population prior)", duration: 500 },
  { id: "denoise-2", label: "디노이징 step 2/4 (FiLM conditioning)", duration: 500 },
  { id: "denoise-3", label: "디노이징 step 3/4 (cluster refinement)", duration: 500 },
  { id: "denoise-4", label: "디노이징 step 4/4 (fine detail)", duration: 500 },
  { id: "infer", label: "변증·치료 추론 + 네트워크 약리학 매핑", duration: 700 },
  { id: "explain", label: "XAI 설명 생성 (근거·기여도 분석)", duration: 500 },
];

export const DEFAULT_PATIENT: PatientInput = {
  symptoms:
    "두통이 심하고 어지러움이 동반됨. 가슴이 답답하며 잠을 잘 이루지 못함. 입이 마르고 쓴맛이 느껴짐.",
  syndrome: "간양상항(肝陽上亢)",
  constitution: "태음인",
  hrv: 42,
  sbp: 148,
  dbp: 92,
  pulse: 88,
  ultrasoundLoaded: true,
  genomeVariants: 12480,
};

export const SYNDROMES = [
  "간양상항(肝陽上亢)",
  "간기울결(肝氣鬱結)",
  "비기허(脾氣虛)",
  "신양허(腎陽虛)",
  "신음허(腎陰虛)",
  "심혈허(心血虛)",
  "담습(痰濕)",
  "기체혈어(氣滯血瘀)",
];

export const CONSTITUTIONS = ["태양인", "태음인", "소양인", "소음인"];
