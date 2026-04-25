import Link from "next/link";
import {
  ArrowRight,
  Database,
  Cpu,
  LayoutDashboard,
  Bot,
  Eye,
  MessageSquare,
  Activity,
  Dna,
  ScrollText,
} from "lucide-react";
import { buttonVariants } from "@/components/ui/button";

const PIPELINE = [
  {
    step: "01",
    han: "資",
    title: "Data Layer",
    subtitle: "데이터레이크",
    desc: "초음파·생체신호·EHR·WGS/오믹스 통합 표준화",
    icon: Database,
  },
  {
    step: "02",
    han: "智",
    title: "Model Layer",
    subtitle: "KH-MFM",
    desc: "한의 임상표현형 기반 멀티모달 파운데이션 모델",
    icon: Cpu,
  },
  {
    step: "03",
    han: "板",
    title: "Platform Layer",
    subtitle: "CDSS",
    desc: "추론·설명·추천 임상의사결정지원시스템",
    icon: LayoutDashboard,
  },
  {
    step: "04",
    han: "證",
    title: "Validation Layer",
    subtitle: "Physical AI · AR",
    desc: "진료보조 로봇 및 가상환경 알파테스트",
    icon: Bot,
  },
];

const MODALITIES = [
  { icon: Eye, han: "視", label: "망진", desc: "Vision · 얼굴·설진" },
  { icon: MessageSquare, han: "聞", label: "문진", desc: "Text · 대화·문답" },
  { icon: Activity, han: "切", label: "절·맥진", desc: "Signal · 생체·촉진" },
  { icon: Dna, han: "基", label: "오믹스", desc: "Gene · WGS·메타" },
];

const MODULES = [
  {
    han: "生",
    badge: "Module A",
    title: "생태계 (Ecosystem)",
    desc: "AI 기반 앱·프로그램 생태계 구축, 임상-연구 폐쇄루프 형성",
  },
  {
    han: "藥",
    badge: "Module B",
    title: "약물유전체 (Pharmacogenomics)",
    desc: "전장유전체 기반 환자 아형 규명 및 합성 유전체 시뮬레이션",
  },
  {
    han: "刀",
    badge: "Module C",
    title: "유전자가위",
    desc: "AI 기반 프라임에디터 설계 지원 (초정밀 타겟 검증)",
  },
  {
    han: "波",
    badge: "Module D",
    title: "초음파·신호",
    desc: "초음파 자동 해석 및 웨어러블 생체신호 결합",
  },
];

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="relative overflow-hidden border-b-2 border-ink">
        {/* 韓 watermark */}
        <div
          className="pointer-events-none absolute -right-10 top-1/2 -translate-y-1/2 select-none font-han text-[42rem] font-bold leading-none text-ink/[0.04]"
          aria-hidden
        >
          韓
        </div>

        <div className="relative mx-auto grid max-w-[1600px] items-center gap-12 px-6 py-20 lg:grid-cols-[1.15fr_1fr]">
          <div className="space-y-6">
            <div className="flex items-center gap-3">
              <span className="seal h-6 w-6 text-[11px]">韓</span>
              <span className="label-doc">PROPOSAL · DATA & AI · 2026</span>
              <span className="h-px flex-1 max-w-[80px] bg-ink/30" />
              <span className="label-doc">DOC 01 / 04</span>
            </div>

            <h1 className="font-han text-5xl font-bold leading-[1.05] tracking-tight text-ink lg:text-7xl">
              韓醫藥
              <br />
              <span className="text-cinnabar">精密醫療</span>
              <br />
              <span className="text-3xl text-ink/70 lg:text-4xl font-sans font-bold">
                Precision Korean Medicine Platform
              </span>
            </h1>

            <div className="flex items-start gap-3 border-l-2 border-cinnabar pl-4">
              <p className="max-w-xl text-sm leading-relaxed text-ink/80">
                한의 임상표현형(<span className="font-han text-ink">望·聞·問·切</span>)을
                멀티모달화하고, 영상·생체신호·유전체와 통합한 파운데이션 모델로{" "}
                <span className="font-semibold text-ink">설명가능한
                임상추론(XAI) CDSS</span>를 구현합니다.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Link href="/platform" className={buttonVariants({ size: "lg" })}>
                CDSS 콘솔 열기
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="#modules"
                className={buttonVariants({ variant: "outline", size: "lg" })}
              >
                모듈 살펴보기
              </Link>
            </div>

          </div>

          {/* Modality → Core diagram (HANJI version) */}
          <div className="relative aspect-square w-full max-w-[500px] mx-auto">
            <svg viewBox="0 0 500 500" className="absolute inset-0 h-full w-full">
              {/* concentric ink rings */}
              {[200, 158, 116].map((r, i) => (
                <circle
                  key={r}
                  cx={250}
                  cy={250}
                  r={r}
                  fill="none"
                  stroke="rgba(26, 22, 18, 0.18)"
                  strokeWidth={i === 0 ? 1.2 : 0.8}
                  strokeDasharray={i === 0 ? undefined : "2 6"}
                />
              ))}
              {/* corner crosshair marks */}
              {[[40, 40], [460, 40], [40, 460], [460, 460]].map(([cx, cy], i) => (
                <g key={i} stroke="#1a1612" strokeWidth={1.2} fill="none">
                  <line x1={cx - 8} y1={cy} x2={cx + 8} y2={cy} />
                  <line x1={cx} y1={cy - 8} x2={cx} y2={cy + 8} />
                </g>
              ))}
              {/* flow lines */}
              {MODALITIES.map((_, i) => {
                const angle = (i / MODALITIES.length) * Math.PI * 2 - Math.PI / 2;
                const x = 250 + Math.cos(angle) * 200;
                const y = 250 + Math.sin(angle) * 200;
                return (
                  <line
                    key={i}
                    x1={x}
                    y1={y}
                    x2={250}
                    y2={250}
                    stroke="rgba(184, 48, 37, 0.4)"
                    strokeWidth={1.2}
                    strokeDasharray="3 5"
                    style={{
                      animation: `pulse-flow 3s ease-in-out ${i * 0.4}s infinite`,
                    }}
                  />
                );
              })}
              {/* core square (印) */}
              <rect
                x={250 - 58}
                y={250 - 58}
                width={116}
                height={116}
                fill="#b83025"
                stroke="#8e2118"
                strokeWidth={1.5}
              />
              <rect
                x={250 - 50}
                y={250 - 50}
                width={100}
                height={100}
                fill="none"
                stroke="rgba(255,255,255,0.35)"
                strokeWidth={1}
              />
            </svg>
            {/* Core label */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center select-none">
              <div className="font-han text-3xl font-bold text-[#faf6ea] leading-none">
                韓醫
              </div>
              <div className="mt-1 text-[10px] font-mono uppercase tracking-widest text-[#faf6ea]/85">
                KH-MFM
              </div>
              <div className="text-[8px] text-[#faf6ea]/70 tracking-wider">
                Foundation
              </div>
            </div>
            {/* Modality plates */}
            {MODALITIES.map((m, i) => {
              const angle = (i / MODALITIES.length) * Math.PI * 2 - Math.PI / 2;
              const left = `calc(50% + ${Math.cos(angle) * 200}px - 42px)`;
              const top = `calc(50% + ${Math.sin(angle) * 200}px - 42px)`;
              const Icon = m.icon;
              return (
                <div
                  key={m.label}
                  className="absolute flex h-[84px] w-[84px] flex-col items-center justify-center gap-0.5 rounded-sm border-2 border-ink bg-surface shadow-[2px_2px_0_0_rgba(26,22,18,0.85)]"
                  style={{ left, top }}
                >
                  <span className="font-han text-xl font-bold text-cinnabar leading-none">
                    {m.han}
                  </span>
                  <Icon className="h-3 w-3 text-ink/60" />
                  <span className="text-[10px] font-bold text-ink leading-none">
                    {m.label}
                  </span>
                  <span className="text-[8px] text-ink/55 tracking-wider">
                    {m.desc.split(" · ")[0]}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="relative border-b-2 border-ink bg-background-2">
        <div className="mx-auto max-w-[1600px] px-6 py-16">
          <div className="mb-10 flex items-end justify-between gap-6 flex-wrap">
            <div className="max-w-2xl">
              <div className="flex items-center gap-3 mb-2">
                <span className="seal h-6 w-6 text-[10px]">章</span>
                <span className="label-doc">CHAPTER · 01 · MASTER PIPELINE</span>
              </div>
              <h2 className="font-han text-3xl font-bold tracking-tight text-ink">
                4단계 전주기 파이프라인
              </h2>
              <p className="mt-2 text-sm text-ink/70 leading-relaxed">
                데이터 수집에서 실증까지 끝나는 선순환 구조 — 국가 연구인프라 확충
              </p>
            </div>
            <span className="label-doc">DOC · 02 / 04</span>
          </div>

          <div className="grid grid-cols-1 gap-0 md:grid-cols-2 lg:grid-cols-4 border-2 border-ink bg-surface">
            {PIPELINE.map((p, i) => {
              const Icon = p.icon;
              return (
                <div
                  key={p.step}
                  className={`group relative p-6 transition-colors hover:bg-surface-2 ${
                    i > 0 ? "border-l-0 lg:border-l border-ink/30" : ""
                  } ${i > 0 && i < 4 ? "border-t md:border-t-0 border-ink/30" : ""} ${
                    i === 2 ? "md:border-t lg:border-t-0" : ""
                  } ${i === 3 ? "md:border-t lg:border-t-0" : ""}`}
                >
                  {/* big han */}
                  <div
                    className="pointer-events-none absolute right-2 top-1 select-none font-han text-7xl text-cinnabar/[0.08] leading-none"
                    aria-hidden
                  >
                    {p.han}
                  </div>

                  <div className="relative">
                    <div className="flex items-baseline justify-between mb-3">
                      <span className="label-doc text-cinnabar">STEP {p.step}</span>
                      <Icon className="h-4 w-4 text-ink/40" />
                    </div>
                    <div className="flex items-baseline gap-2 mb-1">
                      <span className="font-han text-2xl font-bold text-cinnabar">
                        {p.han}
                      </span>
                      <span className="text-base font-bold text-ink">
                        {p.subtitle}
                      </span>
                    </div>
                    <div className="label-doc text-ink/60 mb-2">{p.title}</div>
                    <p className="text-xs leading-relaxed text-ink/70">{p.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Modules */}
      <section id="modules" className="border-b-2 border-ink">
        <div className="mx-auto max-w-[1600px] px-6 py-16">
          <div className="mb-10 flex items-end justify-between gap-6 flex-wrap">
            <div className="max-w-2xl">
              <div className="flex items-center gap-3 mb-2">
                <span className="seal h-6 w-6 text-[10px]">模</span>
                <span className="label-doc">CHAPTER · 02 · MODULAR BLUEPRINT</span>
              </div>
              <h2 className="font-han text-3xl font-bold tracking-tight text-ink">
                핵심 세부 연구모듈 4종
              </h2>
              <p className="mt-2 text-sm text-ink/70 leading-relaxed">
                KH-MFM 멀티모달 파운데이션 모델 위에서 동작하는 도메인 특화 모듈
              </p>
            </div>
            <span className="label-doc">DOC · 03 / 04</span>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {MODULES.map((m) => (
              <article
                key={m.badge}
                className="relative overflow-hidden border-2 border-ink bg-surface p-6 transition-shadow hover:shadow-[4px_4px_0_0_rgba(26,22,18,0.9)]"
              >
                <div className="flex items-start gap-5">
                  <div className="flex flex-col items-center gap-1.5">
                    <div className="seal h-14 w-14 text-2xl">{m.han}</div>
                    <span className="font-mono text-[9px] uppercase tracking-widest text-ink/55">
                      {m.badge.replace("Module ", "MOD·")}
                    </span>
                  </div>
                  <div className="flex-1 border-l border-ink/25 pl-5">
                    <div className="font-han text-xl font-bold text-ink mb-1">
                      {m.title}
                    </div>
                    <p className="text-xs leading-relaxed text-ink/70">{m.desc}</p>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative border-b-2 border-ink bg-ink text-background overflow-hidden">
        {/* 推 watermark */}
        <div
          className="pointer-events-none absolute -left-12 top-1/2 -translate-y-1/2 select-none font-han text-[28rem] font-bold leading-none text-background/[0.04]"
          aria-hidden
        >
          推
        </div>
        <div className="relative mx-auto max-w-[1600px] px-6 py-20 text-center">
          <ScrollText className="mx-auto h-6 w-6 text-cinnabar mb-3" />
          <h2 className="font-han text-3xl font-bold tracking-tight text-background">
            CDSS 콘솔에서 직접 추론을 돌려보세요
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-sm text-background/75">
            한의 문진 텍스트, 생체신호, 유전체 변이를 입력하면 KH-MFM이 멀티모달
            추론을 수행하고 합성 유전체 시뮬레이션을 시각화합니다.
          </p>
          <div className="mt-6">
            <Link
              href="/platform"
              className="inline-flex items-center gap-2 border-2 border-background bg-cinnabar px-8 py-3 text-sm font-bold uppercase tracking-widest text-background hover:bg-background hover:text-ink transition-colors"
            >
              入 · Platform 열기
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      <footer className="bg-background py-6 text-center">
        <div className="mx-auto max-w-[1600px] px-6 flex items-center justify-between">
          <span className="label-doc">© 2026 가천대학교 MRC · AKMPP</span>
          <span className="font-han text-xs text-ink/60">韓醫 · 精密 · 醫療</span>
          <span className="label-doc">DOC 04 / 04</span>
        </div>
      </footer>
    </div>
  );
}
