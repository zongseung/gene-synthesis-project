"use client";

import { useEffect, useRef, useState } from "react";
import { MessageCircle, X, Send, Loader2, BookOpen } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: { book: string; chapter: string }[];
}

const SUGGESTIONS = [
  "두통과 어지러움이 같이 오는데 어떤 변증으로 봐야 할까요?",
  "간양상항(肝陽上亢)의 대표 처방을 알려주세요",
  "동의보감에서 중풍 전조증을 어떻게 설명하나요?",
];

const SCRIPTED_RESPONSES: Record<string, ChatMessage> = {
  default: {
    role: "assistant",
    content:
      "참고 문헌 기반 응답입니다. 두통(頭痛)과 현훈(眩暈)이 동반될 경우 전통적으로 간양상항(肝陽上亢) 또는 간풍내동(肝風內動)으로 변증합니다. 환자의 맥상이 현(弦)하고 혈압이 높다면 평간잠양(平肝潛陽)법이 우선 고려됩니다.\n\n※ 본 답변은 한의 고전 텍스트 기반 참고용이며 임상 진단을 대체하지 않습니다.",
    citations: [
      { book: "東醫寶鑑·雜病篇", chapter: "頭門" },
      { book: "醫學入門", chapter: "眩暈門" },
    ],
  },
  prescription: {
    role: "assistant",
    content:
      "간양상항의 대표 처방은 **천마구등음(天麻鉤藤飮)**입니다. 구성은 천마·구등·석결명·치자·황금·익모초·천우슬·두충·상기생·야교등·복신으로 평간식풍(平肝熄風)·청열활혈(淸熱活血) 효능이 있습니다.\n\n혈압 상승, 두통, 어지러움, 불면, 이명 등의 증상에 활용되며 KH-MFM 추론에서도 신뢰도 0.91로 1순위 추천됩니다.",
    citations: [
      { book: "中醫內科雜病證治新義", chapter: "胡光慈" },
      { book: "東醫寶鑑·湯液篇", chapter: "風門" },
    ],
  },
  warning: {
    role: "assistant",
    content:
      "동의보감 풍문(風門)에서는 \"風者百病之長(풍자백병지장)\"이라 하여 중풍 전조로 ① 무지·소지의 마비감, ② 갑작스러운 두통·현훈, ③ 언어 장애, ④ 한쪽 안검 처짐을 경계 신호로 제시합니다. 환자 사례처럼 두통과 어지러움, ALDH2 변이가 동반되면 추적관찰 주기를 단축할 것을 권고합니다.",
    citations: [
      { book: "東醫寶鑑·雜病篇", chapter: "風門 - 中風先兆" },
      { book: "醫學心悟", chapter: "卷三 中風" },
    ],
  },
};

function pickResponse(input: string): ChatMessage {
  const lower = input.toLowerCase();
  if (lower.includes("처방") || lower.includes("prescription") || lower.includes("간양상항")) {
    return SCRIPTED_RESPONSES.prescription;
  }
  if (lower.includes("중풍") || lower.includes("동의보감") || lower.includes("전조")) {
    return SCRIPTED_RESPONSES.warning;
  }
  return SCRIPTED_RESPONSES.default;
}

export function DiagnosisChatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "안녕하세요. HanMed-LLM (ver5 v3.1) 진단 어시스턴트입니다. 한의 고전 26종 텍스트로 학습되었으며 문헌 기반 참고 응답을 제공합니다.\n\n좌측에서 환자 데이터를 입력하시거나 아래 추천 질문을 눌러보세요.",
    },
  ]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, thinking]);

  const send = (text: string) => {
    if (!text.trim() || thinking) return;
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setThinking(true);
    const response = pickResponse(text);
    setTimeout(() => {
      setMessages((m) => [...m, response]);
      setThinking(false);
    }, 900 + Math.random() * 600);
  };

  return (
    <>
      {/* Floating seal button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center",
          "border-2 border-ink bg-cinnabar text-[#faf6ea] shadow-[3px_3px_0_0_rgba(26,22,18,0.85)]",
          "transition-transform hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0_0_rgba(26,22,18,0.85)] animate-glow"
        )}
        aria-label="진단 챗봇 열기"
      >
        {open ? (
          <X className="h-5 w-5" />
        ) : (
          <span className="font-han text-2xl font-bold leading-none">問</span>
        )}
      </button>

      {/* Chat panel */}
      {open && (
        <div
          className={cn(
            "fixed bottom-24 right-6 z-50 flex h-[580px] w-[420px] max-h-[calc(100vh-120px)] max-w-[calc(100vw-3rem)]",
            "flex-col border-2 border-ink bg-surface shadow-[4px_4px_0_0_rgba(26,22,18,0.85)] overflow-hidden",
            "animate-fade-up"
          )}
        >
          {/* Header */}
          <div className="border-b-2 border-ink bg-background-2 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <div className="seal h-9 w-9 text-base">問</div>
                <div className="leading-tight border-l border-ink/30 pl-2">
                  <div className="font-han text-sm font-bold text-ink">
                    HanMed-LLM 어시스턴트
                  </div>
                  <div className="label-doc">
                    Llama-3-Bllossom-8B · ver5 v3.1
                  </div>
                </div>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-ink/60 hover:text-ink"
                aria-label="닫기"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-2 flex items-center gap-1.5 flex-wrap">
              <Badge variant="muted">
                <span className="font-han mr-0.5">典</span>26 books
              </Badge>
              <Badge variant="muted">
                <span className="font-han mr-0.5">引</span>고전 인용
              </Badge>
              <Badge variant="warning">참고용</Badge>
            </div>
          </div>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-3 bg-background">
            {messages.map((m, i) => (
              <Message key={i} message={m} />
            ))}
            {thinking && (
              <div className="flex items-center gap-2 text-xs text-ink/65">
                <Loader2 className="h-3 w-3 animate-spin text-cinnabar" />
                <span>고전 텍스트 검색 중…</span>
              </div>
            )}
            {messages.length === 1 && !thinking && (
              <div className="space-y-1.5 pt-2">
                <div className="label-doc">추천 질문 · 推薦</div>
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="w-full border border-ink/30 bg-surface px-2.5 py-2 text-left text-xs text-ink hover:bg-ink hover:text-background transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t-2 border-ink p-3 space-y-2 bg-background-2">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send(input);
              }}
              className="flex items-center gap-2"
            >
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="증상 또는 한의 용어를 입력…"
                disabled={thinking}
              />
              <Button type="submit" size="icon" disabled={thinking || !input.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </form>
            <p className="text-[10px] text-ink/55 leading-tight">
              <span className="font-han text-cinnabar mr-0.5">注</span>
              본 챗봇은 한의 고전 텍스트 기반{" "}
              <span className="text-cinnabar font-bold">문헌 참고용</span>이며,
              임상 진단·치료를 대체하지 않습니다. (KIOM mediclassics)
            </p>
          </div>
        </div>
      )}
    </>
  );
}

function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] px-3 py-2 text-xs leading-relaxed whitespace-pre-line border",
          isUser
            ? "bg-ink text-background border-ink"
            : "bg-surface border-ink/30 text-ink"
        )}
      >
        {message.content}
        {message.citations && (
          <div className="mt-2 space-y-1 border-t border-ink/15 pt-1.5">
            {message.citations.map((c, i) => (
              <div key={i} className="flex items-center gap-1.5 text-[10px] text-ink/65">
                <BookOpen className="h-2.5 w-2.5 text-cinnabar" />
                <span className="font-han">《{c.book}》</span>
                <span>· {c.chapter}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
