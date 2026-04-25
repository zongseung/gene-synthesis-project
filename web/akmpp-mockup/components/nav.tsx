import Link from "next/link";

const NAV_LINKS = [
  { href: "/platform", label: "Platform", han: "推" },
  { href: "/#modules", label: "Modules", han: "模" },
  { href: "/#architecture", label: "Architecture", han: "構" },
  { href: "/#roadmap", label: "Roadmap", han: "路" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-40 w-full border-b-2 border-ink bg-background/85 backdrop-blur-md">
      {/* top hairline */}
      <div className="h-1 w-full bg-ink" />
      <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-3 group">
          {/* 朱印 seal */}
          <div className="seal h-11 w-11 text-lg leading-none animate-glow">
            韓
          </div>
          <div className="leading-tight border-l border-ink/30 pl-3">
            <div className="font-han text-base font-bold tracking-tight text-ink">
              韓醫藥 精密醫療
            </div>
            <div className="label-doc -mt-0.5">
              AKMPP · 가천대 MRC · DOC 2026
            </div>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="group flex items-center gap-1.5 rounded-sm border border-transparent px-3 py-1.5 text-xs font-medium uppercase tracking-wider text-ink/70 hover:text-ink hover:border-ink/40 hover:bg-surface transition-colors"
            >
              <span className="font-han text-[11px] text-cinnabar/70 group-hover:text-cinnabar">
                {link.han}
              </span>
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <div className="hidden md:flex flex-col items-end leading-tight">
            <span className="label-doc">SYSTEM ONLINE</span>
            <span className="font-mono text-[10px] text-ink/60">
              KH-MFM v0.3 · HanMed-LLM ver5
            </span>
          </div>
          <div className="hidden sm:flex h-2 w-2 rounded-full bg-cinnabar animate-pulse" />
        </div>
      </div>
    </header>
  );
}
