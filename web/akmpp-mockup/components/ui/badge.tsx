import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-sm px-2 py-0.5 text-[10px] font-mono font-bold uppercase tracking-widest border",
  {
    variants: {
      variant: {
        default: "bg-cinnabar text-[#faf6ea] border-cinnabar-strong",
        accent: "bg-ink text-background border-ink",
        muted: "bg-transparent text-ink/70 border-ink/40",
        outline: "bg-transparent text-ink border-ink",
        success: "bg-moss-soft text-[#3f4b29] border-[#5b6b3d]/50",
        warning: "bg-[#a07b22]/15 text-[#7a5e1a] border-[#a07b22]/50",
        danger: "bg-cinnabar-soft text-cinnabar-strong border-cinnabar/50",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
