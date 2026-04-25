import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

export const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cinnabar focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 border-2",
  {
    variants: {
      variant: {
        default:
          "bg-cinnabar text-[#faf6ea] border-cinnabar-strong hover:bg-cinnabar-strong shadow-[2px_2px_0_0_rgba(26,22,18,0.85)] hover:shadow-[1px_1px_0_0_rgba(26,22,18,0.85)] hover:translate-x-[1px] hover:translate-y-[1px]",
        accent:
          "bg-ink text-background border-ink hover:bg-ink-soft",
        outline:
          "border-ink bg-transparent text-ink hover:bg-ink hover:text-background",
        ghost:
          "border-transparent text-ink hover:bg-surface-2",
        secondary:
          "bg-surface text-ink border-ink/30 hover:border-ink",
      },
      size: {
        default: "h-10 px-4 py-2 rounded-sm",
        sm: "h-8 px-3 text-xs rounded-sm",
        lg: "h-12 px-6 text-sm rounded-sm",
        icon: "h-9 w-9 rounded-sm",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
);
Button.displayName = "Button";
