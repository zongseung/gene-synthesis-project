import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      ref={ref}
      className={cn(
        "flex h-9 w-full border border-ink/40 bg-surface px-3 py-1 text-sm text-ink placeholder:text-ink/35 focus-visible:outline-none focus-visible:border-cinnabar focus-visible:bg-[#faf6ea] transition-colors",
        className
      )}
      {...props}
    />
  )
);
Input.displayName = "Input";
