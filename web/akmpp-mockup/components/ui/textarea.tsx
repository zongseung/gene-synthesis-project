import * as React from "react";
import { cn } from "@/lib/utils";

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "flex min-h-[80px] w-full border border-ink/40 bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink/35 focus-visible:outline-none focus-visible:border-cinnabar focus-visible:bg-[#faf6ea] transition-colors resize-none",
      className
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";
