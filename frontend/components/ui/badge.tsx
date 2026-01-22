"use client";

import React from "react";
import clsx from "clsx";

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: "default" | "outline" | "success" | "danger" | "muted" | "secondary";
};

export function Badge({
  className,
  variant = "default",
  ...props
}: BadgeProps) {
  const base =
    "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide";

  const styles: Record<string, string> = {
    default: "border-sky-500/60 bg-sky-500/10 text-sky-300",
    outline: "border-slate-600 text-slate-200",
    success: "border-emerald-500/60 bg-emerald-500/10 text-emerald-300",
    danger: "border-rose-500/60 bg-rose-500/10 text-rose-300",
    muted: "border-slate-700 bg-slate-800 text-slate-300",
    secondary: "border-indigo-500/60 bg-indigo-500/10 text-indigo-300",
  };

  return (
    <span className={clsx(base, styles[variant], className)} {...props} />
  );
}
