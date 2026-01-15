import * as React from "react";
import clsx from "clsx";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "link" | "destructive" | "secondary";
  size?: "default" | "sm" | "lg" | "icon";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "default", size = "default", ...props }, ref) => {
    const baseStyles = "inline-flex items-center justify-center rounded-md font-medium transition-colors disabled:opacity-50 disabled:pointer-events-none";
    
    const variantStyles = {
      default: "bg-blue-600 text-white hover:bg-blue-500",
      outline: "border border-slate-600 text-slate-200 hover:bg-slate-800",
      ghost: "hover:bg-slate-800 text-slate-200",
      link: "text-blue-400 underline-offset-4 hover:underline",
      destructive: "bg-red-600 text-white hover:bg-red-500",
      secondary: "bg-slate-700 text-slate-200 hover:bg-slate-600",
    };
    
    const sizeStyles = {
      default: "px-3 py-1.5 text-sm",
      sm: "px-2 py-1 text-xs",
      lg: "px-4 py-2 text-base",
      icon: "h-9 w-9",
    };

    return (
      <button
        ref={ref}
        className={clsx(
          baseStyles,
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
