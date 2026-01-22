import * as React from "react";
import clsx from "clsx";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "secondary" | "destructive" | "ghost";
  size?: "default" | "sm" | "lg";
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = "", variant = "default", size = "default", ...props }, ref) => {
    const baseStyles = "inline-flex items-center justify-center rounded-md font-medium transition disabled:opacity-50 disabled:cursor-not-allowed";
    
    const variantStyles: Record<string, string> = {
      default: "bg-blue-600 text-white hover:bg-blue-500",
      outline: "border border-gray-600 bg-transparent text-gray-200 hover:bg-gray-800",
      secondary: "bg-gray-700 text-white hover:bg-gray-600",
      destructive: "bg-red-600 text-white hover:bg-red-500",
      ghost: "bg-transparent text-gray-200 hover:bg-gray-800",
    };
    
    const sizeStyles: Record<string, string> = {
      default: "px-3 py-1.5 text-sm",
      sm: "px-2 py-1 text-xs",
      lg: "px-4 py-2 text-base",
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
