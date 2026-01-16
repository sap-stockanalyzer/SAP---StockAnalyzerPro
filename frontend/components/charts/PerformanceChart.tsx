"use client"

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"

interface PerformanceChartProps {
  data: Array<{ t: string; value: number }>;
  title: string;
  description?: string;
  valueLabel?: string;
  showFooter?: boolean;
  className?: string;
}

export function PerformanceChart({
  data,
  title,
  description,
  valueLabel = "Value",
  showFooter = false,
  className,
}: PerformanceChartProps) {
  // Calculate trend
  const firstValue = data[0]?.value || 0;
  const lastValue = data[data.length - 1]?.value || 0;
  const percentChange = firstValue !== 0 ? ((lastValue - firstValue) / firstValue) * 100 : 0;
  const isPositive = percentChange >= 0;

  const chartConfig = {
    value: {
      label: valueLabel,
      color: isPositive ? "hsl(142, 76%, 36%)" : "hsl(0, 84%, 60%)",
    },
  } satisfies ChartConfig;

  return (
    <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        {description && (
          <CardDescription className="text-white/60">{description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[200px] w-full">
          <LineChart
            data={data}
            margin={{
              left: 12,
              right: 12,
              top: 12,
              bottom: 12,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted))" vertical={false} />
            <XAxis
              dataKey="t"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
              }}
              stroke="hsl(var(--muted-foreground))"
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
              stroke="hsl(var(--muted-foreground))"
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                />
              }
            />
            <Line
              dataKey="value"
              type="monotone"
              stroke={`var(--color-value)`}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
        {showFooter && (
          <div className="mt-4 flex items-center gap-2 text-sm">
            <span className={`font-medium ${isPositive ? "text-green-400" : "text-red-400"}`}>
              {isPositive ? "+" : ""}{percentChange.toFixed(2)}%
            </span>
            <span className="text-white/60">from first data point</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
