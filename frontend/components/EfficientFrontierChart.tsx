"use client";

import { CartesianGrid, Line, LineChart, XAxis, YAxis, Scatter, ScatterChart, ZAxis } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

interface EfficientFrontierChartProps {
  data: Array<{ risk: number; return: number }>;
  currentPortfolio?: { risk: number; return: number };
  className?: string;
}

export function EfficientFrontierChart({
  data,
  currentPortfolio,
  className,
}: EfficientFrontierChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
        <CardHeader>
          <CardTitle className="text-lg">Efficient Frontier</CardTitle>
          <CardDescription className="text-white/60">
            Risk-return trade-off visualization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full flex items-center justify-center text-white/60">
            No efficient frontier data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const chartConfig = {
    return: {
      label: "Expected Return",
      color: "hsl(217, 91%, 60%)",
    },
    current: {
      label: "Current Portfolio",
      color: "hsl(142, 76%, 36%)",
    },
  } satisfies ChartConfig;

  // Combine data for scatter plot
  const scatterData = [
    ...data.map((point) => ({ ...point, type: "frontier" })),
    ...(currentPortfolio ? [{ ...currentPortfolio, type: "current" }] : []),
  ];

  return (
    <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
      <CardHeader>
        <CardTitle className="text-lg">Efficient Frontier</CardTitle>
        <CardDescription className="text-white/60">
          Risk-return trade-off for optimal portfolios
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
          <LineChart
            data={data}
            margin={{
              left: 12,
              right: 12,
              top: 12,
              bottom: 12,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="risk"
              stroke="rgba(255,255,255,0.4)"
              label={{ value: "Risk (Volatility)", position: "insideBottom", offset: -5 }}
              tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
            />
            <YAxis
              dataKey="return"
              stroke="rgba(255,255,255,0.4)"
              label={{ value: "Expected Return", angle: -90, position: "insideLeft" }}
              tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => `Risk: ${(Number(value) * 100).toFixed(2)}%`}
                  formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`}
                />
              }
            />
            <Line
              type="monotone"
              dataKey="return"
              stroke="hsl(217, 91%, 60%)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
        {currentPortfolio && (
          <div className="mt-4 p-3 rounded-lg bg-green-500/10 border border-green-500/30">
            <p className="text-sm text-green-400">
              Current Portfolio: Risk {(currentPortfolio.risk * 100).toFixed(2)}%, Return{" "}
              {(currentPortfolio.return * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
