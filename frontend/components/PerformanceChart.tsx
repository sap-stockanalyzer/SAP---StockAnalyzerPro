"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
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

interface PerformanceChartProps {
  data: Array<{
    epoch: number;
    training_loss: number;
    validation_loss: number;
  }>;
  title?: string;
  className?: string;
}

export function PerformanceChart({
  data,
  title = "Model Performance",
  className,
}: PerformanceChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription className="text-white/60">
            Training and validation metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[250px] w-full flex items-center justify-center text-white/60">
            No performance data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const chartConfig = {
    training_loss: {
      label: "Training Loss",
      color: "hsl(217, 91%, 60%)",
    },
    validation_loss: {
      label: "Validation Loss",
      color: "hsl(142, 76%, 36%)",
    },
  } satisfies ChartConfig;

  return (
    <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription className="text-white/60">
          Training and validation loss over time
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[250px] w-full">
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
              dataKey="epoch"
              stroke="rgba(255,255,255,0.4)"
              label={{ value: "Epoch", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              stroke="rgba(255,255,255,0.4)"
              label={{ value: "Loss", angle: -90, position: "insideLeft" }}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Line
              type="monotone"
              dataKey="training_loss"
              stroke="hsl(217, 91%, 60%)"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="validation_loss"
              stroke="hsl(142, 76%, 36%)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
