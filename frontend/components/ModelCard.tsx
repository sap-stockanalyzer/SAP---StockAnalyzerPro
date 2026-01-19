"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Calendar, TrendingUp } from "lucide-react";
import type { ModelEntry } from "@/lib/types";

interface ModelCardProps {
  model: ModelEntry;
  isSelected?: boolean;
  onSelect: (model: ModelEntry) => void;
  className?: string;
}

export function ModelCard({ model, isSelected, onSelect, className }: ModelCardProps) {
  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      swing: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      intraday: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      ensemble: "bg-green-500/20 text-green-400 border-green-500/30",
    };
    return colors[type] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      active: "bg-green-500/20 text-green-400 border-green-500/30",
      inactive: "bg-slate-500/20 text-slate-400 border-slate-500/30",
      deprecated: "bg-red-500/20 text-red-400 border-red-500/30",
    };
    return colors[status] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return dateString;
    }
  };

  return (
    <Card
      className={`border-white/10 bg-white/5 cursor-pointer transition-all hover:border-blue-500/40 ${
        isSelected ? "border-blue-500/60 bg-blue-500/10" : ""
      } ${className || ""}`}
      onClick={() => onSelect(model)}
    >
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <Brain className="text-blue-400" size={20} />
            <CardTitle className="text-base">{model.name}</CardTitle>
          </div>
          <Badge className={getStatusColor(model.status)}>{model.status}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <Badge className={getTypeColor(model.type)}>{model.type}</Badge>
          <span className="text-xs text-white/60">v{model.version}</span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-1 text-white/60">
            <TrendingUp size={14} />
            <span>Accuracy</span>
          </div>
          <span className={`font-semibold ${
            model.accuracy >= 0.8 ? "text-green-400" :
            model.accuracy >= 0.6 ? "text-yellow-400" :
            "text-red-400"
          }`}>
            {(model.accuracy * 100).toFixed(2)}%
          </span>
        </div>

        <div className="flex items-center gap-1 text-xs text-white/60">
          <Calendar size={12} />
          <span>Trained: {formatDate(model.trained_at)}</span>
        </div>
      </CardContent>
    </Card>
  );
}
