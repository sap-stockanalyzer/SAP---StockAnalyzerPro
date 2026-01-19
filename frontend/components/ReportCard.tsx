"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Eye, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface Report {
  id: string;
  type: "backtest" | "model" | "performance" | "analysis";
  name: string;
  description?: string;
  created_at: string;
  size_bytes: number;
  formats_available: ("pdf" | "csv" | "json")[];
  status: "ready" | "generating" | "failed";
}

interface ReportCardProps {
  report: Report;
  onView: (report: Report) => void;
  onDownload: (reportId: string, format: "pdf" | "csv" | "json") => void;
  className?: string;
}

export function ReportCard({ report, onView, onDownload, className }: ReportCardProps) {
  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      backtest: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      model: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      performance: "bg-green-500/20 text-green-400 border-green-500/30",
      analysis: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    };
    return colors[type] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      ready: "bg-green-500/20 text-green-400 border-green-500/30",
      generating: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
      failed: "bg-red-500/20 text-red-400 border-red-500/30",
    };
    return colors[status] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateString;
    }
  };

  return (
    <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <FileText className="text-white/60" size={20} />
              <CardTitle className="text-base truncate">{report.name}</CardTitle>
            </div>
            {report.description && (
              <p className="text-sm text-white/60 line-clamp-2">{report.description}</p>
            )}
          </div>
          <div className="flex flex-col gap-1 items-end">
            <Badge className={getTypeColor(report.type)}>{report.type}</Badge>
            <Badge className={getStatusColor(report.status)}>{report.status}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 text-xs text-white/60 space-y-1">
            <div>Created: {formatDate(report.created_at)}</div>
            <div>Size: {formatFileSize(report.size_bytes)}</div>
          </div>
          <div className="flex items-center gap-2">
            {report.status === "ready" && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onView(report)}
                  className="border-white/10"
                >
                  <Eye size={14} className="mr-1" />
                  View
                </Button>
                {report.formats_available.map((format) => (
                  <Button
                    key={format}
                    variant="ghost"
                    size="sm"
                    onClick={() => onDownload(report.id, format)}
                    className="border-white/10"
                  >
                    <Download size={14} className="mr-1" />
                    {format.toUpperCase()}
                  </Button>
                ))}
              </>
            )}
            {report.status === "generating" && (
              <div className="text-sm text-yellow-400">Generating...</div>
            )}
            {report.status === "failed" && (
              <div className="text-sm text-red-400">Failed</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
