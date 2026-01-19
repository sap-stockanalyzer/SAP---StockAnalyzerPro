"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { X, Brain, TrendingUp, Calendar, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ModelDetails } from "@/lib/types";

interface ModelDetailsPanelProps {
  model: ModelDetails;
  onClose: () => void;
  className?: string;
}

export function ModelDetailsPanel({ model, onClose, className }: ModelDetailsPanelProps) {
  const [activeTab, setActiveTab] = useState("overview");
  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      swing: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      intraday: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      ensemble: "bg-green-500/20 text-green-400 border-green-500/30",
    };
    return colors[type] || "bg-slate-500/20 text-slate-400 border-slate-500/30";
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className={`fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 ${className || ""}`}>
      <Card className="border-white/10 bg-slate-900 w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        <CardHeader className="border-b border-white/10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Brain className="text-blue-400" size={24} />
              <div>
                <CardTitle className="text-xl">{model.name}</CardTitle>
                {model.description && (
                  <p className="text-sm text-white/60 mt-1">{model.description}</p>
                )}
              </div>
              <Badge className={getTypeColor(model.type)}>{model.type}</Badge>
            </div>
            <Button variant="ghost" size="sm" onClick={onClose}>
              <X size={16} />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto p-6">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-white/5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="metrics">Metrics</TabsTrigger>
              <TabsTrigger value="training">Training</TabsTrigger>
              <TabsTrigger value="hyperparameters">Config</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4 mt-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <MetricBox label="Version" value={`v${model.version}`} />
                <MetricBox
                  label="Accuracy"
                  value={`${(model.accuracy * 100).toFixed(2)}%`}
                />
                <MetricBox label="Status" value={model.status} />
                <MetricBox label="Training Data" value={model.training_data_size.toLocaleString()} />
                <MetricBox label="Features" value={model.features_count.toString()} />
                <MetricBox label="Algorithm" value={model.training_config.algorithm} />
              </div>

              <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Calendar size={16} className="text-white/60" />
                  <h3 className="text-sm font-semibold">Training Information</h3>
                </div>
                <div className="space-y-1 text-sm text-white/80">
                  <p>Trained: {formatDate(model.trained_at)}</p>
                  <p>Data Source: {model.training_config.training_data_source}</p>
                  <p>
                    Validation/Test Split: {(model.training_config.validation_split * 100).toFixed(0)}%
                    / {(model.training_config.test_split * 100).toFixed(0)}%
                  </p>
                </div>
              </div>

              {model.feature_importance && model.feature_importance.length > 0 && (
                <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                  <h3 className="text-sm font-semibold mb-3">Top Features</h3>
                  <div className="space-y-2">
                    {model.feature_importance.slice(0, 10).map((feature) => (
                      <div key={feature.feature_name} className="flex items-center gap-2">
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-sm text-white">{feature.feature_name}</span>
                            <span className="text-xs text-white/60">
                              {(feature.importance_score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-blue-500"
                              style={{ width: `${feature.importance_score * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="metrics" className="mt-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                <MetricBox
                  label="Accuracy"
                  value={`${(model.performance_metrics.accuracy * 100).toFixed(2)}%`}
                />
                <MetricBox
                  label="Precision"
                  value={`${(model.performance_metrics.precision * 100).toFixed(2)}%`}
                />
                <MetricBox
                  label="Recall"
                  value={`${(model.performance_metrics.recall * 100).toFixed(2)}%`}
                />
                <MetricBox
                  label="F1 Score"
                  value={model.performance_metrics.f1_score.toFixed(3)}
                />
                <MetricBox
                  label="AUC-ROC"
                  value={model.performance_metrics.auc_roc.toFixed(3)}
                />
              </div>

              {model.performance_metrics.confusion_matrix && (
                <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                  <h3 className="text-sm font-semibold mb-3">Confusion Matrix</h3>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-white/10">
                          <TableHead className="text-white/80"></TableHead>
                          <TableHead className="text-white/80 text-center">Predicted Negative</TableHead>
                          <TableHead className="text-white/80 text-center">Predicted Positive</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {model.performance_metrics.confusion_matrix.map((row, i) => (
                          <TableRow key={i} className="border-white/10">
                            <TableCell className="font-medium text-white">
                              {i === 0 ? "Actual Negative" : "Actual Positive"}
                            </TableCell>
                            {row.map((value, j) => (
                              <TableCell key={j} className="text-center text-white">
                                {value}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="training" className="mt-4">
              {model.training_history && model.training_history.length > 0 ? (
                <div className="rounded-lg border border-white/10 bg-white/5 overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10">
                        <TableHead className="text-white/80">Epoch</TableHead>
                        <TableHead className="text-white/80 text-right">Training Loss</TableHead>
                        <TableHead className="text-white/80 text-right">Validation Loss</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {model.training_history.slice(-20).map((step) => (
                        <TableRow key={step.epoch} className="border-white/10">
                          <TableCell className="text-white">{step.epoch}</TableCell>
                          <TableCell className="text-right text-white">
                            {step.training_loss.toFixed(4)}
                          </TableCell>
                          <TableCell className="text-right text-white">
                            {step.validation_loss.toFixed(4)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-center py-8 text-white/60">
                  No training history available
                </div>
              )}
            </TabsContent>

            <TabsContent value="hyperparameters" className="mt-4">
              <div className="space-y-4">
                <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Settings size={16} className="text-white/60" />
                    <h3 className="text-sm font-semibold">Hyperparameters</h3>
                  </div>
                  <pre className="text-sm text-white/80 overflow-x-auto">
                    {JSON.stringify(model.hyperparameters, null, 2)}
                  </pre>
                </div>

                <div className="rounded-lg border border-white/10 bg-white/5 p-4">
                  <h3 className="text-sm font-semibold mb-3">Training Configuration</h3>
                  <pre className="text-sm text-white/80 overflow-x-auto">
                    {JSON.stringify(model.training_config, null, 2)}
                  </pre>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 p-3">
      <div className="text-xs text-white/60 mb-1">{label}</div>
      <div className="text-base font-semibold text-white">{value}</div>
    </div>
  );
}
