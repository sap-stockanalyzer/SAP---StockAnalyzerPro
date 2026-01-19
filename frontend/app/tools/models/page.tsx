"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ModelCard } from "@/components/ModelCard";
import { ModelDetailsPanel } from "@/components/ModelDetailsPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { Search, AlertCircle, Brain } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelEntry, ModelDetails } from "@/lib/types";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.models.list();
      setModels(data);
    } catch (err: any) {
      setError(err.message || "Failed to load models");
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = async (model: ModelEntry) => {
    try {
      const details = await api.models.getDetails(model.id);
      setSelectedModel(details);
    } catch (err: any) {
      setError(err.message || "Failed to load model details");
    }
  };

  // Filter models
  const filteredModels = models.filter((model) => {
    const matchesSearch =
      model.name.toLowerCase().includes(search.toLowerCase()) ||
      model.description?.toLowerCase().includes(search.toLowerCase());
    const matchesType = typeFilter === "all" || model.type === typeFilter;
    const matchesStatus = statusFilter === "all" || model.status === statusFilter;
    return matchesSearch && matchesType && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2">Model Registry</h1>
          <p className="text-white/60">
            Browse and manage ML models for swing, intraday, and ensemble predictions
          </p>
        </div>

        {/* Toolbar */}
        <Card className="border-white/10 bg-white/5">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row gap-4">
              {/* Search */}
              <div className="flex-1 relative">
                <Search
                  size={16}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40"
                />
                <Input
                  placeholder="Search models..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10 bg-slate-900 border-white/10"
                />
              </div>

              {/* Type Filter */}
              <div className="w-full md:w-40">
                <Select value={typeFilter} onValueChange={setTypeFilter}>
                  <SelectTrigger className="bg-slate-900 border-white/10">
                    <SelectValue placeholder="Type" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-white/10">
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="swing">Swing</SelectItem>
                    <SelectItem value="intraday">Intraday</SelectItem>
                    <SelectItem value="ensemble">Ensemble</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Status Filter */}
              <div className="w-full md:w-40">
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="bg-slate-900 border-white/10">
                    <SelectValue placeholder="Status" />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-white/10">
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="active">Active</SelectItem>
                    <SelectItem value="inactive">Inactive</SelectItem>
                    <SelectItem value="deprecated">Deprecated</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading State */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <Skeleton key={i} className="h-48 w-full" />
            ))}
          </div>
        )}

        {/* Error State */}
        {error && (
          <Card className="border-red-500/40 bg-red-500/10">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-red-200">
                <AlertCircle size={20} />
                <div>
                  <h3 className="font-semibold mb-1">Error Loading Models</h3>
                  <p className="text-sm">{error}</p>
                  <p className="text-xs mt-2 text-red-300/80">
                    Make sure the backend models endpoint is available at{" "}
                    <code className="bg-red-500/20 px-1 py-0.5 rounded">/api/models/list</code>
                    <br />
                    The backend should stream data from{" "}
                    <code className="bg-red-500/20 px-1 py-0.5 rounded">
                      ml_data/model_registry.jsonl
                    </code>
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadModels}
                    className="mt-3 border-red-500/40"
                  >
                    Retry
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Models Grid */}
        {!loading && !error && filteredModels.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredModels.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                isSelected={selectedModel?.id === model.id}
                onSelect={handleSelect}
              />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && filteredModels.length === 0 && (
          <Card className="border-white/10 bg-white/5">
            <CardContent className="py-12">
              <div className="text-center text-white/60">
                <Brain size={48} className="mx-auto mb-4 opacity-40" />
                <p className="text-lg font-medium mb-2">
                  {models.length === 0 ? "No Models Available" : "No Matching Models"}
                </p>
                <p className="text-sm">
                  {models.length === 0
                    ? "Models will appear here once registered in the backend"
                    : "Try adjusting your search or filter criteria"}
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Model Details Panel */}
        {selectedModel && (
          <ModelDetailsPanel
            model={selectedModel}
            onClose={() => setSelectedModel(null)}
          />
        )}
      </div>
    </div>
  );
}