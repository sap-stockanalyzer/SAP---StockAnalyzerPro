"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { ReportCard } from "@/components/ReportCard";
import { ReportViewer } from "@/components/ReportViewer";
import { Skeleton } from "@/components/ui/skeleton";
import { Search, AlertCircle, FileText } from "lucide-react";
import { api } from "@/lib/api";
import type { Report, ReportDetails } from "@/lib/types";

const filterOptions = [
  { label: "All Types", value: "all" },
  { label: "Backtest", value: "backtest" },
  { label: "Model", value: "model" },
  { label: "Performance", value: "performance" },
  { label: "Analysis", value: "analysis" },
];

export default function ReportsPage() {
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedReport, setSelectedReport] = useState<ReportDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    loadReports();
  }, []);

  const loadReports = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.reports.list();
      setReports(data);
    } catch (err: any) {
      setError(err.message || "Failed to load reports");
    } finally {
      setLoading(false);
    }
  };

  const handleView = async (report: Report) => {
    try {
      const details = await api.reports.getDetails(report.id);
      setSelectedReport(details);
    } catch (err: any) {
      setError(err.message || "Failed to load report details");
    }
  };

  const handleDownload = async (reportId: string, format: "pdf" | "csv" | "json") => {
    try {
      const blob = await api.reports.download(reportId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report-${reportId}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      setError(err.message || "Failed to download report");
    }
  };

  // Filter reports
  const filteredReports = reports.filter((report) => {
    const matchesSearch = report.name.toLowerCase().includes(search.toLowerCase()) ||
      report.description?.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === "all" || report.type === filter;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold mb-2">Reports</h1>
          <p className="text-white/60">
            View, download, and manage model and backtest reports
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
                  placeholder="Search reports..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-10 bg-slate-900 border-white/10"
                />
              </div>

              {/* Filter */}
              <div className="w-full md:w-48">
                <Select value={filter} onChange={setFilter} options={filterOptions} />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading State */}
        {loading && (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-32 w-full" />
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
                  <h3 className="font-semibold mb-1">Error Loading Reports</h3>
                  <p className="text-sm">{error}</p>
                  <p className="text-xs mt-2 text-red-300/80">
                    Make sure the backend reports endpoint is available at{" "}
                    <code className="bg-red-500/20 px-1 py-0.5 rounded">/api/reports/list</code>
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadReports}
                    className="mt-3 border-red-500/40"
                  >
                    Retry
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Reports List */}
        {!loading && !error && filteredReports.length > 0 && (
          <div className="space-y-4">
            {filteredReports.map((report) => (
              <ReportCard
                key={report.id}
                report={report}
                onView={handleView}
                onDownload={handleDownload}
              />
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && filteredReports.length === 0 && (
          <Card className="border-white/10 bg-white/5">
            <CardContent className="py-12">
              <div className="text-center text-white/60">
                <FileText size={48} className="mx-auto mb-4 opacity-40" />
                <p className="text-lg font-medium mb-2">
                  {reports.length === 0 ? "No Reports Available" : "No Matching Reports"}
                </p>
                <p className="text-sm">
                  {reports.length === 0
                    ? "Reports will appear here once generated by the backend"
                    : "Try adjusting your search or filter criteria"}
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Report Viewer Modal */}
        {selectedReport && (
          <ReportViewer
            report={selectedReport}
            onClose={() => setSelectedReport(null)}
            onDownload={handleDownload}
          />
        )}
      </div>
    </div>
  );
}