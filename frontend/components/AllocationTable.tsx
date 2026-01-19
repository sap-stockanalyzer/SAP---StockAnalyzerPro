"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface Allocation {
  symbol: string;
  allocation_pct: number;
  shares?: number;
  dollar_amount?: number;
}

interface AllocationTableProps {
  allocations: Allocation[];
  className?: string;
}

export function AllocationTable({ allocations, className }: AllocationTableProps) {
  if (!allocations || allocations.length === 0) {
    return (
      <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
        <CardHeader>
          <CardTitle className="text-lg">Recommended Allocations</CardTitle>
          <CardDescription className="text-white/60">
            Optimal portfolio positions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-white/60">
            No allocations available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`border-white/10 bg-white/5 ${className || ""}`}>
      <CardHeader>
        <CardTitle className="text-lg">Recommended Allocations</CardTitle>
        <CardDescription className="text-white/60">
          Optimal portfolio positions ({allocations.length} assets)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-white/10">
          <Table>
            <TableHeader>
              <TableRow className="border-white/10 hover:bg-white/5">
                <TableHead className="text-white/80">Symbol</TableHead>
                <TableHead className="text-white/80 text-right">Allocation</TableHead>
                {allocations.some((a) => a.shares !== undefined) && (
                  <TableHead className="text-white/80 text-right">Shares</TableHead>
                )}
                {allocations.some((a) => a.dollar_amount !== undefined) && (
                  <TableHead className="text-white/80 text-right">Amount</TableHead>
                )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {allocations.map((allocation) => (
                <TableRow
                  key={allocation.symbol}
                  className="border-white/10 hover:bg-white/5"
                >
                  <TableCell className="font-medium text-white">
                    {allocation.symbol}
                  </TableCell>
                  <TableCell className="text-right text-white">
                    {(allocation.allocation_pct * 100).toFixed(2)}%
                  </TableCell>
                  {allocations.some((a) => a.shares !== undefined) && (
                    <TableCell className="text-right text-white/80">
                      {allocation.shares?.toLocaleString() || "-"}
                    </TableCell>
                  )}
                  {allocations.some((a) => a.dollar_amount !== undefined) && (
                    <TableCell className="text-right text-white/80">
                      {allocation.dollar_amount
                        ? `$${allocation.dollar_amount.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}`
                        : "-"}
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
