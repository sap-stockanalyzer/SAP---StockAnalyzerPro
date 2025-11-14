"""
gnn_model.py — v1.2 (Import-Safe Graph Neural Network Scaffold)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Provides a minimal Graph Neural Network class using PyTorch Geometric.
- If torch or torch_geometric are unavailable, defines no-op stubs so imports never fail.
- Safe to import even on lightweight environments or during CI/CD builds.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv  # type: ignore
    TORCH_OK = True
except Exception:
    TORCH_OK = False


if TORCH_OK:

    class SimpleGCN(nn.Module):
        """Minimal 2-layer Graph Convolutional Network."""

        def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 1):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, out_dim)

        def forward(self, x, edge_index):
            """Forward pass through two GCN layers."""
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x


else:

    class SimpleGCN:  # type: ignore
        """Fallback stub when torch/torch_geometric are unavailable."""

        def __init__(self, *args, **kwargs):
            self.available = False

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                "❌ SimpleGCN unavailable — install PyTorch + torch_geometric to use this model."
            )
