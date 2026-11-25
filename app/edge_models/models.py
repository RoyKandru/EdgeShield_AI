"""Compact PyTorch models for RK3588 NPU deployment.

- Network1DCNN / CharacterCNN: 1D-CNN architectures for RK3588 INT8/FP16 conversion.
- FullSplitBiGRUMaskedPool / NPUSplitBiGRUCore: Exact mathematical equivalent of
  Phap (2026) `pack_padded_sequence + BiGRU + Masked Global Mean Pooling + WBCE`
  compiled 100% onto RK3588 NPU (exGRU x2 + exMatMul x2 + Concat + Conv in FP16).
"""
from __future__ import annotations
import torch
from torch import nn


class Network1DCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(), nn.Linear(24 * 8, 24), nn.ReLU(), nn.Dropout(0.1), nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.features(x).squeeze(-1)


class CharacterCNN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(vocab_size, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
            nn.Flatten(), nn.Linear(48 * 8, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SMSBiGRU(nn.Module):
    """Legacy compact BiGRU stub retained for backwards compatibility."""
    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        e = self.embedding(x.long())
        _, h = self.gru(e)
        return self.head(torch.cat([h[-2], h[-1]], dim=1)).squeeze(-1)


class FullSplitBiGRUMaskedPool(nn.Module):
    """Phap (2026) BiGRU + Masked Global Mean Pooling with exact pack_padded_sequence equivalence.

    Forward GRU processes valid prefix [x_1..x_L, 0..0] and backward GRU processes reversed valid
    prefix [x_L..x_1, 0..0]. Because mean pooling is order-invariant, pooling both outputs with
    pool_w = [1/L, ..., 1/L, 0, ..., 0] is bit-for-bit identical (< 1e-7) to packed BiGRU + Eq. 13.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru_fwd = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=False)
        self.gru_bwd = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, ids_fwd: torch.Tensor, ids_bwd: torch.Tensor, pool_w: torch.Tensor) -> torch.Tensor:
        ef = self.embedding(ids_fwd)
        eb = self.embedding(ids_bwd)
        hf, _ = self.gru_fwd(ef)
        hb, _ = self.gru_bwd(eb)
        pf = torch.bmm(pool_w, hf).squeeze(1)
        pb = torch.bmm(pool_w, hb).squeeze(1)
        return self.fc(self.dropout(torch.cat([pf, pb], dim=1))).squeeze(-1)


class NPUSplitBiGRUCore(nn.Module):
    """RK3588 NPU FP16 compute core (receives CPU-looked-up embeddings + pool_w)."""
    def __init__(self, full: FullSplitBiGRUMaskedPool):
        super().__init__()
        self.gru_fwd = full.gru_fwd
        self.gru_bwd = full.gru_bwd
        self.fc = full.fc

    def forward(self, seq_fwd: torch.Tensor, seq_bwd: torch.Tensor, pool_w: torch.Tensor) -> torch.Tensor:
        hf, _ = self.gru_fwd(seq_fwd)
        hb, _ = self.gru_bwd(seq_bwd)
        pf = torch.bmm(pool_w, hf).squeeze(1)
        pb = torch.bmm(pool_w, hb).squeeze(1)
        return self.fc(torch.cat([pf, pb], dim=1))
