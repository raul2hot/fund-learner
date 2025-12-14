"""Co-Attention Fusion Layer for SPH-Net"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoAttentionFusion(nn.Module):
    """
    Bi-directional co-attention between temporal and feature tokens.

    Temporal tokens attend to feature tokens (and vice versa).
    Returns fused representation combining both streams.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Temporal -> Feature attention
        self.q_temporal = nn.Linear(d_model, d_model)
        self.k_feature = nn.Linear(d_model, d_model)
        self.v_feature = nn.Linear(d_model, d_model)

        # Feature -> Temporal attention
        self.q_feature = nn.Linear(d_model, d_model)
        self.k_temporal = nn.Linear(d_model, d_model)
        self.v_temporal = nn.Linear(d_model, d_model)

        # Output projections
        self.out_temporal = nn.Linear(d_model, d_model)
        self.out_feature = nn.Linear(d_model, d_model)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)

    def _attention(self, Q, K, V):
        """Scaled dot-product attention"""
        # Q, K, V: [batch, n_heads, seq_len, head_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _reshape_for_attention(self, x, proj):
        """Project and reshape for multi-head attention"""
        batch, seq_len, _ = x.shape
        x = proj(x)
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]

    def forward(self, temporal_tokens, feature_tokens):
        """
        Args:
            temporal_tokens: [batch, T, d_model]
            feature_tokens: [batch, T, d_model]
        Returns:
            fused: [batch, T, d_model]
        """
        batch = temporal_tokens.shape[0]

        # Temporal attends to features
        Q_t = self._reshape_for_attention(temporal_tokens, self.q_temporal)
        K_f = self._reshape_for_attention(feature_tokens, self.k_feature)
        V_f = self._reshape_for_attention(feature_tokens, self.v_feature)

        attn_t = self._attention(Q_t, K_f, V_f)
        attn_t = attn_t.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        attn_t = self.out_temporal(attn_t)
        fused_temporal = self.norm_t(temporal_tokens + attn_t)

        # Features attend to temporal
        Q_f = self._reshape_for_attention(feature_tokens, self.q_feature)
        K_t = self._reshape_for_attention(temporal_tokens, self.k_temporal)
        V_t = self._reshape_for_attention(temporal_tokens, self.v_temporal)

        attn_f = self._attention(Q_f, K_t, V_t)
        attn_f = attn_f.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        attn_f = self.out_feature(attn_f)
        fused_features = self.norm_f(feature_tokens + attn_f)

        # Combine both streams
        combined = torch.cat([fused_temporal, fused_features], dim=-1)
        fused = self.fusion(combined)

        return fused
