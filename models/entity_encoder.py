# models/entity_encoder.py
"""Entity-based observation encoder with multi-head self-attention.

Implements the core architecture for E8.1: replace the fixed-size concatenation
observation with a variable-length sequence of entity tokens (one per visible unit),
processed by a transformer encoder (multi-head self-attention).

Entity Token Schema (ENTITY_TOKEN_DIM = 16)
-------------------------------------------
Each entity is described by a 16-dimensional token:

+----------+------+---------+-------------------------------+
| Field    | Dims | Range   | Description                   |
+==========+======+=========+===============================+
| unit_type| 3    | one-hot | infantry / cavalry / artillery|
+----------+------+---------+-------------------------------+
| position | 2    | [0, 1]  | x, y normalised by map dims   |
+----------+------+---------+-------------------------------+
| heading  | 2    | [-1, 1] | (cos θ, sin θ)                |
+----------+------+---------+-------------------------------+
| strength | 1    | [0, 1]  | battalion strength fraction   |
+----------+------+---------+-------------------------------+
| formation| 4    | one-hot | LINE/COLUMN/SQUARE/SKIRMISH   |
+----------+------+---------+-------------------------------+
| ammo     | 1    | [0, 1]  | ammunition fraction           |
+----------+------+---------+-------------------------------+
| morale   | 1    | [0, 1]  | morale level                  |
+----------+------+---------+-------------------------------+
| team     | 2    | one-hot | blue / red                    |
+----------+------+---------+-------------------------------+

Total: 3 + 2 + 2 + 1 + 4 + 1 + 1 + 2 = 16 dimensions

Variable-length sequences are handled via a boolean **padding mask**: positions
set to ``True`` are ignored by the attention mechanism, allowing batches with
different numbers of entities to be processed together.

Typical usage::

    from models.entity_encoder import EntityEncoder, EntityActorCriticPolicy

    # Encoder only
    encoder = EntityEncoder(token_dim=16, d_model=64, n_heads=4, n_layers=2)
    tokens = torch.zeros(batch, n_entities, 16)           # (B, N, 16)
    pad_mask = torch.zeros(batch, n_entities, dtype=bool) # False = keep
    out = encoder(tokens, pad_mask)                       # (B, d_model)

    # Full actor-critic policy
    policy = EntityActorCriticPolicy(
        token_dim=16,
        action_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    actions, log_probs = policy.act(tokens, pad_mask)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = [
    "ENTITY_TOKEN_DIM",
    "UNIT_TYPE_INFANTRY",
    "UNIT_TYPE_CAVALRY",
    "UNIT_TYPE_ARTILLERY",
    "TEAM_BLUE",
    "TEAM_RED",
    "SpatialPositionalEncoding",
    "EntityEncoder",
    "EntityActorCriticPolicy",
]

# ---------------------------------------------------------------------------
# Entity token schema constants
# ---------------------------------------------------------------------------

#: Total dimensionality of one entity token.
ENTITY_TOKEN_DIM: int = 16

# Unit-type one-hot indices (within the first 3 dims)
UNIT_TYPE_INFANTRY: int = 0
UNIT_TYPE_CAVALRY: int = 1
UNIT_TYPE_ARTILLERY: int = 2

# Team one-hot indices (within the last 2 dims)
TEAM_BLUE: int = 0
TEAM_RED: int = 1

# Slice boundaries for each field
_SLICE_UNIT_TYPE = slice(0, 3)
_SLICE_POSITION = slice(3, 5)
_SLICE_HEADING = slice(5, 7)
_SLICE_STRENGTH = slice(7, 8)
_SLICE_FORMATION = slice(8, 12)
_SLICE_AMMO = slice(12, 13)
_SLICE_MORALE = slice(13, 14)
#: Default scale factor for the transformer feedforward sublayer width
#: relative to d_model.  The value of 4 follows the original "Attention Is
#: All You Need" paper (Vaswani et al., 2017).
_DEFAULT_FFN_SCALE: int = 4


# ---------------------------------------------------------------------------
# 2-D Fourier positional encoding
# ---------------------------------------------------------------------------


class SpatialPositionalEncoding(nn.Module):
    """Additive 2-D Fourier positional encoding for entity (x, y) positions.

    Computes:

    .. code-block:: text

        PE(x, y) = concat([sin(2π k x), cos(2π k x),
                           sin(2π k y), cos(2π k y)]  for k in 1..n_freqs)

    and projects the resulting ``4 * n_freqs``-dimensional vector to
    ``d_model`` via a single linear layer.

    Parameters
    ----------
    d_model:
        Output dimension (must match the transformer d_model).
    n_freqs:
        Number of frequency bands per axis.  Defaults to ``8``.
    """

    def __init__(self, d_model: int, n_freqs: int = 8) -> None:
        super().__init__()
        self.n_freqs = n_freqs
        # Frequencies: [1, 2, …, n_freqs]
        freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)  # (n_freqs,)
        self.proj = nn.Linear(4 * n_freqs, d_model)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Compute positional embeddings for a batch of (x, y) pairs.

        Parameters
        ----------
        xy:
            Positions of shape ``(..., 2)`` in ``[0, 1]``.

        Returns
        -------
        pe : torch.Tensor — shape ``(..., d_model)``
        """
        x = xy[..., 0:1]  # (..., 1)
        y = xy[..., 1:2]  # (..., 1)
        freqs = self.freqs  # (n_freqs,)
        # (..., n_freqs)
        ax = 2 * math.pi * x * freqs
        ay = 2 * math.pi * y * freqs
        pe = torch.cat([torch.sin(ax), torch.cos(ax),
                        torch.sin(ay), torch.cos(ay)], dim=-1)  # (..., 4*n_freqs)
        return self.proj(pe)  # (..., d_model)


# ---------------------------------------------------------------------------
# Entity Encoder
# ---------------------------------------------------------------------------


class EntityEncoder(nn.Module):
    """Multi-head self-attention encoder over a variable-length entity sequence.

    Architecture::

        entity tokens (B, N, token_dim)
              │
        token_embed: Linear(token_dim → d_model)
              │ + SpatialPositionalEncoding(x, y) [optional]
              │
        TransformerEncoder (n_layers × TransformerEncoderLayer)
           └── MultiheadAttention (n_heads, d_model)
           └── FFN (dim_feedforward = 4 * d_model)
           └── LayerNorm + residual
              │
        mean-pool over non-padded entities  →  (B, d_model)
              │
        output projection: Linear(d_model → d_model)  [identity-initialised]

    Padding is handled via ``src_key_padding_mask``: a boolean tensor of shape
    ``(B, N)`` where ``True`` marks *padded* (ignored) positions.

    Parameters
    ----------
    token_dim:
        Dimensionality of each input entity token.  Defaults to
        ``ENTITY_TOKEN_DIM`` (16).
    d_model:
        Internal transformer dimension.
    n_heads:
        Number of attention heads.  Must evenly divide ``d_model``.
    n_layers:
        Number of transformer encoder layers.
    dim_feedforward:
        Feed-forward sublayer width.  Defaults to ``4 * d_model``.
    dropout:
        Dropout probability inside the transformer.
    use_spatial_pe:
        When ``True``, add a 2-D Fourier positional encoding derived from
        the position field ``(x, y)`` of each token.
    n_freq_bands:
        Number of Fourier frequency bands used by
        :class:`SpatialPositionalEncoding` when ``use_spatial_pe=True``.
    """

    def __init__(
        self,
        token_dim: int = ENTITY_TOKEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        use_spatial_pe: bool = True,
        n_freq_bands: int = 8,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_spatial_pe = use_spatial_pe

        if dim_feedforward is None:
            dim_feedforward = _DEFAULT_FFN_SCALE * d_model

        # Project raw token features → d_model
        self.token_embed = nn.Linear(token_dim, d_model)

        # Optional 2-D Fourier positional encoding
        if use_spatial_pe:
            self.spatial_pe = SpatialPositionalEncoding(d_model, n_freqs=n_freq_bands)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, N, d_model) convention
            norm_first=True,   # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,  # required when norm_first=True
        )

        # Output projection (initialised as identity-like)
        self.out_proj = nn.Linear(d_model, d_model)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the pooled output vector."""
        return self.d_model

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of entity sequences.

        Parameters
        ----------
        tokens:
            Entity token tensor of shape ``(B, N, token_dim)``.
        pad_mask:
            Boolean mask of shape ``(B, N)``.  ``True`` marks padded
            (ignored) positions.  If ``None``, all positions are attended.
        return_attention:
            When ``True``, also return the averaged attention weights from
            the *last* transformer layer as a second return value of shape
            ``(B, N, N)``.

        Returns
        -------
        encoding : torch.Tensor — shape ``(B, d_model)``
            Mean-pooled sequence encoding.
        attn_weights : torch.Tensor — shape ``(B, N, N)``
            Averaged attention weights from the last layer.
            Only returned when ``return_attention=True``.
        """
        B, N, _ = tokens.shape

        # Token embedding
        x = self.token_embed(tokens)  # (B, N, d_model)

        # Optional 2-D Fourier positional encoding from (x, y) position fields
        if self.use_spatial_pe:
            xy = tokens[..., _SLICE_POSITION]  # (B, N, 2)
            x = x + self.spatial_pe(xy)

        if return_attention:
            # Run layers manually to extract attention from the last layer
            for i, layer in enumerate(self.transformer.layers):
                if i < len(self.transformer.layers) - 1:
                    x = layer(x, src_key_padding_mask=pad_mask)
                else:
                    # Last layer: extract attention weights
                    x_norm = layer.norm1(x) if layer.norm_first else x
                    attn_out, attn_weights = layer.self_attn(
                        x_norm, x_norm, x_norm,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                    # Complete the residual path
                    if layer.norm_first:
                        x = x + layer.dropout1(attn_out)
                        x = x + layer.dropout2(
                            layer.linear2(
                                layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))
                            )
                        )
                    else:
                        x = layer.norm1(x + layer.dropout1(attn_out))
                        x = layer.norm2(
                            x + layer.dropout2(
                                layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                            )
                        )
        else:
            x = self.transformer(x, src_key_padding_mask=pad_mask)
            attn_weights = None

        # Mean-pool over non-padded positions
        if pad_mask is not None:
            # Invert mask: True = keep
            keep = ~pad_mask  # (B, N)
            # Avoid division by zero if all positions are padded
            n_valid = keep.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1); clamp avoids ÷0 if all padded
            x = (x * keep.unsqueeze(-1).float()).sum(dim=1) / n_valid  # (B, d_model)
        else:
            x = x.mean(dim=1)  # (B, d_model)

        encoding = self.out_proj(x)  # (B, d_model)

        if return_attention:
            return encoding, attn_weights  # type: ignore[return-value]
        return encoding

    @staticmethod
    def make_padding_mask(n_valid: torch.Tensor, max_n: int) -> torch.Tensor:
        """Create a padding mask from per-sample entity counts.

        Parameters
        ----------
        n_valid:
            Integer tensor of shape ``(B,)`` with the number of real
            (non-padded) entities in each sample.
        max_n:
            Total number of positions in the padded sequence.

        Returns
        -------
        pad_mask : torch.BoolTensor — shape ``(B, max_n)``
            ``True`` where the position is padded (ignored).
        """
        B = n_valid.size(0)
        idx = torch.arange(max_n, device=n_valid.device).unsqueeze(0)  # (1, max_n)
        return idx >= n_valid.unsqueeze(1)  # (B, max_n)


# ---------------------------------------------------------------------------
# Entity-based Actor-Critic Policy
# ---------------------------------------------------------------------------


def _build_mlp(
    in_dim: int,
    hidden_sizes: Tuple[int, ...],
    out_dim: int,
) -> nn.Sequential:
    """Build a simple MLP with LayerNorm + Tanh hidden layers."""
    layers: list[nn.Module] = []
    cur = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(cur, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.Tanh())
        cur = h
    layers.append(nn.Linear(cur, out_dim))
    return nn.Sequential(*layers)


class EntityActorCriticPolicy(nn.Module):
    """Actor-critic policy that uses an :class:`EntityEncoder` as the backbone.

    Both the actor and the centralized critic share a single entity encoder
    (weight-sharing is optional and controlled by ``shared_encoder``).

    Actor head
    ----------
    Takes the pooled entity encoding ``(B, d_model)`` and produces a diagonal
    Gaussian action distribution.

    Critic head
    -----------
    Takes the pooled encoding of the **global** entity sequence (all units
    from both teams) and produces a scalar value estimate.

    Parameters
    ----------
    token_dim:
        Entity token dimensionality.  Defaults to ``ENTITY_TOKEN_DIM``.
    action_dim:
        Continuous action space dimensionality.
    d_model:
        Transformer internal dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of transformer encoder layers.
    actor_hidden_sizes:
        MLP hidden sizes applied on top of the entity encoding for the actor.
    critic_hidden_sizes:
        MLP hidden sizes applied on top of the entity encoding for the critic.
    shared_encoder:
        When ``True`` (default), actor and critic share the same
        :class:`EntityEncoder` weights.  When ``False`` they each have an
        independent encoder.
    dropout:
        Dropout probability in the transformer layers.
    use_spatial_pe:
        Enable 2-D Fourier positional encoding.
    """

    def __init__(
        self,
        token_dim: int = ENTITY_TOKEN_DIM,
        action_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        actor_hidden_sizes: Tuple[int, ...] = (128, 64),
        critic_hidden_sizes: Tuple[int, ...] = (128, 64),
        shared_encoder: bool = True,
        dropout: float = 0.0,
        use_spatial_pe: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.shared_encoder = shared_encoder

        # Entity encoders
        self.actor_encoder = EntityEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_spatial_pe=use_spatial_pe,
        )
        if shared_encoder:
            self.critic_encoder = self.actor_encoder
        else:
            self.critic_encoder = EntityEncoder(
                token_dim=token_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                use_spatial_pe=use_spatial_pe,
            )

        # Actor head: encoding → action mean
        self.actor_head = _build_mlp(d_model, actor_hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: encoding → scalar value
        self.critic_head = _build_mlp(d_model, critic_hidden_sizes, 1)

    # ------------------------------------------------------------------
    # Actor helpers
    # ------------------------------------------------------------------

    def get_distribution(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Normal:
        """Compute the action distribution for a batch of entity sequences.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)``.
        pad_mask:
            Boolean padding mask of shape ``(B, N)``.

        Returns
        -------
        dist : :class:`~torch.distributions.Normal`
        """
        enc = self.actor_encoder(tokens, pad_mask)  # (B, d_model)
        mean = self.actor_head(enc)  # (B, action_dim)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or select deterministically) actions for a batch.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)`` or ``(N, token_dim)`` for a single
            sample (batch dimension will be added automatically).
        pad_mask:
            Padding mask of shape ``(B, N)`` or ``(N,)`` for single sample.
        deterministic:
            Return the distribution mean instead of sampling.

        Returns
        -------
        actions   : torch.Tensor — shape ``(B, action_dim)``
        log_probs : torch.Tensor — shape ``(B,)``
        """
        squeezed = tokens.dim() == 2
        if squeezed:
            tokens = tokens.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)
        dist = self.get_distribution(tokens, pad_mask)
        actions = dist.mean if deterministic else dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for given token sequences and actions.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)``.
        actions:
            Shape ``(B, action_dim)``.
        pad_mask:
            Padding mask of shape ``(B, N)``.

        Returns
        -------
        log_probs : torch.Tensor — shape ``(B,)``
        entropy   : torch.Tensor — shape ``(B,)``
        """
        dist = self.get_distribution(tokens, pad_mask)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy

    # ------------------------------------------------------------------
    # Critic helpers
    # ------------------------------------------------------------------

    def get_value(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return value estimates for a batch of global entity sequences.

        Parameters
        ----------
        tokens:
            Global entity sequences of shape ``(B, N, token_dim)``.
        pad_mask:
            Padding mask of shape ``(B, N)``.

        Returns
        -------
        values : torch.Tensor — shape ``(B,)``
        """
        enc = self.critic_encoder(tokens, pad_mask)  # (B, d_model)
        return self.critic_head(enc).squeeze(-1)      # (B,)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict[str, int]:
        """Return a dict with actor and critic parameter counts."""
        actor_params = (
            sum(p.numel() for p in self.actor_encoder.parameters())
            + sum(p.numel() for p in self.actor_head.parameters())
            + self.log_std.numel()
        )
        if self.shared_encoder:
            critic_params = sum(p.numel() for p in self.critic_head.parameters())
        else:
            critic_params = (
                sum(p.numel() for p in self.critic_encoder.parameters())
                + sum(p.numel() for p in self.critic_head.parameters())
            )
        return {
            "actor": actor_params,
            "critic": critic_params,
            "total": actor_params + critic_params,
        }
