# SPDX-License-Identifier: MIT
# models/wfm1.py
"""Wargames Foundation Model 1 (WFM-1) — E12.1.

WFM-1 is a single transformer-based policy trained on the full v11 distribution
(procedural + real terrain, all weather, corps to battalion scale, all unit
types).  It zero-shot generalises to unseen scenarios and can be fine-tuned in
< 10 k gradient steps via a lightweight scenario-card adapter.

Architecture overview::

    entity tokens (B, N, token_dim)  [per echelon]
          │
    EchelonEncoder  →  echelon_enc (B, d_model)  [battalion / brigade / division / corps]
          │
    cross-echelon tokens (B, E, d_model)  [E = num echelons present]
          │
    CrossEchelonTransformer  →  fused_enc (B, d_model)
          │
    ScenarioCard (context vector, d_model)  ← map metadata, scale, weather …
          │
    FiLM modulation: scale = γ(card), shift = β(card)
    fused_enc ← fused_enc × γ + β
          │
    actor head  →  Gaussian action distribution
    critic head →  scalar value estimate

Echelons
--------
The model handles four echelon levels simultaneously:
* ``battalion`` — individual battalion units (used in 1v1 / 2v2 envs)
* ``brigade``   — brigade-level aggregation
* ``division``  — division-level aggregation
* ``corps``     — corps-level orchestration

Each echelon produces a pooled encoding via a shared :class:`EchelonEncoder`
(weight-tying across echelons by default; disabled with ``share_echelon_encoders=False``).
The cross-echelon transformer then integrates these representations.

Context-conditioned fine-tuning
--------------------------------
A :class:`ScenarioCard` encodes scenario metadata (map size, scale, weather,
unit-type mix, etc.) into a fixed-length conditioning vector.  FiLM
(Feature-wise Linear Modulation) applies per-feature scale and shift to the
fused encoding, allowing the policy to adapt to novel scenarios without
changing the base transformer weights.

Typical usage (inference)::

    from models.wfm1 import WFM1Policy, ScenarioCard

    policy = WFM1Policy(token_dim=16, action_dim=3)

    card = ScenarioCard(
        map_scale=0.5,          # normalised map size  [0, 1]
        echelon_level=1,        # 0=battalion, 1=brigade, 2=division, 3=corps
        weather_code=0,         # 0=clear, 1=rain, 2=fog, 3=snow
        n_blue_units=8,
        n_red_units=8,
        terrain_type=0,         # 0=procedural, 1=gis_waterloo, …
    )

    tokens = torch.zeros(1, 8, 16)  # (B, N, token_dim)
    actions, log_probs = policy.act(tokens, card=card)

Fine-tuning (10 k steps on a new scenario)::

    opt = torch.optim.Adam(policy.adapter_parameters(), lr=3e-4)
    for batch in finetune_dataloader:
        loss = policy.finetune_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from models.entity_encoder import (
    ENTITY_TOKEN_DIM,
    EntityEncoder,
    SpatialPositionalEncoding,
    _build_mlp,
)

__all__ = [
    "ECHELON_BATTALION",
    "ECHELON_BRIGADE",
    "ECHELON_DIVISION",
    "ECHELON_CORPS",
    "WEATHER_CLEAR",
    "WEATHER_RAIN",
    "WEATHER_FOG",
    "WEATHER_SNOW",
    "TERRAIN_PROCEDURAL",
    "TERRAIN_GIS_WATERLOO",
    "TERRAIN_GIS_AUSTERLITZ",
    "TERRAIN_GIS_BORODINO",
    "TERRAIN_GIS_SALAMANCA",
    "ScenarioCard",
    "EchelonEncoder",
    "CrossEchelonTransformer",
    "WFM1Policy",
]

# ---------------------------------------------------------------------------
# Echelon constants
# ---------------------------------------------------------------------------

ECHELON_BATTALION: int = 0
ECHELON_BRIGADE: int = 1
ECHELON_DIVISION: int = 2
ECHELON_CORPS: int = 3

_ECHELON_NAMES: Dict[int, str] = {
    ECHELON_BATTALION: "battalion",
    ECHELON_BRIGADE: "brigade",
    ECHELON_DIVISION: "division",
    ECHELON_CORPS: "corps",
}

_N_ECHELONS: int = 4

# ---------------------------------------------------------------------------
# Weather / terrain constants
# ---------------------------------------------------------------------------

WEATHER_CLEAR: int = 0
WEATHER_RAIN: int = 1
WEATHER_FOG: int = 2
WEATHER_SNOW: int = 3

TERRAIN_PROCEDURAL: int = 0
TERRAIN_GIS_WATERLOO: int = 1
TERRAIN_GIS_AUSTERLITZ: int = 2
TERRAIN_GIS_BORODINO: int = 3
TERRAIN_GIS_SALAMANCA: int = 4

#: Dimensionality of the raw scenario card feature vector (before projection).
_SCENARIO_CARD_RAW_DIM: int = 12

# ---------------------------------------------------------------------------
# ScenarioCard
# ---------------------------------------------------------------------------


@dataclass
class ScenarioCard:
    """Metadata descriptor for a training/evaluation scenario.

    Used to condition WFM-1 via FiLM modulation.  All numeric fields are
    expected to be pre-normalised to reasonable ranges; the policy encodes
    them into a conditioning vector via a small MLP.

    Attributes
    ----------
    map_scale:
        Map area normalised to ``[0, 1]`` (0 = smallest battalion map,
        1 = largest corps map).
    echelon_level:
        Primary echelon of the scenario.  One of :data:`ECHELON_BATTALION`,
        :data:`ECHELON_BRIGADE`, :data:`ECHELON_DIVISION`, :data:`ECHELON_CORPS`.
    weather_code:
        Active weather condition.  One of :data:`WEATHER_CLEAR`,
        :data:`WEATHER_RAIN`, :data:`WEATHER_FOG`, :data:`WEATHER_SNOW`.
    n_blue_units:
        Number of blue (friendly) units, normalised to ``[0, 1]`` before
        encoding (divided by ``max_units``; default 64).
    n_red_units:
        Number of red (enemy) units, similarly normalised.
    terrain_type:
        Integer terrain identifier.  0 = procedural; 1–4 = GIS sites.
    cavalry_fraction:
        Fraction of units that are cavalry (``[0, 1]``).
    artillery_fraction:
        Fraction of units that are artillery (``[0, 1]``).
    supply_pressure:
        Supply depletion index (``[0, 1]``; 0 = full supply, 1 = exhausted).
    time_of_day:
        Normalised time in ``[0, 1]`` (0 = dawn, 1 = dusk).
    max_units:
        Normalisation constant for unit counts.  Not encoded — used only
        during the :meth:`to_tensor` conversion.
    """

    map_scale: float = 0.5
    echelon_level: int = ECHELON_BATTALION
    weather_code: int = WEATHER_CLEAR
    n_blue_units: float = 8.0
    n_red_units: float = 8.0
    terrain_type: int = TERRAIN_PROCEDURAL
    cavalry_fraction: float = 0.0
    artillery_fraction: float = 0.0
    supply_pressure: float = 0.0
    time_of_day: float = 0.5
    max_units: float = 64.0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode the card as a 1-D float tensor of shape ``(_SCENARIO_CARD_RAW_DIM,)``.

        Encoding layout (12 dims):
        * [0]   map_scale                    float [0, 1]
        * [1:5] echelon one-hot              4 dims
        * [5:9] weather one-hot              4 dims
        * [9]   n_blue_units / max_units     float [0, 1]
        * [10]  n_red_units / max_units      float [0, 1]
        * [11]  terrain_type / 4             float [0, 1]

        Unit counts (``n_blue_units``, ``n_red_units``) are normalised
        internally by dividing by ``max_units``; callers do not need to
        pre-normalise them.

        Note: the remaining floating-point fields (``cavalry_fraction``,
        ``artillery_fraction``, ``supply_pressure``, ``time_of_day``) are
        stored on the dataclass but are **not** included in this 12-dim
        vector.  They can be appended manually for experimental extensions.
        """
        vec = torch.zeros(_SCENARIO_CARD_RAW_DIM)
        vec[0] = float(self.map_scale)

        echelon_idx = max(0, min(self.echelon_level, _N_ECHELONS - 1))
        vec[1 + echelon_idx] = 1.0  # one-hot [1:5]

        weather_idx = max(0, min(self.weather_code, 3))
        vec[5 + weather_idx] = 1.0  # one-hot [5:9]

        vec[9] = float(self.n_blue_units) / max(self.max_units, 1.0)
        vec[10] = float(self.n_red_units) / max(self.max_units, 1.0)
        vec[11] = float(self.terrain_type) / 4.0

        if device is not None:
            vec = vec.to(device)
        return vec


# ---------------------------------------------------------------------------
# EchelonEncoder
# ---------------------------------------------------------------------------


class EchelonEncoder(nn.Module):
    """Per-echelon entity encoder based on :class:`~models.entity_encoder.EntityEncoder`.

    Wraps an :class:`~models.entity_encoder.EntityEncoder` and optionally
    adds an echelon embedding that is summed into the token embeddings before
    the transformer layers.

    Parameters
    ----------
    token_dim:
        Entity token dimensionality.
    d_model:
        Transformer hidden dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of transformer encoder layers.
    dropout:
        Dropout probability in transformer layers.
    use_spatial_pe:
        Whether to add 2-D Fourier positional encoding.
    n_freq_bands:
        Fourier frequency bands for the spatial PE.
    use_echelon_embedding:
        When ``True``, learn a separate embedding per echelon level that is
        added to the projected token features before the transformer.
    """

    def __init__(
        self,
        token_dim: int = ENTITY_TOKEN_DIM,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.0,
        use_spatial_pe: bool = True,
        n_freq_bands: int = 8,
        use_echelon_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_echelon_embedding = use_echelon_embedding

        self.encoder = EntityEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_spatial_pe=use_spatial_pe,
            n_freq_bands=n_freq_bands,
        )

        if use_echelon_embedding:
            self.echelon_embed = nn.Embedding(_N_ECHELONS, d_model)
            nn.init.normal_(self.echelon_embed.weight, std=0.02)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the pooled output."""
        return self.d_model

    def forward(
        self,
        tokens: torch.Tensor,
        echelon: int,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode entity tokens for a given echelon level.

        Parameters
        ----------
        tokens:
            Entity token tensor of shape ``(B, N, token_dim)``.
        echelon:
            Integer echelon identifier (0–3).
        pad_mask:
            Boolean padding mask ``(B, N)``.  ``True`` = ignored position.

        Returns
        -------
        enc : torch.Tensor — shape ``(B, d_model)``
        """
        B, N, _ = tokens.shape

        # Get base token embeddings from the encoder's embed layer
        x = self.encoder.token_embed(tokens)  # (B, N, d_model)

        # Spatial positional encoding
        if self.encoder.use_spatial_pe:
            from models.entity_encoder import _SLICE_POSITION
            xy = tokens[..., _SLICE_POSITION]
            x = x + self.encoder.spatial_pe(xy)

        # Echelon embedding: broadcast over the entity sequence
        if self.use_echelon_embedding:
            echelon_int = int(echelon)
            num_echelons = self.echelon_embed.num_embeddings
            if not (0 <= echelon_int < num_echelons):
                raise ValueError(
                    f"Invalid echelon id {echelon!r}; expected integer in "
                    f"[0, {num_echelons - 1}]."
                )
            ech_idx = torch.tensor(echelon_int, device=tokens.device, dtype=torch.long)
            ech_emb = self.echelon_embed(ech_idx)  # (d_model,)
            x = x + ech_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)

        # Transformer
        x = self.encoder.transformer(x, src_key_padding_mask=pad_mask)

        # Mean-pool over non-padded positions
        if pad_mask is not None:
            keep = ~pad_mask  # (B, N)
            n_valid = keep.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            x = (x * keep.unsqueeze(-1).float()).sum(dim=1) / n_valid
        else:
            x = x.mean(dim=1)

        return self.encoder.out_proj(x)  # (B, d_model)


# ---------------------------------------------------------------------------
# CrossEchelonTransformer
# ---------------------------------------------------------------------------


class CrossEchelonTransformer(nn.Module):
    """Transformer that integrates encodings from multiple echelon levels.

    Takes a sequence of echelon encodings ``(B, E, d_model)`` where E is the
    number of active echelons, and applies multi-head self-attention to fuse
    information across echelon boundaries.

    Parameters
    ----------
    d_model:
        Hidden dimension (must match the echelon encoder output).
    n_heads:
        Number of attention heads.
    n_layers:
        Depth of the cross-echelon transformer.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Positional embedding for echelon order (battalion < brigade < …)
        self.echelon_pos_embed = nn.Embedding(_N_ECHELONS, d_model)
        nn.init.normal_(self.echelon_pos_embed.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        echelon_encs: torch.Tensor,
        echelon_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multi-echelon encodings.

        Parameters
        ----------
        echelon_encs:
            Stacked echelon encodings of shape ``(B, E, d_model)`` where
            E ≤ 4 is the number of active echelons.
        echelon_ids:
            Integer echelon identifiers of shape ``(E,)`` used to add
            order-aware positional embeddings.

        Returns
        -------
        fused : torch.Tensor — shape ``(B, d_model)``
            Mean-pooled fused representation.
        """
        # Add echelon-order positional embeddings
        pos = self.echelon_pos_embed(echelon_ids)  # (E, d_model)
        x = echelon_encs + pos.unsqueeze(0)        # (B, E, d_model)

        x = self.transformer(x)                    # (B, E, d_model)
        x = self.out_norm(x.mean(dim=1))           # (B, d_model)
        return x


# ---------------------------------------------------------------------------
# ScenarioCardEncoder (FiLM conditioning)
# ---------------------------------------------------------------------------


class _ScenarioCardEncoder(nn.Module):
    """Encode a :class:`ScenarioCard` and produce FiLM scale/shift parameters.

    Parameters
    ----------
    card_raw_dim:
        Dimensionality of the raw card feature vector.
    d_model:
        Target feature dimension (must match the fused encoder output).
    hidden_size:
        Width of the intermediate MLP.
    """

    def __init__(
        self,
        card_raw_dim: int = _SCENARIO_CARD_RAW_DIM,
        d_model: int = 128,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.mlp = nn.Sequential(
            nn.Linear(card_raw_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        # γ (scale) initialised to 1, β (shift) initialised to 0
        self.gamma_proj = nn.Linear(hidden_size, d_model)
        self.beta_proj = nn.Linear(hidden_size, d_model)

        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)  # γ = 1 initially (identity)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)  # β = 0 initially

    def forward(
        self,
        card_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute FiLM (γ, β) from a scenario card feature vector.

        Parameters
        ----------
        card_vec:
            Tensor of shape ``(B, card_raw_dim)`` or ``(card_raw_dim,)``
            (automatically expanded to batch dim).

        Returns
        -------
        gamma : torch.Tensor — shape ``(B, d_model)``
        beta  : torch.Tensor — shape ``(B, d_model)``
        """
        if card_vec.dim() == 1:
            card_vec = card_vec.unsqueeze(0)  # (1, card_raw_dim)
        h = self.mlp(card_vec)
        return self.gamma_proj(h), self.beta_proj(h)


# ---------------------------------------------------------------------------
# WFM1Policy
# ---------------------------------------------------------------------------


class WFM1Policy(nn.Module):
    """WFM-1 hierarchical transformer policy.

    A single policy that operates across battalion, brigade, division, and
    corps echelons.  Multi-echelon information is fused via cross-echelon
    attention.  A lightweight ScenarioCard FiLM adapter conditions the policy
    on scenario metadata, enabling efficient fine-tuning.

    Architecture::

        For each active echelon e ∈ {battalion, brigade, division, corps}:
          tokens_e (B, N_e, token_dim)
                │
          EchelonEncoder(echelon=e)  →  enc_e (B, d_model)

        Stack: echelon_encs = [enc_e₁, enc_e₂, …]  shape (B, E, d_model)
                │
        CrossEchelonTransformer  →  fused (B, d_model)
                │
        FiLM(ScenarioCard) :  fused ← fused × γ + β
                │
        actor head  →  Gaussian action distribution
        critic head →  scalar value

    Parameters
    ----------
    token_dim:
        Entity token dimensionality.  Defaults to :data:`~models.entity_encoder.ENTITY_TOKEN_DIM`.
    action_dim:
        Continuous action space dimensionality.
    d_model:
        Transformer hidden dimension.
    n_heads:
        Attention heads for both echelon encoder and cross-echelon transformer.
    n_echelon_layers:
        Transformer depth for each :class:`EchelonEncoder`.
    n_cross_layers:
        Transformer depth for :class:`CrossEchelonTransformer`.
    actor_hidden_sizes:
        MLP hidden sizes for the actor head.
    critic_hidden_sizes:
        MLP hidden sizes for the critic head.
    dropout:
        Dropout probability (transformer only).
    use_spatial_pe:
        Enable 2-D Fourier positional encoding in the echelon encoders.
    share_echelon_encoders:
        When ``True`` (default), all four echelon levels share a single
        :class:`EchelonEncoder` instance.  When ``False`` each echelon has
        independent weights.
    card_hidden_size:
        Width of the FiLM adapter MLP.
    """

    def __init__(
        self,
        token_dim: int = ENTITY_TOKEN_DIM,
        action_dim: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        n_echelon_layers: int = 4,
        n_cross_layers: int = 2,
        actor_hidden_sizes: Tuple[int, ...] = (256, 128),
        critic_hidden_sizes: Tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        use_spatial_pe: bool = True,
        share_echelon_encoders: bool = True,
        card_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.share_echelon_encoders = share_echelon_encoders
        self.dropout = dropout

        # --- Echelon encoders -------------------------------------------------
        if share_echelon_encoders:
            _shared = EchelonEncoder(
                token_dim=token_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_echelon_layers,
                dropout=dropout,
                use_spatial_pe=use_spatial_pe,
                use_echelon_embedding=True,
            )
            self.echelon_encoders = nn.ModuleList([_shared] * _N_ECHELONS)
        else:
            self.echelon_encoders = nn.ModuleList([
                EchelonEncoder(
                    token_dim=token_dim,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_echelon_layers,
                    dropout=dropout,
                    use_spatial_pe=use_spatial_pe,
                    use_echelon_embedding=True,
                )
                for _ in range(_N_ECHELONS)
            ])

        # --- Cross-echelon transformer ----------------------------------------
        self.cross_echelon = CrossEchelonTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_cross_layers,
            dropout=dropout,
        )

        # --- Scenario card FiLM adapter ---------------------------------------
        self.card_encoder = _ScenarioCardEncoder(
            card_raw_dim=_SCENARIO_CARD_RAW_DIM,
            d_model=d_model,
            hidden_size=card_hidden_size,
        )

        # --- Actor / critic heads ---------------------------------------------
        self.actor_head = _build_mlp(d_model, actor_hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = _build_mlp(d_model, critic_hidden_sizes, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        tokens_per_echelon: Dict[int, torch.Tensor],
        pad_masks: Optional[Dict[int, torch.Tensor]] = None,
        card: Optional[ScenarioCard] = None,
        card_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce the FiLM-modulated fused encoding.

        Parameters
        ----------
        tokens_per_echelon:
            Mapping from echelon id → entity token tensor ``(B, N_e, token_dim)``.
            At least one echelon must be provided.
        pad_masks:
            Optional mapping from echelon id → padding mask ``(B, N_e)``.
        card:
            :class:`ScenarioCard` to condition on.  Mutually exclusive with
            ``card_vec``; if both are ``None`` the FiLM adapter produces
            identity transforms (γ=1, β=0).
        card_vec:
            Pre-computed scenario card tensor ``(B, card_raw_dim)`` or
            ``(card_raw_dim,)``.  Takes priority over ``card``.

        Returns
        -------
        fused : torch.Tensor — shape ``(B, d_model)``
        """
        if not tokens_per_echelon:
            raise ValueError("tokens_per_echelon must contain at least one echelon.")

        pad_masks = pad_masks or {}
        echelon_ids_list: List[int] = sorted(tokens_per_echelon.keys())

        # Encode each echelon
        echelon_encs: List[torch.Tensor] = []
        for eid in echelon_ids_list:
            enc = self.echelon_encoders[eid](
                tokens_per_echelon[eid],
                echelon=eid,
                pad_mask=pad_masks.get(eid),
            )  # (B, d_model)
            echelon_encs.append(enc)

        # Stack → (B, E, d_model) for cross-echelon transformer
        stacked = torch.stack(echelon_encs, dim=1)  # (B, E, d_model)
        echelon_ids_t = torch.tensor(
            echelon_ids_list,
            device=stacked.device,
            dtype=torch.long,
        )  # (E,)
        fused = self.cross_echelon(stacked, echelon_ids_t)  # (B, d_model)

        # FiLM modulation from scenario card
        if card_vec is not None:
            if card_vec.device != fused.device:
                card_vec = card_vec.to(fused.device)
            gamma, beta = self.card_encoder(card_vec)
        elif card is not None:
            cv = card.to_tensor(device=fused.device)
            gamma, beta = self.card_encoder(cv)
        else:
            # Identity FiLM: γ=1, β=0 (no modulation)
            gamma = torch.ones(fused.shape[0], self.d_model, device=fused.device)
            beta = torch.zeros(fused.shape[0], self.d_model, device=fused.device)

        fused = fused * gamma + beta  # (B, d_model)
        return fused

    # ------------------------------------------------------------------
    # Single-echelon convenience shortcut
    # ------------------------------------------------------------------

    def _encode_single(
        self,
        tokens: torch.Tensor,
        echelon: int = ECHELON_BATTALION,
        pad_mask: Optional[torch.Tensor] = None,
        card: Optional[ScenarioCard] = None,
        card_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode tokens from a single echelon (most common use-case)."""
        return self._encode(
            tokens_per_echelon={echelon: tokens},
            pad_masks={echelon: pad_mask} if pad_mask is not None else None,
            card=card,
            card_vec=card_vec,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        tokens: Optional[torch.Tensor],
        pad_mask: Optional[torch.Tensor] = None,
        echelon: int = ECHELON_BATTALION,
        card: Optional[ScenarioCard] = None,
        card_vec: Optional[torch.Tensor] = None,
        tokens_per_echelon: Optional[Dict[int, torch.Tensor]] = None,
        pad_masks: Optional[Dict[int, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions (no gradient).

        Can be called with a single-echelon ``tokens`` tensor or with a
        full ``tokens_per_echelon`` dict for multi-echelon inputs.

        Parameters
        ----------
        tokens:
            Single-echelon entity tokens ``(B, N, token_dim)``.
            Ignored when ``tokens_per_echelon`` is provided.
        pad_mask:
            Padding mask for ``tokens``.
        echelon:
            Active echelon level for single-echelon mode.
        card:
            Scenario conditioning card.
        card_vec:
            Pre-computed scenario card tensor (alternative to ``card``).
        tokens_per_echelon:
            Multi-echelon input dict; overrides ``tokens`` when given.
        pad_masks:
            Padding masks for multi-echelon mode.
        deterministic:
            When ``True``, return the distribution mean instead of a sample.

        Returns
        -------
        actions   : torch.Tensor — shape ``(B, action_dim)``
        log_probs : torch.Tensor — shape ``(B,)``
        """
        if tokens_per_echelon is not None:
            fused = self._encode(
                tokens_per_echelon, pad_masks, card=card, card_vec=card_vec
            )
        else:
            fused = self._encode_single(tokens, echelon, pad_mask, card, card_vec)

        mean = self.actor_head(fused)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        actions = mean if deterministic else dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def get_value(
        self,
        tokens: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        echelon: int = ECHELON_BATTALION,
        card: Optional[ScenarioCard] = None,
        card_vec: Optional[torch.Tensor] = None,
        tokens_per_echelon: Optional[Dict[int, torch.Tensor]] = None,
        pad_masks: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute scalar value estimates.

        Returns
        -------
        values : torch.Tensor — shape ``(B,)``
        """
        if tokens_per_echelon is not None:
            fused = self._encode(
                tokens_per_echelon, pad_masks, card=card, card_vec=card_vec
            )
        else:
            fused = self._encode_single(tokens, echelon, pad_mask, card, card_vec)

        return self.critic_head(fused).squeeze(-1)  # (B,)

    def evaluate_actions(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        echelon: int = ECHELON_BATTALION,
        card: Optional[ScenarioCard] = None,
        card_vec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, entropy, and values for given actions.

        Used during PPO update.

        Returns
        -------
        log_probs : torch.Tensor — shape ``(B,)``
        entropy   : torch.Tensor — shape ``(B,)``
        values    : torch.Tensor — shape ``(B,)``
        """
        fused = self._encode_single(tokens, echelon, pad_mask, card, card_vec)

        mean = self.actor_head(fused)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic_head(fused).squeeze(-1)
        return log_probs, entropy, values

    # ------------------------------------------------------------------
    # Fine-tuning adapter API
    # ------------------------------------------------------------------

    def adapter_parameters(self) -> List[nn.Parameter]:
        """Return only the FiLM adapter parameters.

        Call this to get a parameter group for lightweight scenario-specific
        fine-tuning (adapter-only gradient updates leave the base transformer
        frozen).

        Returns
        -------
        list of :class:`torch.nn.Parameter`
        """
        return list(self.card_encoder.parameters())

    def base_parameters(self) -> List[nn.Parameter]:
        """Return all non-adapter (base model) parameters."""
        adapter_ids = {id(p) for p in self.adapter_parameters()}
        return [p for p in self.parameters() if id(p) not in adapter_ids]

    def freeze_base(self) -> None:
        """Freeze all base-model parameters; only the adapter remains trainable."""
        for p in self.base_parameters():
            p.requires_grad_(False)

    def unfreeze_base(self) -> None:
        """Unfreeze all base-model parameters."""
        for p in self.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Checkpoint utilities
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model state dict to *path* (``.pt``)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "token_dim": self.token_dim,
                    "action_dim": self.action_dim,
                    "d_model": self.d_model,
                    "n_heads": self.echelon_encoders[0].encoder.n_heads,
                    "n_echelon_layers": self.echelon_encoders[0].encoder.n_layers,
                    "n_cross_layers": self.cross_echelon.transformer.num_layers,
                    "actor_hidden_sizes": tuple(
                        layer.out_features
                        for layer in self.actor_head
                        if isinstance(layer, nn.Linear)
                    )[:-1],
                    "critic_hidden_sizes": tuple(
                        layer.out_features
                        for layer in self.critic_head
                        if isinstance(layer, nn.Linear)
                    )[:-1],
                    "dropout": self.dropout,
                    "share_echelon_encoders": self.share_echelon_encoders,
                    "card_hidden_size": self.card_encoder.mlp[0].out_features,
                },
            },
            out,
        )
        return out

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: Optional[torch.device] = None,
        **kwargs,
    ) -> "WFM1Policy":
        """Load a WFM-1 checkpoint produced by :meth:`save_checkpoint`.

        Extra keyword arguments override the saved configuration.
        """
        ckpt = torch.load(path, map_location=map_location)
        cfg = {**ckpt["config"], **kwargs}
        policy = cls(**cfg)
        policy.load_state_dict(ckpt["state_dict"])
        return policy

    def finetune_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute a supervised fine-tuning loss from a demonstration batch.

        The batch dict must contain:
        * ``"tokens"``   — shape ``(B, N, token_dim)``
        * ``"actions"``  — shape ``(B, action_dim)``

        Optional keys:
        * ``"pad_mask"`` — shape ``(B, N)``
        * ``"echelon"``  — scalar int (default: ``ECHELON_BATTALION``)
        * ``"card_vec"`` — shape ``(B, card_raw_dim)`` or ``(card_raw_dim,)``

        Returns
        -------
        loss : torch.Tensor — scalar behaviour-cloning MSE loss
        """
        tokens = batch["tokens"]
        target_actions = batch["actions"]
        pad_mask = batch.get("pad_mask")
        echelon = int(batch["echelon"]) if "echelon" in batch else ECHELON_BATTALION
        card_vec = batch.get("card_vec")

        fused = self._encode_single(tokens, echelon, pad_mask, card_vec=card_vec)
        pred_mean = self.actor_head(fused)
        return nn.functional.mse_loss(pred_mean, target_actions)
