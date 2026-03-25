# SPDX-License-Identifier: MIT
# models/recurrent_policy.py
"""Recurrent actor-critic policy with LSTM memory module.

Implements E8.2: wraps :class:`~models.entity_encoder.EntityEncoder` with a
multi-layer LSTM so agents accumulate temporal context across timesteps.
This enables tracking unit positions that have moved out of LOS, recalling
recent enemy behaviour, and maintaining operational intent under fog-of-war.

Architecture overview::

    entity tokens (B, N, token_dim)
          │
    EntityEncoder  →  pooled encoding (B, d_model)
          │
    nn.LSTM (num_layers × hidden_size)
          │  ← (h_t, c_t) passed in / out each step
    LSTM output (B, hidden_size)
          │
    actor head  →  Gaussian action distribution
    critic head →  scalar value

Typical usage (single-step inference)::

    from models.recurrent_policy import RecurrentActorCriticPolicy

    policy = RecurrentActorCriticPolicy(
        token_dim=16,
        action_dim=3,
        d_model=64,
        lstm_hidden_size=128,
        lstm_num_layers=1,
    )

    # Initial zero hidden state for batch of 1
    hx = policy.initial_state(batch_size=1)

    tokens = torch.zeros(1, 8, 16)          # (B=1, N=8, token_dim=16)
    actions, log_probs, hx = policy.act(tokens, hx)

    # At episode boundary — reset hidden state to zeros
    hx = policy.initial_state(batch_size=1)

Rollout buffer usage::

    from models.recurrent_policy import RecurrentRolloutBuffer

    buf = RecurrentRolloutBuffer(
        n_steps=128,
        max_entities=16,
        token_dim=16,
        action_dim=3,
        lstm_hidden_size=128,
        lstm_num_layers=1,
    )

    buf.add(tokens, hx, action, log_prob, reward, done, value)
    buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
    for batch in buf.get_sequences(seq_len=16, device=device):
        # batch: dict with "tokens", "hx_h", "hx_c", "actions", etc.
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from models.entity_encoder import (
    ENTITY_TOKEN_DIM,
    EntityEncoder,
    _build_mlp,
)

#: Small constant added to the standard deviation during advantage normalisation
#: to avoid division by zero when all advantages are identical.
_NORMALIZATION_EPSILON: float = 1e-8

__all__ = [
    "LSTMHiddenState",
    "RecurrentEntityEncoder",
    "RecurrentActorCriticPolicy",
    "RecurrentRolloutBuffer",
]


# ---------------------------------------------------------------------------
# Hidden-state container
# ---------------------------------------------------------------------------


@dataclass
class LSTMHiddenState:
    """Container for LSTM ``(h, c)`` states with episode-reset utilities.

    Both tensors have shape ``(num_layers, batch, hidden_size)``.

    Parameters
    ----------
    h:
        Hidden state tensor.
    c:
        Cell state tensor.
    """

    h: torch.Tensor  # (num_layers, batch, hidden_size)
    c: torch.Tensor  # (num_layers, batch, hidden_size)

    def detach(self) -> "LSTMHiddenState":
        """Return a new :class:`LSTMHiddenState` with detached tensors."""
        return LSTMHiddenState(h=self.h.detach(), c=self.c.detach())

    def to(self, device: torch.device) -> "LSTMHiddenState":
        """Move states to *device*."""
        return LSTMHiddenState(h=self.h.to(device), c=self.c.to(device))

    def reset_at(self, done_mask: torch.Tensor) -> "LSTMHiddenState":
        """Zero out hidden states for episodes that have ended.

        Parameters
        ----------
        done_mask:
            Boolean tensor of shape ``(batch,)`` — ``True`` where the episode
            ended and the hidden state should be cleared.

        Returns
        -------
        LSTMHiddenState with states zeroed at ``done_mask`` positions.
        """
        mask = done_mask.to(self.h.device)  # (batch,)
        # Broadcast mask over (num_layers, batch, hidden_size)
        mask_3d = mask.unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)
        h_new = self.h.masked_fill(mask_3d, 0.0)
        c_new = self.c.masked_fill(mask_3d, 0.0)
        return LSTMHiddenState(h=h_new, c=c_new)

    @classmethod
    def zeros(
        cls,
        num_layers: int,
        hidden_size: int,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> "LSTMHiddenState":
        """Create a zero-initialised hidden state.

        Parameters
        ----------
        num_layers:
            Number of LSTM layers.
        hidden_size:
            LSTM hidden dimension.
        batch_size:
            Batch dimension.
        device:
            Target device (defaults to CPU).
        """
        if device is None:
            device = torch.device("cpu")
        shape = (num_layers, batch_size, hidden_size)
        return cls(
            h=torch.zeros(*shape, device=device),
            c=torch.zeros(*shape, device=device),
        )

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(h, c)`` tuple accepted by ``nn.LSTM``."""
        return (self.h, self.c)


# ---------------------------------------------------------------------------
# Recurrent entity encoder
# ---------------------------------------------------------------------------


class RecurrentEntityEncoder(nn.Module):
    """Entity encoder followed by a multi-layer LSTM for temporal memory.

    The entity encoder reduces a variable-length set of entity tokens to a
    fixed-size pooled vector.  The LSTM then integrates this encoding across
    timesteps, maintaining an internal model of unobserved unit positions.

    Architecture::

        tokens (B, N, token_dim)
              │
        EntityEncoder  →  enc (B, d_model)
              │
        nn.LSTM(d_model → hidden_size, num_layers)
              │  ← (h_t, c_t)  in / out
        lstm_out (B, hidden_size)

    Parameters
    ----------
    token_dim:
        Entity token dimensionality.
    d_model:
        Transformer internal dimension (EntityEncoder output).
    n_heads:
        Attention heads for the entity encoder.
    n_layers:
        Transformer encoder layers.
    lstm_hidden_size:
        LSTM hidden state dimensionality.
    lstm_num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied inside transformer and LSTM.
    use_spatial_pe:
        Enable 2-D Fourier positional encoding on entity tokens.
    """

    def __init__(
        self,
        token_dim: int = ENTITY_TOKEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        dropout: float = 0.0,
        use_spatial_pe: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.entity_encoder = EntityEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_spatial_pe=use_spatial_pe,
        )

        lstm_dropout = dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    @property
    def output_dim(self) -> int:
        """Dimensionality of the LSTM output vector."""
        return self.lstm_hidden_size

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> LSTMHiddenState:
        """Return a zero-initialised hidden state for *batch_size* samples."""
        return LSTMHiddenState.zeros(
            self.lstm_num_layers, self.lstm_hidden_size, batch_size, device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        hx: LSTMHiddenState,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LSTMHiddenState]:
        """Encode a single timestep of entity tokens.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)`` — entity tokens for one timestep.
        hx:
            Current LSTM hidden state.
        pad_mask:
            Boolean padding mask of shape ``(B, N)``.

        Returns
        -------
        out : torch.Tensor — shape ``(B, lstm_hidden_size)``
            LSTM output for this timestep.
        new_hx : LSTMHiddenState
            Updated hidden and cell states.
        """
        enc = self.entity_encoder(tokens, pad_mask)  # (B, d_model)
        # LSTM expects (B, seq_len, input_size); seq_len = 1 for single-step
        lstm_in = enc.unsqueeze(1)  # (B, 1, d_model)
        lstm_out, (h_new, c_new) = self.lstm(lstm_in, hx.as_tuple())
        out = lstm_out.squeeze(1)  # (B, lstm_hidden_size)
        return out, LSTMHiddenState(h=h_new, c=c_new)

    def forward_sequence(
        self,
        tokens_seq: torch.Tensor,
        hx: LSTMHiddenState,
        pad_mask_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LSTMHiddenState]:
        """Encode a *sequence* of timesteps for BPTT during training.

        Parameters
        ----------
        tokens_seq:
            Shape ``(B, T, N, token_dim)`` — entity token sequences.
        hx:
            Initial hidden state at the start of the sequence.
        pad_mask_seq:
            Padding mask of shape ``(B, T, N)`` or ``None``.

        Returns
        -------
        out_seq : torch.Tensor — shape ``(B, T, lstm_hidden_size)``
            LSTM outputs for every timestep.
        new_hx : LSTMHiddenState
            Hidden state after the last timestep.
        """
        B, T, N, _ = tokens_seq.shape

        # Flatten (B, T) → (B*T) for the entity encoder
        tokens_flat = tokens_seq.reshape(B * T, N, -1)
        pad_flat = (
            pad_mask_seq.reshape(B * T, N) if pad_mask_seq is not None else None
        )
        enc_flat = self.entity_encoder(tokens_flat, pad_flat)  # (B*T, d_model)
        enc_seq = enc_flat.reshape(B, T, self.d_model)  # (B, T, d_model)

        lstm_out, (h_new, c_new) = self.lstm(enc_seq, hx.as_tuple())
        return lstm_out, LSTMHiddenState(h=h_new, c=c_new)


# ---------------------------------------------------------------------------
# Recurrent actor-critic policy
# ---------------------------------------------------------------------------


class RecurrentActorCriticPolicy(nn.Module):
    """Actor-critic policy with LSTM temporal memory.

    Both actor and critic run through the same :class:`RecurrentEntityEncoder`
    (weight-sharing optional).  Hidden states are passed in/out explicitly so
    the caller controls episode boundaries and checkpointing.

    Parameters
    ----------
    token_dim:
        Entity token dimensionality.
    action_dim:
        Continuous action space dimensionality.
    d_model:
        Transformer internal dimension.
    n_heads:
        Attention heads in the entity encoder.
    n_layers:
        Transformer encoder layers.
    lstm_hidden_size:
        LSTM hidden state dimensionality.
    lstm_num_layers:
        Number of stacked LSTM layers.
    actor_hidden_sizes:
        MLP hidden sizes on top of the LSTM output for the actor head.
    critic_hidden_sizes:
        MLP hidden sizes on top of the LSTM output for the critic head.
    shared_encoder:
        When ``True`` (default), actor and critic share the same
        :class:`RecurrentEntityEncoder` weights.
    dropout:
        Dropout probability.
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
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
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
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.shared_encoder = shared_encoder

        # Build actor recurrent encoder
        self.actor_encoder = RecurrentEntityEncoder(
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
            use_spatial_pe=use_spatial_pe,
        )

        if shared_encoder:
            self.critic_encoder = self.actor_encoder
        else:
            self.critic_encoder = RecurrentEntityEncoder(
                token_dim=token_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                dropout=dropout,
                use_spatial_pe=use_spatial_pe,
            )

        # Actor head: lstm_out → action mean
        self.actor_head = _build_mlp(lstm_hidden_size, actor_hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: lstm_out → scalar value
        self.critic_head = _build_mlp(lstm_hidden_size, critic_hidden_sizes, 1)

    # ------------------------------------------------------------------
    # Hidden-state utilities
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> LSTMHiddenState:
        """Return a zero-initialised LSTM hidden state.

        Call at the start of each episode to reset temporal memory.

        Parameters
        ----------
        batch_size:
            Number of parallel environments / samples.
        device:
            Target device.
        """
        return LSTMHiddenState.zeros(
            self.lstm_num_layers, self.lstm_hidden_size, batch_size, device
        )

    # ------------------------------------------------------------------
    # Single-step inference
    # ------------------------------------------------------------------

    def get_distribution(
        self,
        tokens: torch.Tensor,
        hx: LSTMHiddenState,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Normal, LSTMHiddenState]:
        """Compute action distribution for a single timestep.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)`` or ``(N, token_dim)`` for unbatched.
        hx:
            Current LSTM hidden state.
        pad_mask:
            Boolean padding mask ``(B, N)`` or ``(N,)`` for unbatched.

        Returns
        -------
        dist : :class:`~torch.distributions.Normal`
        new_hx : :class:`LSTMHiddenState`
        """
        squeezed = tokens.dim() == 2
        if squeezed:
            tokens = tokens.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)

        out, new_hx = self.actor_encoder(tokens, hx, pad_mask)  # (B, hidden)
        mean = self.actor_head(out)  # (B, action_dim)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), new_hx

    @torch.no_grad()
    def act(
        self,
        tokens: torch.Tensor,
        hx: LSTMHiddenState,
        pad_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, LSTMHiddenState]:
        """Sample (or select deterministically) an action for a single step.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)`` or ``(N, token_dim)`` for single sample.
        hx:
            Current LSTM hidden state.
        pad_mask:
            Padding mask.
        deterministic:
            Return the distribution mean instead of sampling.

        Returns
        -------
        actions   : torch.Tensor — shape ``(B, action_dim)``
        log_probs : torch.Tensor — shape ``(B,)``
        new_hx    : :class:`LSTMHiddenState` — updated hidden state
        """
        dist, new_hx = self.get_distribution(tokens, hx, pad_mask)
        actions = dist.mean if deterministic else dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs, new_hx

    @torch.no_grad()
    def get_value(
        self,
        tokens: torch.Tensor,
        hx: LSTMHiddenState,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LSTMHiddenState]:
        """Compute value estimates for a single timestep.

        Parameters
        ----------
        tokens:
            Shape ``(B, N, token_dim)`` or ``(N, token_dim)`` for single sample.
        hx:
            Current LSTM hidden state.
        pad_mask:
            Padding mask.

        Returns
        -------
        values  : torch.Tensor — shape ``(B,)``
        new_hx  : :class:`LSTMHiddenState`
        """
        squeezed = tokens.dim() == 2
        if squeezed:
            tokens = tokens.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)

        out, new_hx = self.critic_encoder(tokens, hx, pad_mask)  # (B, hidden)
        values = self.critic_head(out).squeeze(-1)  # (B,)
        return values, new_hx

    # ------------------------------------------------------------------
    # Sequence-level training (BPTT)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        tokens_seq: torch.Tensor,
        hx: LSTMHiddenState,
        actions_seq: torch.Tensor,
        pad_mask_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probs, entropy and values for a sequence of actions.

        Used during PPO update to compute the ratio ``π_new / π_old``.

        Parameters
        ----------
        tokens_seq:
            Entity token sequences of shape ``(B, T, N, token_dim)``.
        hx:
            Initial LSTM hidden state at the start of the sequence.
        actions_seq:
            Actions to evaluate, shape ``(B, T, action_dim)``.
        pad_mask_seq:
            Padding mask of shape ``(B, T, N)`` or ``None``.

        Returns
        -------
        log_probs : torch.Tensor — shape ``(B, T)``
        entropy   : torch.Tensor — shape ``(B, T)``
        values    : torch.Tensor — shape ``(B, T)``
        """
        B, T, N, _ = tokens_seq.shape

        # Actor forward through the sequence
        actor_out, _ = self.actor_encoder.forward_sequence(
            tokens_seq, hx, pad_mask_seq
        )  # (B, T, hidden)
        mean = self.actor_head(actor_out)  # (B, T, action_dim)
        std = self.log_std.exp().unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1)  # (B, T)
        entropy = dist.entropy().sum(dim=-1)  # (B, T)

        # Critic forward — separate pass if encoders are not shared
        if self.shared_encoder:
            critic_out = actor_out  # reuse
        else:
            critic_out, _ = self.critic_encoder.forward_sequence(
                tokens_seq, hx, pad_mask_seq
            )  # (B, T, hidden)
        values = self.critic_head(critic_out).squeeze(-1)  # (B, T)

        return log_probs, entropy, values

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model weights to *path* (``torch.save`` format).

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        **kwargs,  # forwarded to __init__
    ) -> "RecurrentActorCriticPolicy":
        """Restore a policy from a checkpoint created by :meth:`save_checkpoint`.

        Parameters
        ----------
        path:
            Path to the checkpoint file.
        device:
            Device to load weights onto.
        **kwargs:
            Constructor arguments — must match those used when the checkpoint
            was saved.

        Returns
        -------
        policy : :class:`RecurrentActorCriticPolicy`
        """
        policy = cls(**kwargs)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        policy.load_state_dict(state_dict)
        if device is not None:
            policy.to(device)
        return policy

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parameter_count(self) -> Dict[str, int]:
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


# ---------------------------------------------------------------------------
# Recurrent rollout buffer
# ---------------------------------------------------------------------------


class RecurrentRolloutBuffer:
    """On-policy rollout buffer that stores LSTM hidden states for BPTT.

    Stores per-step hidden states alongside the usual rollout data so that PPO
    updates can re-run the LSTM through consecutive sequences (truncated BPTT)
    starting from the exact hidden state present during collection.

    Note
    ----
    This buffer does not automatically reset LSTM hidden states at episode
    boundaries.  When ``done=True`` at step *t*, the caller is responsible for
    providing an appropriately reset (typically zeroed) hidden state for
    step *t+1* via :meth:`~models.recurrent_policy.RecurrentActorCriticPolicy.initial_state`.

    Parameters
    ----------
    n_steps:
        Number of environment steps per rollout.
    max_entities:
        Maximum number of entity tokens per step (pad shorter observations).
    token_dim:
        Entity token dimensionality.
    action_dim:
        Action space dimensionality.
    lstm_hidden_size:
        LSTM hidden state dimensionality.
    lstm_num_layers:
        Number of LSTM layers.
    gamma:
        Discount factor for GAE.
    gae_lambda:
        GAE smoothing parameter λ.
    """

    def __init__(
        self,
        n_steps: int,
        max_entities: int,
        token_dim: int = ENTITY_TOKEN_DIM,
        action_dim: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.max_entities = max_entities
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Zero all storage arrays and reset the write pointer."""
        T = self.n_steps
        N = self.max_entities
        A = self.action_dim
        H = self.lstm_hidden_size
        L = self.lstm_num_layers

        # Main transition data
        self.tokens = np.zeros((T, N, self.token_dim), dtype=np.float32)
        self.pad_masks = np.zeros((T, N), dtype=bool)
        self.actions = np.zeros((T, A), dtype=np.float32)
        self.log_probs = np.zeros(T, dtype=np.float32)
        self.rewards = np.zeros(T, dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.float32)
        self.values = np.zeros(T, dtype=np.float32)

        # LSTM hidden states at the *start* of each step (before processing)
        self.hx_h = np.zeros((T, L, H), dtype=np.float32)
        self.hx_c = np.zeros((T, L, H), dtype=np.float32)

        # Filled by compute_returns_and_advantages
        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)

        self._ptr = 0
        self._full = False

    def reset(self) -> None:
        """Public reset — call before each new rollout collection."""
        self._reset()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add(
        self,
        tokens: np.ndarray,
        hx: LSTMHiddenState,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        pad_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Store one environment transition.

        Parameters
        ----------
        tokens:
            Entity tokens of shape ``(N_obs, token_dim)`` — the observation
            for this step.  Automatically zero-padded to ``max_entities``.
        hx:
            LSTM hidden state **at the start of this step** (before feeding
            *tokens* through the encoder).
        action:
            Action taken, shape ``(action_dim,)``.
        log_prob:
            Log-probability of *action* under the collection policy.
        reward:
            Scalar reward received.
        done:
            Whether the episode ended after this step.
        value:
            Critic value estimate for this step.
        pad_mask:
            Boolean padding mask of shape ``(N_obs,)``.  When omitted, valid
            positions (``[:n_obs]``) are left unmasked (``False``) and any
            remaining padded positions (``[n_obs:]``) are set to ``True``
            (ignored by attention).
        """
        if self._full:
            raise RuntimeError(
                "RecurrentRolloutBuffer is full — call reset() before adding."
            )
        t = self._ptr

        # Store tokens (with zero-padding to max_entities)
        n_obs = min(tokens.shape[0], self.max_entities)
        self.tokens[t, :n_obs] = tokens[:n_obs]
        if pad_mask is not None:
            self.pad_masks[t, :n_obs] = pad_mask[:n_obs]
            self.pad_masks[t, n_obs:] = True  # pad the rest
        else:
            self.pad_masks[t, n_obs:] = True

        # LSTM hidden state at start of this step (squeeze batch dim = 1).
        # NOTE: RecurrentRolloutBuffer currently assumes a single environment
        # (batch size = 1) for the LSTM hidden state. If a batched hidden state
        # is passed in (e.g., from multiple parallel envs), we raise explicitly
        # to avoid silently dropping all but the first environment.
        if hx.h.dim() != 3 or hx.c.dim() != 3:
            raise ValueError(
                f"Expected hx.h and hx.c to have 3 dimensions "
                f"(num_layers, batch_size, hidden_size); "
                f"got hx.h.dim()={hx.h.dim()}, hx.c.dim()={hx.c.dim()}."
            )
        if hx.h.size(1) != 1 or hx.c.size(1) != 1:
            raise ValueError(
                "RecurrentRolloutBuffer.add currently supports only a single "
                "environment (batch_size=1) for LSTM hidden state. "
                f"Got hx.h.shape={tuple(hx.h.shape)}, "
                f"hx.c.shape={tuple(hx.c.shape)}."
            )
        self.hx_h[t] = hx.h[:, 0, :].detach().cpu().numpy()
        self.hx_c[t] = hx.c[:, 0, :].detach().cpu().numpy()

        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.dones[t] = float(done)
        self.values[t] = value

        self._ptr += 1
        if self._ptr == self.n_steps:
            self._full = True

    # ------------------------------------------------------------------
    # GAE return computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Must be called once the buffer is full (all *n_steps* transitions
        have been added).

        Parameters
        ----------
        last_value:
            Critic value estimate for the state **after** the last stored step
            (bootstrap value).  Set to ``0.0`` when the last step was terminal.
        last_done:
            Whether the last step was terminal.
        """
        if not self._full and self._ptr != self.n_steps:
            raise RuntimeError(
                "Buffer is not yet full — add all transitions before computing returns."
            )
        gae = 0.0
        next_value = last_value

        for t in reversed(range(self.n_steps)):
            # Use the stored done flag for the *current* step to decide whether
            # to bootstrap from the next value.  Using dones[t] directly avoids
            # the off-by-one error that would arise from carrying next_done
            # across iterations.
            not_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * not_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * not_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

            next_value = self.values[t]

    # ------------------------------------------------------------------
    # Sequence batching for BPTT
    # ------------------------------------------------------------------

    def get_sequences(
        self,
        seq_len: int,
        device: torch.device,
        normalize_advantages: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Split the buffer into non-overlapping sequences for BPTT.

        Each returned batch is a dict with keys:

        * ``"tokens"``      — shape ``(n_seqs, seq_len, max_entities, token_dim)``
        * ``"pad_masks"``   — shape ``(n_seqs, seq_len, max_entities)``
        * ``"hx_h"``        — shape ``(lstm_num_layers, n_seqs, lstm_hidden_size)``
        * ``"hx_c"``        — shape ``(lstm_num_layers, n_seqs, lstm_hidden_size)``
        * ``"actions"``     — shape ``(n_seqs, seq_len, action_dim)``
        * ``"log_probs"``   — shape ``(n_seqs, seq_len)``
        * ``"advantages"``  — shape ``(n_seqs, seq_len)``
        * ``"returns"``     — shape ``(n_seqs, seq_len)``
        * ``"values"``      — shape ``(n_seqs, seq_len)``

        The initial hidden states ``hx_h`` / ``hx_c`` are taken from the
        *first* step of each sequence.

        Parameters
        ----------
        seq_len:
            Length of each sub-sequence.  Must evenly divide ``n_steps``.
        device:
            Destination device for returned tensors.
        normalize_advantages:
            Normalise advantages to zero mean, unit variance (recommended).

        Returns
        -------
        batches : list of dicts (each dict is one sequence batch)
        """
        if self.n_steps % seq_len != 0:
            raise ValueError(
                f"seq_len={seq_len} must evenly divide n_steps={self.n_steps}."
            )

        adv = self.advantages.copy()
        if normalize_advantages:
            adv_std = adv.std() + _NORMALIZATION_EPSILON
            adv = (adv - adv.mean()) / adv_std

        n_seqs = self.n_steps // seq_len
        batches: List[Dict[str, torch.Tensor]] = []

        for i in range(n_seqs):
            start = i * seq_len
            end = start + seq_len
            sl = slice(start, end)

            # Hidden states from the first step of this sequence
            hx_h = self.hx_h[start]  # (L, H)
            hx_c = self.hx_c[start]  # (L, H)

            batch = {
                "tokens": torch.as_tensor(
                    self.tokens[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len, N, token_dim)
                "pad_masks": torch.as_tensor(
                    self.pad_masks[sl], dtype=torch.bool, device=device
                ).unsqueeze(0),  # (1, seq_len, N)
                "hx_h": torch.as_tensor(
                    hx_h, dtype=torch.float32, device=device
                ).unsqueeze(1),  # (L, 1, H)
                "hx_c": torch.as_tensor(
                    hx_c, dtype=torch.float32, device=device
                ).unsqueeze(1),  # (L, 1, H)
                "actions": torch.as_tensor(
                    self.actions[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len, action_dim)
                "log_probs": torch.as_tensor(
                    self.log_probs[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len)
                "advantages": torch.as_tensor(
                    adv[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len)
                "returns": torch.as_tensor(
                    self.returns[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len)
                "values": torch.as_tensor(
                    self.values[sl], dtype=torch.float32, device=device
                ).unsqueeze(0),  # (1, seq_len)
            }
            batches.append(batch)

        return batches

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Approximate memory usage of stored arrays in bytes."""
        arrays = [
            self.tokens, self.pad_masks, self.actions,
            self.log_probs, self.rewards, self.dones, self.values,
            self.hx_h, self.hx_c, self.advantages, self.returns,
        ]
        return sum(a.nbytes for a in arrays)

    def __len__(self) -> int:
        return self._ptr
