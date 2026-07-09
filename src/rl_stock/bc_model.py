"""Behavior Cloning (BC) architecture using a causal Transformer without RTG conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class BehaviorCloning(nn.Module):
    """Behavior Cloning model for sequence modeling of RL trajectories.

    Predicts actions given a sequence of (State, Action) tokens (no Return-to-Go).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        context_length: int = 20,
        dropout: float = 0.1,
        max_episode_len: int = 2048,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.context_length = context_length

        # Embedding layers (no embed_rtg)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(action_dim, d_model)
        self.embed_timestep = nn.Embedding(max_episode_len, d_model)

        # Layer normalization for embeddings
        self.embed_ln = nn.LayerNorm(d_model)

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Action prediction head (predicts a_t from s_t output representation)
        self.predict_action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),  # continuous target is in [-1, 1]
        )

    def forward(
        self,
        states: torch.Tensor,     # shape: (batch, K, state_dim)
        actions: torch.Tensor,    # shape: (batch, K, action_dim)
        timesteps: torch.Tensor,  # shape: (batch, K)
        attention_mask: torch.Tensor | None = None,  # shape: (batch, K), 1 = active, 0 = padded
    ) -> torch.Tensor:
        """Forward pass predicting action sequence.

        Returns predicted actions of shape (batch, K, action_dim).
        """
        batch_size, seq_len, _ = states.shape

        # Compute embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        # Positional/timestep embedding
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # Interleave tokens: (s_1, a_1, s_2, a_2, ...)
        # We construct a tensor of shape (batch, 2 * K, d_model)
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings),
            dim=2,
        ).reshape(batch_size, 2 * seq_len, self.d_model)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Build combined causal + padding mask (float, additive)
        device = states.device
        total_len = 2 * seq_len

        # Start with a causal mask (upper-triangular -inf)
        attn_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=device),
            diagonal=1,
        )

        if attention_mask is not None:
            # Expand (batch, K) → (batch, 2K) boolean: True = active
            stacked_active = torch.stack(
                (attention_mask, attention_mask),
                dim=2,
            ).reshape(batch_size, total_len).bool()

            # Column mask: block padded keys. Shape (batch, 1, total_len)
            col_mask = torch.where(
                stacked_active.unsqueeze(1),  # (B, 1, T)
                torch.tensor(0.0, device=device),
                torch.tensor(float("-inf"), device=device),
            )  # (B, 1, T)

            # Combine: causal_mask (T,T) + col_mask (B,1,T) → (B,T,T)
            combined = attn_mask.unsqueeze(0) + col_mask  # (B, T, T)

            # Fix: ensure padded query positions can attend to themselves (diagonal = 0)
            padded_rows = ~stacked_active  # (B, T), True for padded
            diag_idx = torch.arange(total_len, device=device)
            combined[:, diag_idx, diag_idx] = torch.where(
                padded_rows,
                torch.tensor(0.0, device=device),
                combined[:, diag_idx, diag_idx],
            )

            # Expand for multi-head: (B*nH, T, T)
            n_heads = self.transformer.layers[0].self_attn.num_heads
            attn_mask = combined.repeat_interleave(n_heads, dim=0)  # (B*nH, T, T)

        # Transformer forward pass
        transformer_outputs = self.transformer(
            src=stacked_inputs,
            mask=attn_mask,
            is_causal=False,
        )

        # Extract outputs corresponding to state tokens to predict action
        # In the sequence: s_1, a_1, s_2, a_2, ...
        # States are at indices 0, 2, 4, ... which is 2 * i
        state_outputs = transformer_outputs[:, 0::2, :]

        # Predict actions
        pred_actions = self.predict_action_head(state_outputs)
        return pred_actions
