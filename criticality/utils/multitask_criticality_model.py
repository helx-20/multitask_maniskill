"""Multi-task MoE classifier for criticality (obs + force -> P(critical)).

Mirrors the policy MoE in examples/baselines/ppo/multitask_agent.py:

    (obs_i + force_i)
            │  shape: (B, obs_dim_i + force_dim_i)
            ▼
        Proj_i  (Linear -> 256, ReLU)        per-task
            │
            ▼  h ∈ R^256
            ├── Gate(h) ─► softmax over 4 experts
            ├── Expert_k(h) ─► o_k ∈ R^2     k = 0..3
            ▼
        Σ w_k · o_k  ──► logits (B, 2)

`SimpleClassifier` (existing in criticality_model.py) is kept untouched as a
fallback for any single-task script that hasn't been migrated.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN = 256
EXPERT_HIDDEN = 256
NUM_EXPERTS_DEFAULT = 4
GATE_HIDDEN = 64


class TaskInputProj(nn.Module):
    """Per-task Linear(obs_dim_i + force_dim_i -> HIDDEN) + ReLU."""

    def __init__(self, input_dims: Sequence[int], hidden: int = HIDDEN):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, hidden) for d in input_dims])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return F.relu(self.projs[task_id](x))


class CritExpert(nn.Module):
    """One classifier-expert trunk: Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> num_classes).

    Same depth as the single-task `SimpleClassifier` with `hidden_layer=1`.
    """

    def __init__(self, num_classes: int = 2,
                 hidden_in: int = HIDDEN, hidden_mid: int = EXPERT_HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(hidden_in, hidden_mid)
        self.fc2 = nn.Linear(hidden_mid, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(h)))


class CritMoEHead(nn.Module):
    def __init__(self, num_classes: int = 2, num_experts: int = NUM_EXPERTS_DEFAULT,
                 hidden: int = HIDDEN, gate_hidden: int = GATE_HIDDEN):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([CritExpert(num_classes) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(hidden, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, num_experts),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(h)
        gate_w = F.softmax(gate_logits, dim=-1)                              # (B, K)
        expert_outs = torch.stack([e(h) for e in self.experts], dim=1)       # (B, K, C)
        out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)                # (B, C)
        return out, gate_w


class MultiTaskClassifier(nn.Module):
    """MoE binary classifier over (obs+force) for N tasks.

    obs_dims    : list of state obs dims per task    (e.g. [35, 42, 48, 43])
    force_dims  : list of force vector dims per task (e.g. [2, 3, 3, 3])
    The per-task input projection accepts obs_dim_i + force_dim_i features.
    """

    def __init__(self, obs_dims: Sequence[int], force_dims: Sequence[int],
                 num_classes: int = 2, num_experts: int = NUM_EXPERTS_DEFAULT):
        super().__init__()
        assert len(obs_dims) == len(force_dims), "obs_dims / force_dims length mismatch"
        self.num_tasks = len(obs_dims)
        self.input_dims = [o + f for o, f in zip(obs_dims, force_dims)]
        self.proj = TaskInputProj(self.input_dims)
        self.head = CritMoEHead(num_classes=num_classes, num_experts=num_experts)
        self._last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """x: (B, obs_dim_i + force_dim_i). Returns logits (B, num_classes)."""
        h = self.proj(x, task_id)
        logits, gate = self.head(h)
        self._last_gate = gate
        return logits

    def forward_mixed(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Forward a batch whose rows belong to mixed task_ids.

        x         : (B, max_input_dim) — caller pads or splits.
        task_ids  : (B,) long tensor
        This default impl loops per unique task_id (typically <= 4 distinct).
        """
        out = torch.empty((x.shape[0], 2), device=x.device, dtype=x.dtype)
        for tid in torch.unique(task_ids).tolist():
            tid = int(tid)
            mask = (task_ids == tid)
            xi = x[mask, :self.input_dims[tid]]
            out[mask] = self.forward(xi, tid)
        return out

    # ---- diagnostics ----
    def gate_load_balance_loss(self) -> torch.Tensor:
        if self._last_gate is None:
            return torch.zeros((), device=next(self.parameters()).device)
        g = self._last_gate
        mean_g = g.mean(dim=0)
        uniform = torch.full_like(mean_g, 1.0 / g.shape[-1])
        kl_to_uniform = (mean_g * (mean_g.clamp_min(1e-9).log() - uniform.log())).sum()
        per_sample_entropy = -(g * g.clamp_min(1e-9).log()).sum(dim=-1).mean()
        return kl_to_uniform - 0.1 * per_sample_entropy


# ---------- quick sanity check ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    obs_dims = [35, 42, 48, 43]
    force_dims = [2, 3, 3, 3]
    model = MultiTaskClassifier(obs_dims, force_dims)
    for tid, (o, f) in enumerate(zip(obs_dims, force_dims)):
        x = torch.randn(7, o + f)
        logits = model(x, tid)
        assert logits.shape == (7, 2), logits.shape
        print(f"task {tid}: ok  logits={tuple(logits.shape)}  "
              f"gate={tuple(model._last_gate.shape)}")
    print("balance loss:", float(model.gate_load_balance_loss()))
    print(f"#params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
