"""Unified multi-task MoE actor-critic for ManiSkill PPO.

Architecture (mirrored for actor and critic):

     obs_i (obs_dim_i)  -- padded to max_obs_dim
         │
         ▼
     (zero-pad to max_obs_dim) → shared Gate + Experts
         │
         ▼  experts each contain an input projection (max_obs_dim → 256)
         ├── Gate(h) ──► softmax weights (1, K)  (shared by actor & critic)
         │
         ├── Expert_0(obs) ─► o_0          experts share hidden dim 256;
         ├── Expert_1(obs) ─► o_1          trunk = Linear(256→512→512→256→out)
         ├── Expert_2(obs) ─► o_2          can be warm-started from per-task ckpts
         └── Expert_3(obs) ─► o_3          (drop first layer, copy layers 2..5)
                        │
                        ▼
                Σ w_i · o_i  ──► action_mean (8) / V(s) (1)

actor_logstd is a global (1, 8) Parameter, not per-task — keeps PPO simple.

A small load-balancing loss is exposed (see `last_gate_stats`) so the trainer
can encourage even expert utilisation across the batch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


HIDDEN = 256
EXPERT_TRUNK = (256, 512, 512, 256)
ACTION_OUT_STD = 0.01 * np.sqrt(2)
NUM_EXPERTS_DEFAULT = 4
GATE_HIDDEN = 256


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Gate(nn.Module):
    """Shared gate network producing soft weights over experts.

    Now accepts the padded observation directly as input (shape: (B, input_dim)).
    """

    def __init__(self, input_dim: int = HIDDEN, gate_hidden: int = GATE_HIDDEN, num_experts: int = NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, gate_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(gate_hidden, num_experts), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        w = F.softmax(logits, dim=-1)
        return w, logits


class Expert(nn.Module):
    """One expert's trunk: input projection (max_obs_dim -> HIDDEN) + 3 hidden Linears + output.

    The per-task projection previously lived outside the expert. We merge it
    into the expert as `in_proj`. Inputs smaller than the configured
    `input_dim` should be padded externally (see MultiTaskAgent).
    """

    def __init__(self, input_dim: int, out_dim: int, out_std: float = np.sqrt(2)):
        super().__init__()
        # input projection: max_obs_dim -> EXPERT_TRUNK[0] (HIDDEN)
        self.in_proj = layer_init(nn.Linear(input_dim, EXPERT_TRUNK[0]))
        self.h1 = layer_init(nn.Linear(EXPERT_TRUNK[0], EXPERT_TRUNK[1]))
        self.h2 = layer_init(nn.Linear(EXPERT_TRUNK[1], EXPERT_TRUNK[2]))
        self.h3 = layer_init(nn.Linear(EXPERT_TRUNK[2], EXPERT_TRUNK[3]))
        self.out = layer_init(nn.Linear(EXPERT_TRUNK[3], out_dim), std=out_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, input_dim)
        h = torch.tanh(self.in_proj(obs))
        h = torch.tanh(self.h1(h))
        h = torch.tanh(self.h2(h))
        h = torch.tanh(self.h3(h))
        return self.out(h)


class MoEHead(nn.Module):
    """Bank of N experts + optional external/shared gate.

    If `gate_module` is provided we use that for gate logits/weights. Otherwise
    we build an internal gate identical to the previous implementation.
    """

    def __init__(self, input_dim: int, out_dim: int, num_experts: int = NUM_EXPERTS_DEFAULT,
                 gate_module: Optional[nn.Module] = None, gate_hidden: int = GATE_HIDDEN,
                 expert_out_std: float = np.sqrt(2)):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_dim, out_dim, out_std=expert_out_std) for _ in range(num_experts)])
        if gate_module is None:
            self.gate = Gate(input_dim=input_dim, gate_hidden=gate_hidden, num_experts=num_experts)
        else:
            self.gate = gate_module

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: (B, input_dim)
        # get per-expert outputs
        expert_outs = torch.stack([e(obs) for e in self.experts], dim=1)  # (B, K, out)
        # gate should take the padded observation directly as input
        gate_w, gate_logits = self.gate(obs)  # (B, K), (B, K)
        out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)           # (B, out)
        return out, gate_w, gate_logits


class MultiTaskAgent(nn.Module):
    """MoE actor-critic for N tasks.

    obs_dims      : list of per-task obs dims (one Proj per task)
    action_dim    : Panda pd_joint_delta_pos = 8 for all four ManiSkill tasks
    num_experts   : default 4 (== number of tasks)
    """

    def __init__(self, obs_dims: Sequence[int], action_dim: int,
                 num_experts: int = NUM_EXPERTS_DEFAULT):
        super().__init__()
        self.num_tasks = len(obs_dims)
        self.num_experts = num_experts
        self.action_dim = action_dim

        # unify observation dimension to the maximum across tasks and pad
        # smaller observations with zeros at runtime
        self.input_dim = int(max(obs_dims))
        # shared gate used by both actor and critic (takes padded obs as input)
        self.gate = Gate(input_dim=self.input_dim, gate_hidden=GATE_HIDDEN, num_experts=num_experts)
        # MoE heads now accept raw (padded) obs and each expert contains its own
        # input projection (obs -> hidden). Actor and critic share the same gate.
        self.actor_moe = MoEHead(input_dim=self.input_dim, out_dim=action_dim,
                     num_experts=num_experts, gate_module=self.gate,
                     expert_out_std=ACTION_OUT_STD)
        self.critic_moe = MoEHead(input_dim=self.input_dim, out_dim=1,
                      num_experts=num_experts, gate_module=self.gate,
                      expert_out_std=1.0)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -1.5)

        # Last forward's gate weights, for load-balancing logging/loss.
        self._last_actor_gate: Optional[torch.Tensor] = None
        self._last_critic_gate: Optional[torch.Tensor] = None
        # also keep logits for optional task supervision
        self._last_actor_gate_logits: Optional[torch.Tensor] = None
        self._last_critic_gate_logits: Optional[torch.Tensor] = None

    # ---- single-task forward ----
    # Always operates on a batch from ONE task. The PPO loop iterates over
    # tasks; the update loop also slices minibatches per task. This keeps the
    # implementation simple and fast (no per-sample dispatch).

    def _pad_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim) where obs_dim <= self.input_dim
        if obs.shape[-1] == self.input_dim:
            return obs
        # pad zeros on the right
        pad = obs.new_zeros((*obs.shape[:-1], self.input_dim - obs.shape[-1]))
        return torch.cat([obs, pad], dim=-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._pad_obs(obs)
        v, gate, gate_logits = self.critic_moe(x)
        self._last_critic_gate = gate
        self._last_critic_gate_logits = gate_logits
        return v

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        x = self._pad_obs(obs)
        mean, gate, gate_logits = self.actor_moe(x)
        self._last_actor_gate = gate
        self._last_actor_gate_logits = gate_logits
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        return Normal(mean, std).sample()

    def get_action_and_value(self, obs: torch.Tensor,
                             action: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._pad_obs(obs)
        mean, gate_a, gate_a_logits = self.actor_moe(x)
        value, gate_c, gate_c_logits = self.critic_moe(x)
        self._last_actor_gate = gate_a
        self._last_critic_gate = gate_c
        self._last_actor_gate_logits = gate_a_logits
        self._last_critic_gate_logits = gate_c_logits
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logp, entropy, value

    # ---- diagnostics ----
    def gate_load_balance_loss(self) -> torch.Tensor:
        """Switch-Transformer-style balance loss on the most recent forward.

        Encourages avg gate distribution to be uniform AND each gate to be
        somewhat peaky. Sum of:
          - KL(mean_gate || uniform): pushes avg utilisation toward 1/K
          - -mean(entropy(per-sample gate)): pushes each gate to specialise
        Scaled mildly; the trainer can multiply by a small coefficient.
        """
        gates = [g for g in (self._last_actor_gate, self._last_critic_gate) if g is not None]
        if not gates:
            return torch.zeros((), device=next(self.parameters()).device)
        loss = 0.0
        for g in gates:
            mean_g = g.mean(dim=0)                                          # (K,)
            uniform = torch.full_like(mean_g, 1.0 / g.shape[-1])
            kl_to_uniform = (mean_g * (mean_g.clamp_min(1e-9).log() - uniform.log())).sum()
            per_sample_entropy = -(g * g.clamp_min(1e-9).log()).sum(dim=-1).mean()
            loss = loss + kl_to_uniform - 0.1 * per_sample_entropy
        return loss


# ---------- checkpoint warm-start ----------

def _load_state_dict_lenient(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _copy_linear(target: nn.Linear, src_sd: Dict[str, torch.Tensor],
                 src_prefix: str, msgs: list) -> bool:
    w_key = f"{src_prefix}.weight"
    b_key = f"{src_prefix}.bias"
    if w_key not in src_sd or b_key not in src_sd:
        msgs.append(f"  missing {w_key} / {b_key}")
        return False
    sw, sb = src_sd[w_key], src_sd[b_key]
    # allow copying into a larger target by zero-padding the src on the
    # right (useful when warm-starting input projections into a larger
    # unified input_dim). Exact matches are copied directly.
    tw_shape = tuple(target.weight.shape)
    tb_shape = tuple(target.bias.shape)
    if sw.shape == target.weight.shape and sb.shape == target.bias.shape:
        with torch.no_grad():
            target.weight.copy_(sw)
            target.bias.copy_(sb)
        return True
    # allow src with smaller in_features (sw: out x in_src) to be copied into
    # target (out x in_tgt) when out matches. Pad remaining columns with 0.
    if sw.ndim == 2 and sw.shape[0] == target.weight.shape[0] and sw.shape[1] < target.weight.shape[1] and sb.shape == target.bias.shape:
        neww = torch.zeros_like(target.weight)
        neww[:, :sw.shape[1]] = sw
        with torch.no_grad():
            target.weight.copy_(neww)
            target.bias.copy_(sb)
        return True
    msgs.append(f"  shape mismatch at {src_prefix}: src {tuple(sw.shape)} vs dst {tw_shape}")
    return False
    return True


def init_experts_from_per_task_ckpts(agent: MultiTaskAgent,
                                     ckpt_paths: Sequence[Optional[str]],
                                     device: str = "cpu") -> None:
    """Warm-start each task's expert trunks from the corresponding per-task PPO ckpt.

    The single-task `Agent` in ppo.py has actor_mean and critic as
    nn.Sequential with this layer order:

        0:  Linear(obs_dim, 256)
        1:  Tanh
        2:  Linear(256, 512)        ─┐
        3:  Tanh                     │
        4:  Linear(512, 512)         │
        5:  Tanh                     │
        6:  Linear(512, 256)         │  ──► Expert_i.{h1,h2,h3,out}
        7:  Tanh                     │
        8:  Linear(256, out_dim)    ─┘
        (no tanh on output)

    We copy 0->proj, 2→h1, 4→h2, 6→h3, 8→out. Mismatches are logged and skipped.
    Pass None at index i to skip that task.
    """
    for tid, path in enumerate(ckpt_paths):
        if path is None or path == "":
            print(f"[init_experts] task {tid}: no ckpt provided, leaving random init")
            continue
        try:
            sd = _load_state_dict_lenient(path, device=device)
        except Exception as e:
            print(f"[init_experts] task {tid}: failed to load {path}: {e}")
            continue

        msgs: list = []
        n_loaded = 0
        # copy input projection (layer 0 of single-task Sequential) -> Expert.in_proj
        a_proj = agent.actor_moe.experts[tid].in_proj
        if _copy_linear(a_proj, sd, "actor_mean.0", msgs):
            n_loaded += 1
        c_proj = agent.critic_moe.experts[tid].in_proj
        if _copy_linear(c_proj, sd, "critic.0", msgs):
            n_loaded += 1

        # actor
        a_expert = agent.actor_moe.experts[tid]
        for dst_layer, src_idx in [(a_expert.h1, 2), (a_expert.h2, 4),
                                    (a_expert.h3, 6), (a_expert.out, 8)]:
            if _copy_linear(dst_layer, sd, f"actor_mean.{src_idx}", msgs):
                n_loaded += 1
        # critic
        c_expert = agent.critic_moe.experts[tid]
        for dst_layer, src_idx in [(c_expert.h1, 2), (c_expert.h2, 4),
                                    (c_expert.h3, 6), (c_expert.out, 8)]:
            if _copy_linear(dst_layer, sd, f"critic.{src_idx}", msgs):
                n_loaded += 1

        print(f"[init_experts] task {tid}: loaded {n_loaded}/10 trunk linears from {path}")
        for m in msgs:
            print(m)


# ---------- quick sanity check ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    obs_dims = [35, 42, 48, 43]
    act_dim = 8
    agent = MultiTaskAgent(obs_dims, act_dim)
    for tid, d in enumerate(obs_dims):
        x = torch.randn(7, d)
        a, lp, ent, v = agent.get_action_and_value(x, tid)
        assert a.shape == (7, act_dim), a.shape
        assert lp.shape == (7,), lp.shape
        assert v.shape == (7, 1), v.shape
        print(f"task {tid}: ok  act={tuple(a.shape)}  v={tuple(v.shape)}  gate={tuple(agent._last_actor_gate.shape)}")
    bl = agent.gate_load_balance_loss()
    print("balance loss:", float(bl))
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"#params: {n_params/1e6:.2f}M")
