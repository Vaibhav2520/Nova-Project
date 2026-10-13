"""NovaState: the working memory of the operator (the current rule and what happened while refining it)."""

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class NovaState:
    z: torch.Tensor
    t: torch.Tensor
    halted: torch.Tensor
    rho_hist: torch.Tensor
    w: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None

    @classmethod
    def init(cls, batch: int, d: int, t_max: int, device=None, dtype=torch.float32) -> "NovaState":
        """Create a fresh state: zero rules, zero steps, nothing halted, no errors measured yet."""
        return cls(
            z=torch.zeros(batch, d, device=device, dtype=dtype),
            t=torch.zeros(batch, dtype=torch.long, device=device),
            halted=torch.zeros(batch, dtype=torch.bool, device=device),
            rho_hist=torch.full((batch, t_max + 1), -1.0, device=device),
        )

    def reset(self) -> None:
        """Put the state back to its starting values without making new tensors."""
        self.z = torch.zeros_like(self.z)
        self.t = torch.zeros_like(self.t)
        self.halted = torch.zeros_like(self.halted)
        self.rho_hist = torch.full_like(self.rho_hist, -1.0)
        self.w, self.v, self.R = None, None, None

    def detach(self) -> "NovaState":
        """Return a copy cut off from the gradient graph. For inspection tools only, never inside training."""
        cut = lambda x: None if x is None else x.detach()
        return NovaState(cut(self.z), cut(self.t), cut(self.halted), cut(self.rho_hist), cut(self.w), cut(self.v), cut(self.R))

    def clone(self) -> "NovaState":
        """Return an independent copy whose tensors do not share memory with this one."""
        cp = lambda x: None if x is None else x.clone()
        return NovaState(cp(self.z), cp(self.t), cp(self.halted), cp(self.rho_hist), cp(self.w), cp(self.v), cp(self.R))

    def to(self, device) -> "NovaState":
        """Return a copy moved to another device."""
        mv = lambda x: None if x is None else x.to(device)
        return NovaState(mv(self.z), mv(self.t), mv(self.halted), mv(self.rho_hist), mv(self.w), mv(self.v), mv(self.R))

    def state_dict(self) -> dict:
        """Return a dict of tensors that torch.save can store."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    @classmethod
    def load_state_dict(cls, data: dict) -> "NovaState":
        """Rebuild a state from the dict produced by state_dict()."""
        required = {"z", "t", "halted", "rho_hist"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"NovaState is missing fields: {sorted(missing)}")
        return cls(**{k: data.get(k) for k in ("z", "t", "halted", "rho_hist", "w", "v", "R")})
