"""Exact checkpoint q01/q99 scaling, without importing training datasets."""
import torch


class Normalizer:
    def __init__(self, state):
        if state.get("norm_mode") != "q01_q99":
            raise ValueError("This mobile release requires checkpoint q01_q99 statistics")
        self.eps = float(state["eps"])
        if not 0 < self.eps < 1e-3:
            raise ValueError("Invalid normalization epsilon")
        self.stats = {}
        for key, row in state["stats_by_key"].items():
            q01, q99 = [torch.as_tensor(row[k], dtype=torch.float32) for k in ("q01", "q99")]
            mask = torch.as_tensor(row["mask"], dtype=torch.bool)
            if any(x.shape != (16,) for x in (q01, q99, mask)) or not torch.isfinite(q01).all() or not torch.isfinite(q99).all():
                raise ValueError(f"Invalid 16D statistics for {key}")
            if (q99 < q01).any():
                raise ValueError("q99 must not be less than q01")
            self.stats[key] = (q01, q99 - q01 + self.eps, mask & ((q99 - q01).abs() >= 1e-6))
        if not self.stats:
            raise ValueError("Empty normalization statistics")

    def transform(self, x, key, *, inverse=False):
        if key not in self.stats:
            raise KeyError(f"Unknown statistics key {key!r}; available: {sorted(self.stats)}")
        q01, scale, mask = [v.to(x.device) for v in self.stats[key]]
        transformed = (x + 1.) / 2. * scale + q01 if inverse else 2. * (x - q01) / scale - 1.
        return torch.where(mask, transformed, x)
