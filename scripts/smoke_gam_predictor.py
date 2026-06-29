"""Lightweight sanity smoke for the modernized gam AR predictor.

Usage (inside a CUDA Python environment):
    PYTHONPATH=src:$PYTHONPATH \
    python scripts/smoke_gam_predictor.py

Checks performed:
    1. Build the predictor with default kwargs (no legacy keys).
    2. Count trainable parameters.
    3. bf16 forward at H=4 and H=16: shape/dtype/finite checks.
    4. flex_attention availability (informational).
    5. Causal-leak tests:
       a. Past invariance: modify inputs at step t, verify outputs at
          steps < t are bitwise identical (no future → past leak).
       b. Future sensitivity: modify inputs at step t, verify outputs at
          steps > t actually change (causal forward flow works).
       c. Within-step bidirectional: modify proprio at step t, verify
          that visual output at the SAME step t changes (intra-block is
          full attention within a step).
       d. Mask-backend parity: same input through flex_attention BlockMask
          and through the dense-mask fallback must produce matching
          outputs (≤ 5e-3 max abs diff in fp32). Only runs if flex is
          available.

The "past invariance" test is the canonical proof of no causal leak: if
modifying anything at step t changes outputs at step k < t, the mask is
broken.

This smoke exercises the predictor in isolation without DA3, dataset, or
DeepSpeed, so it is safe to run on a single GPU or even CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from robot.modeling.future_predictor import (
    GAMFuturePredictor,
    build_future_predictor,
    _HAS_FLEX,
)


def human(n: int) -> str:
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.1f}K"


def _build(device: torch.device, H_like_training: bool = True) -> GAMFuturePredictor:
    torch.manual_seed(0)
    predictor = build_future_predictor(
        cfg={
            "type": "gam",
            "d_da3": 1536,
            "d_model": 1024,
            "depth": 12,
            "num_heads": 16,
            "ffn_ratio": 4.0,
            "num_patches_per_view": 256,
            "num_register_tokens": 4,
            "proprio_dim": 7,
            "action_dim": 7,
            "action_chunk_size": 2,
            "use_language": True,
        }
    ).to(device)
    predictor.train()   # BN needs train mode to track stats
    return predictor


def _sample_inputs(
    predictor: GAMFuturePredictor,
    H: int,
    B: int = 1,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    V = 2
    return dict(
        past_visual_tokens=torch.randn(B, H, V, predictor.visual_tokens_per_view, predictor.d_da3, device=device),
        proprio_history=torch.randn(B, H, predictor.proprio_dim, device=device),
        past_action_history=torch.randn(B, H, predictor.action_chunk_size, predictor.action_dim, device=device),
        lang_feats=torch.randn(B, predictor.language_len, predictor.language_dim, device=device),
        lang_padding_mask=torch.ones(B, predictor.language_len, dtype=torch.bool, device=device),
    )


def _forward(predictor, inputs, dtype):
    device = next(predictor.parameters()).device
    with torch.amp.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=dtype != torch.float32,
    ):
        return predictor(
            past_visual_tokens=inputs["past_visual_tokens"],
            proprio=None,
            proprio_history=inputs["proprio_history"],
            past_action_history=inputs["past_action_history"],
            lang_feats=inputs["lang_feats"],
            lang_padding_mask=inputs["lang_padding_mask"],
        )


# --------------------------------------------------------------------------
# Test 1: shape / finite
# --------------------------------------------------------------------------

def test_shape_and_finite(device: torch.device, dtype: torch.dtype) -> None:
    predictor = _build(device)
    for H in (4, 16):
        inp = _sample_inputs(predictor, H=H, device=device)
        V = 2
        out = _forward(predictor, inp, dtype)
        tokens_per_view = predictor.visual_tokens_per_view
        assert out["predicted_next_visual_tokens"].shape == (1, H, V, tokens_per_view, predictor.d_da3)
        assert out["predicted_next_proprio"].shape == (1, H, predictor.proprio_dim)
        assert out["predicted_action_tokens"].shape == (1, H, V, predictor.d_da3)
        for k in (
            "predicted_next_visual_tokens",
            "predicted_next_proprio",
            "predicted_action_tokens",
        ):
            assert torch.isfinite(out[k]).all(), f"{k} contains NaN/Inf at H={H}"
        print(f"  [shape/finite]  H={H}  OK")


# --------------------------------------------------------------------------
# Test 2: causal leak: past invariance
# --------------------------------------------------------------------------

def _extract_step_outputs(out: dict, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (visual, proprio, action) at a given step t as detached cpu tensors."""
    return (
        out["predicted_next_visual_tokens"][:, step].detach().cpu(),
        out["predicted_next_proprio"][:, step].detach().cpu(),
        out["predicted_action_tokens"][:, step].detach().cpu(),
    )


def test_causal_past_invariance(device: torch.device) -> None:
    """Modify inputs at step t; verify outputs at every step < t are IDENTICAL.

    This is the definitive "no future-to-past leak" check. Any difference at
    step < t means the mask is broken.
    """
    # fp32 + eval() for deterministic comparison (BN in train would change
    # running stats between the two forwards). Disable autocast.
    predictor = _build(device).eval()
    H = 6
    V = 2

    base_inp = _sample_inputs(predictor, H=H, device=device, seed=7)

    # Pick t=3 so several past steps can be checked.
    t_perturb = 3

    # Build a modified input: bump visual / proprio / action at step t_perturb
    # by a large amount.
    mod_inp = {k: v.clone() for k, v in base_inp.items()}
    mod_inp["past_visual_tokens"][:, t_perturb] += 7.0
    mod_inp["proprio_history"][:, t_perturb] += 7.0
    mod_inp["past_action_history"][:, t_perturb] += 7.0

    with torch.no_grad():
        out_base = _forward(predictor, base_inp, torch.float32)
        out_mod = _forward(predictor, mod_inp, torch.float32)

    max_diffs = []
    for step in range(H):
        v_b, p_b, a_b = _extract_step_outputs(out_base, step)
        v_m, p_m, a_m = _extract_step_outputs(out_mod, step)
        v_diff = (v_b - v_m).abs().max().item()
        p_diff = (p_b - p_m).abs().max().item()
        a_diff = (a_b - a_m).abs().max().item()
        max_diffs.append((step, v_diff, p_diff, a_diff))

    # Past invariance assertion
    bad = []
    for step, v, p, a in max_diffs:
        if step < t_perturb:
            # Must be exactly zero (or within fp32 epsilon).
            if max(v, p, a) > 1e-5:
                bad.append((step, v, p, a))
    if bad:
        print("  [CAUSAL LEAK DETECTED] past steps differ after perturbing step", t_perturb)
        for step, v, p, a in bad:
            print(f"    step={step}  visual_diff={v:.3e}  proprio_diff={p:.3e}  action_diff={a:.3e}")
        raise AssertionError("Causal mask is leaking past → perturbed-step modified past outputs")

    # Future sensitivity: step >= t_perturb MUST change
    sensitive = [(s, v, p, a) for s, v, p, a in max_diffs if s >= t_perturb]
    unmoved = [e for e in sensitive if max(e[1], e[2], e[3]) < 1e-3]
    if unmoved:
        print("  [CAUSAL FORWARD BROKEN] perturbing step left future outputs unchanged")
        for step, v, p, a in unmoved:
            print(f"    step={step}  v={v:.3e}  p={p:.3e}  a={a:.3e}")
        raise AssertionError("Causal forward flow appears dead")

    print(
        f"  [past-invariance]  perturb@t={t_perturb}: past steps <{t_perturb} "
        f"max diff {max(max(v, p, a) for s, v, p, a in max_diffs if s < t_perturb):.3e} (≤ 1e-5 required)"
    )
    print(
        f"  [future-sensitivity]  steps >={t_perturb} min max-diff "
        f"{min(max(v, p, a) for s, v, p, a in max_diffs if s >= t_perturb):.3e} (> 1e-3 required)"
    )


# --------------------------------------------------------------------------
# Test 3: within-step bidirectional
# --------------------------------------------------------------------------

def test_within_step_bidirectional(device: torch.device) -> None:
    """Modify ONLY proprio at step t; verify visual output at SAME step t
    changes under bidirectional within-block attention.
    """
    predictor = _build(device).eval()
    H = 4
    base_inp = _sample_inputs(predictor, H=H, device=device, seed=11)
    t = 2
    mod_inp = {k: v.clone() for k, v in base_inp.items()}
    mod_inp["proprio_history"][:, t] += 5.0

    with torch.no_grad():
        out_base = _forward(predictor, base_inp, torch.float32)
        out_mod = _forward(predictor, mod_inp, torch.float32)

    v_b, _, _ = _extract_step_outputs(out_base, t)
    v_m, _, _ = _extract_step_outputs(out_mod, t)
    diff = (v_b - v_m).abs().max().item()
    if diff < 1e-4:
        raise AssertionError(
            f"Within-step bidirectional attention appears broken: perturbing proprio at "
            f"step {t} did not change visual output at step {t} (diff={diff:.3e})"
        )
    print(f"  [within-step bi-dir]  visual@t diff after proprio@t perturb = {diff:.3e} (> 1e-4 required)")


# --------------------------------------------------------------------------
# Test 4: mask backend parity (flex vs dense)
# --------------------------------------------------------------------------

def test_mask_backend_parity(device: torch.device) -> None:
    """Run the same input through flex_attention and dense-mask backends, then
    verify output parity. Only runs if flex is available.
    """
    if not _HAS_FLEX or device.type != "cuda":
        print("  [mask-parity]  SKIP (flex unavailable or cpu)")
        return

    predictor = _build(device).eval()
    H = 4
    inp = _sample_inputs(predictor, H=H, device=device, seed=23)

    with torch.no_grad():
        # Normal path uses flex if available
        out_flex = _forward(predictor, inp, torch.float32)

        # Force dense by disabling flex for this run
        original_get = predictor._get_flex_block_mask
        predictor._get_flex_block_mask = lambda H, V, device: None  # type: ignore
        try:
            out_dense = _forward(predictor, inp, torch.float32)
        finally:
            predictor._get_flex_block_mask = original_get  # type: ignore

    max_diff = 0.0
    for key in ("predicted_next_visual_tokens", "predicted_next_proprio", "predicted_action_tokens"):
        d = (out_flex[key] - out_dense[key]).abs().max().item()
        max_diff = max(max_diff, d)
    if max_diff > 5e-3:
        raise AssertionError(
            f"flex_attention and dense-mask backends diverge: max diff {max_diff:.3e} > 5e-3 tolerance"
        )
    print(f"  [mask-parity]  flex vs dense max diff = {max_diff:.3e} (≤ 5e-3 required)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}  autocast_dtype={dtype}  flex_attention_available={_HAS_FLEX}")
    pred_for_count = _build(device)
    n_params = sum(p.numel() for p in pred_for_count.parameters())
    print(f"predictor params: {human(n_params)}")

    test_shape_and_finite(device, dtype)
    test_causal_past_invariance(device)
    test_within_step_bidirectional(device)
    test_mask_backend_parity(device)

    print("smoke_gam_predictor: ALL TESTS PASSED")


if __name__ == "__main__":
    main()
