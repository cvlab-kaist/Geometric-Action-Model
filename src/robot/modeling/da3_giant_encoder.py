"""DA3 encoder wrapper with optional action-token fine-tuning support."""

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# flex_attention availability for memory-cheap block-causal masking on the
# DA3 deep stack's global attention path. SDPA with a custom additive mask
# falls back to math/mem-efficient backends that materialize the full
# [B, H, L, S] attention score tensor, which OOMs at H=8 V=2 K=258 (seq=4128
# per timestep block). FlexAttention uses a sparse BlockMask + a per-block
# kernel that never materializes the dense score matrix.
#
# CRITICAL: flex_attention must be torch.compile()'d to use the fused kernel.
# Eager calls materialize full scores and OOM at production scale: verified at
# L=4128, H=24, bf16 with eager peak 5.39 GB vs compiled 0.07 GB per call.
# Model-level torch.compile around the encoder is insufficient because graph
# breaks in our deep-stack loop drop the flex_attention call back to eager. We
# wrap the imported function at module load.
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention_raw,
        create_block_mask as _create_block_mask,
    )
    _flex_attention = torch.compile(_flex_attention_raw)
    _HAS_FLEX_ATTENTION = True
except Exception:
    _flex_attention = None
    _create_block_mask = None
    _HAS_FLEX_ATTENTION = False


def _make_deep_block_causal_mask_mod(token_count: int, v_count: int):
    """Return a flex_attention mask_mod closure that allows a query at flat
    seq index q to attend to a key at flat seq index k iff the key's timestep
    is <= the query's timestep.

    Layout assumption (matches `_propagate_shallow_with_actions_impl` line
    1293 reshape from (B, steps, V, token_count, D) to (B, total_view,
    token_count, D), where total_view = steps * V): flat seq index i maps to
    (view_slot=i//token_count, token_in_view=i%token_count) and view_slot
    s = t * V + v, so timestep(i) = (i // token_count) // V.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_t = (q_idx // token_count) // v_count
        k_t = (kv_idx // token_count) // v_count
        return q_t >= k_t

    return mask_mod


_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DA3_SRC_DIR = os.path.join(_ROOT_DIR, "Depth-Anything-3", "src")
if os.path.isdir(_DA3_SRC_DIR) and _DA3_SRC_DIR not in sys.path:
    sys.path.insert(0, _DA3_SRC_DIR)


def _cuda_profile_mark(profile: Optional[Dict[str, object]], name: str) -> None:
    if profile is None or not torch.cuda.is_available():
        return
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    profile.setdefault("_cuda_marks", []).append((name, event))


class _StubCallable:
    """Dummy object that swallows attribute access / calls / indexing.

    Used to satisfy `from pkg import Name` style imports from DA3's optional
    side deps without pulling in the real package. Any attempt to actually
    invoke the stub raises, so runtime crashes stay loud.
    """

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"<DA3-stub {self._name}>"

    def __getattr__(self, item: str):
        return _StubCallable(f"{self._name}.{item}")

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._name} is a stub: the real module is missing in "
            "this environment. Stage 2 frozen-encoder inference skips it."
        )

    def __getitem__(self, key):
        return _StubCallable(f"{self._name}[{key!r}]")


class _StubModule:
    """Minimal module-like object that hands out _StubCallable for any attr."""

    def __init__(self, name: str):
        self.__name__ = name
        self.__path__ = []  # marks it as a package, lets submodules resolve

    def __getattr__(self, item: str):
        return _StubCallable(f"{self.__name__}.{item}")


class _AttrDict(dict):
    """Minimal addict.Dict replacement: dict with attribute access.

    DA3's DPT head (`depth_anything_3.model.dualdpt`) wraps its output in
    `addict.Dict(out_dict)` at the very end of `forward`. The wrapper only
    needs dict-style lookup and attribute-style lookup, which a dict
    subclass with `__getattr__` covers. Without this shim, any call to
    `dpt_head(...)` during monitoring crashes because the previous stub
    raised on `__call__`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        # DA3 forward calls `del output.ray` / `del output.ray_conf` during
        # `_process_camera_estimation`. Route attribute deletion to dict key
        # deletion so the stub matches real addict.Dict semantics.
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __contains__(self, key):  # keep dict semantics explicit for `"ray" in output`
        return super().__contains__(key)


class _AddictStubModule(_StubModule):
    """Special-cased addict stub: exposes a real dict-like `Dict`.

    Everything else on the module still falls through to `_StubCallable`,
    so unrelated addict symbols keep the loud-crash behavior.
    """

    def __getattr__(self, item: str):
        if item == "Dict":
            return _AttrDict
        return _StubCallable(f"{self.__name__}.{item}")


def _install_da3_optional_stubs() -> None:
    stub_names = (
        "moviepy",
        "moviepy.editor",
        "pycolmap",
        "trimesh",
        "imageio",
        "imageio.v2",
        "imageio.v3",
        "evo",
        "evo.core",
        "evo.core.trajectory",
        "plyfile",
    )
    import importlib

    def _try_import(name: str) -> bool:
        try:
            importlib.import_module(name)
            return True
        except Exception:
            return False

    for name in stub_names:
        if name in sys.modules:
            continue
        if _try_import(name):
            # Real package is installed; leave it alone.
            continue
        sys.modules[name] = _StubModule(name)
    if "addict" not in sys.modules:
        if _try_import("addict"):
            pass  # real addict wins
        else:
            sys.modules["addict"] = _AddictStubModule("addict")


class DA3GiantEncoder(nn.Module):
    """DA3 encoder with raw multi-level feature extraction.

    The class name is kept for checkpoint/import compatibility; `model_name`
    selects the actual DA3 preset and defaults to `da3-giant`.
    """

    OUT_LAYERS = [19, 27, 33, 39]
    PATCH_SIZE = 14

    def __init__(
        self,
        ckpt_path: str = os.path.join(
            os.environ.get("DA3_ROOT", "."), "checkpoints/track4world_da3.pth"
        ),
        model_name: str = "da3-giant",
        encoder_input_size: int = 224,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        freeze_backbone: bool = True,
        n_action_steps: int = 0,
        views_per_timestep: int = 2,
        action_steps_per_token: int = 1,
        use_temporal_embed: bool = False,
        action_input_rate: float = 0.4,
        action_only_frame_attn: bool = False,
    ):
        super().__init__()

        self.model_name = str(model_name or "da3-giant")
        self.patch_size = self.PATCH_SIZE
        self.encoder_input_size = encoder_input_size
        self.eps = eps
        self.n_action_steps = int(n_action_steps)
        self.views_per_timestep = int(views_per_timestep)
        self.action_steps_per_token = int(action_steps_per_token)
        # VGA-style (arXiv:2604.12908): at frame-wise local attention layers,
        # action tokens attend only among themselves while image tokens do
        # per-view local attention. Global attention merges the two streams.
        # Weights are SHARED with the baseline DA3 block. Only the token
        # grouping into the same block call is changed. Default False keeps
        # current behavior bit-identical.
        self.action_only_frame_attn = bool(action_only_frame_attn)

        h_patches = encoder_input_size // self.PATCH_SIZE
        w_patches = encoder_input_size // self.PATCH_SIZE
        self.num_patches = h_patches * w_patches
        self.h_patches = h_patches
        self.w_patches = w_patches

        self.da3_model = self._load_full_model(ckpt_path, model_name=self.model_name)
        self.backbone = self.da3_model.model.backbone
        self.dpt_head = self.da3_model.model.head
        # Cache for flex_attention BlockMask used by the deep-stack global
        # attention path when predictor.deep_temporal_causal_mask=true.
        # Keyed by (steps, v_count, token_count, device).
        self._deep_flex_block_mask_cache: Dict[tuple, object] = {}

        trans = self.backbone.pretrained
        self.embed_dim = int(trans.embed_dim)
        self.hidden_size = self.embed_dim * 2 if trans.cat_token else self.embed_dim
        self.out_layers = [int(x) for x in getattr(self.backbone, "out_layers", self.OUT_LAYERS)]
        if len(self.out_layers) != len(self.OUT_LAYERS):
            raise ValueError(
                f"{self.model_name} exposes {len(self.out_layers)} DA3 out layers; "
                f"expected {len(self.OUT_LAYERS)} for DPT decode compatibility."
            )
        self.shallow_target_layer = int(getattr(trans, "alt_start", -1)) - 1
        self.num_register_tokens = int(getattr(trans, "num_register_tokens", 0))

        self.register_buffer(
            "encoder_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "encoder_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

        self.use_temporal_embed = use_temporal_embed
        self.action_input_rate = float(action_input_rate)
        max_views = 8  # supports up to 8 cameras
        max_timesteps = 32  # supports up to 32 timesteps
        if self.n_action_steps > 0:
            # Shared base token (1, 1, D)
            self.action_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            # Per-timestep embedding (1, T_max, D), sinusoidal init, learnable
            self.action_timestep_embed = nn.Parameter(
                torch.zeros(1, max_timesteps, self.embed_dim)
            )
            # Per-view embedding (1, V_max, D), learnable, distinguishes cameras
            self.action_view_embed = nn.Parameter(
                torch.zeros(1, max_views, self.embed_dim)
            )
            # Initialized with randn (like DA3 camera_token)
            nn.init.normal_(self.action_token, std=1.0)
            nn.init.normal_(self.action_view_embed, std=0.02)
            self._init_action_timestep_embedding(max_timesteps)
            # Temporal embedding for image patches (shared with action tokens)
            if use_temporal_embed:
                self.temporal_embed = nn.Parameter(
                    torch.zeros(1, max_timesteps, self.embed_dim)
                )
                self._init_sinusoidal_embed(self.temporal_embed, max_timesteps)
            else:
                self.register_parameter("temporal_embed", None)
        else:
            self.register_parameter("action_token", None)
            self.register_parameter("action_timestep_embed", None)
            self.register_parameter("action_view_embed", None)
            self.register_parameter("temporal_embed", None)

        # Action input projection (for autoencoding: feed GT action as token input)
        self.action_input_proj = nn.Linear(7 * self.action_steps_per_token, self.embed_dim)

        self._init_normalization(normalization_stat_path)
        self._set_backbone_trainable(not freeze_backbone)

        self.dpt_head.eval()
        for p in self.dpt_head.parameters():
            p.requires_grad = False
        self._prepare_dpt_head_for_float_decode()
        # Keep camera-decoder LayerNorm weights in float32 under bf16 autocast.
        if self.da3_model.model.cam_dec is not None:
            for m in self.da3_model.model.cam_dec.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.float()
        # CameraDec.forward has explicit `feat.float()` casts before fc_t/fc_qvec/fc_fov,
        # which creates float32 input against bf16 Linear weights under bf16 autocast.
        # Cast the whole cam_dec to float32 so it stays consistent. It's small (<1M params).
        if self.da3_model.model.cam_dec is not None:
            self.da3_model.model.cam_dec.float()

    def _prepare_dpt_head_for_float_decode(self) -> None:
        """Keep DPT decode weights consistent with float32 monitor features."""
        self.dpt_head.float()
        for m in self.dpt_head.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.float()

    def _load_full_model(self, ckpt_path: str, model_name: str):
        # DA3's api.py eagerly pulls in export / pose_align / output_processor,
        # which transitively import moviepy, pycolmap, trimesh, imageio, evo,
        # addict. Frozen-encoder inference skips those modules, so register
        # recursive stubs in sys.modules
        # before importing DA3.
        _install_da3_optional_stubs()
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3(model_name=model_name)
        state_path = ckpt_path
        if os.path.isdir(state_path):
            candidates = (
                "model.safetensors",
                "pytorch_model.bin",
                "model.pt",
                "model.pth",
            )
            for name in candidates:
                candidate = os.path.join(state_path, name)
                if os.path.exists(candidate):
                    state_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No DA3 checkpoint file found under directory: {ckpt_path}"
                )
        if str(state_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            ckpt = load_file(state_path)
        else:
            ckpt = torch.load(state_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "module"):
                nested = ckpt.get(key)
                if isinstance(nested, dict):
                    ckpt = nested
                    break
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unsupported DA3 checkpoint object from {state_path}: {type(ckpt)!r}")

        ckpt = {str(k).replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        if any(k.startswith("backbone.") for k in ckpt):
            da3_weights = {
                k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")
            }
        else:
            da3_weights = ckpt
        try:
            missing, unexpected = model.load_state_dict(da3_weights, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load DA3 weights for model_name={model_name!r} "
                f"from {ckpt_path!r}. Check that stage_1.ckpt_path matches "
                "stage_1.model_name."
            ) from exc
        print(
            f"DA3 {model_name} loaded: {len(da3_weights)} keys, missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  Missing (first 5): {missing[:5]}")
        return model

    def _set_backbone_trainable(self, trainable: bool):
        for p in self.da3_model.parameters():
            p.requires_grad = False
        if trainable:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def freeze_blocks_before(self, block_idx: int):
        """Freeze patch embed and all backbone blocks below ``block_idx``."""
        trans = self.backbone.pretrained
        for name, param in trans.named_parameters():
            parts = name.split(".")
            if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
                param.requires_grad = int(parts[1]) >= block_idx
            else:
                param.requires_grad = False

    def _init_normalization(self, stat_path: Optional[str]):
        if stat_path is not None:
            stats = torch.load(stat_path, map_location="cpu")
            self.register_buffer("latent_mean", stats.get("mean", None))
            self.register_buffer("latent_var", stats.get("var", None))
            self.do_normalization = True
        else:
            self.latent_mean = None
            self.latent_var = None
            self.do_normalization = False

    @staticmethod
    def _init_sinusoidal_embed(param: nn.Parameter, max_len: int):
        embed_dim = param.shape[-1]
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(embed_dim, 1))
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        with torch.no_grad():
            param.copy_(pe.unsqueeze(0))

    def _init_action_timestep_embedding(self, max_len: int):
        if self.action_timestep_embed is None:
            return
        self._init_sinusoidal_embed(self.action_timestep_embed, max_len)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.do_normalization:
            return z
        mean = self.latent_mean.to(z.device, z.dtype)
        var = self.latent_var.to(z.device, z.dtype)
        std = torch.sqrt(var + self.eps)
        if z.ndim == 3:
            return (z - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        if z.ndim == 4:
            return (z - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return z

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.do_normalization:
            return z
        mean = self.latent_mean.to(z.device, z.dtype)
        var = self.latent_var.to(z.device, z.dtype)
        std = torch.sqrt(var + self.eps)
        if z.ndim == 3:
            return z * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        if z.ndim == 4:
            return z * std.reshape(1, -1, 1, 1) + mean.reshape(1, -1, 1, 1)
        return z

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            b, v, c, h, w = images.shape
            images = images.reshape(b * v, c, h, w)
            images = (images - self.encoder_mean) / self.encoder_std
            return images.reshape(b, v, c, h, w)
        return (images - self.encoder_mean) / self.encoder_std

    def project_proprio(self, proprio: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Deprecated: kept for checkpoint compatibility. No longer used."""
        return None

    def _build_camera_tokens(
        self,
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
        cam_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cam_token is not None:
            return cam_token.to(device=device, dtype=dtype)
        trans = self.backbone.pretrained
        ref_token = trans.camera_token[:, :1].expand(batch_size, -1, -1)
        if num_views <= 1:
            return ref_token.to(dtype=dtype)
        src_token = trans.camera_token[:, 1:].expand(batch_size, num_views - 1, -1)
        return torch.cat([ref_token, src_token], dim=1).to(dtype=dtype)

    def _build_action_tokens(
        self,
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
        action_input: Optional[torch.Tensor] = None,
        override_n_steps: Optional[int] = None,
        force_action_input: bool = False,
        per_slot_ae_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build (B, T*V, 1, D) action tokens.

        Conditioning modes (mutually exclusive, evaluated in order):
          1. `per_slot_ae_mask` given (preferred for unified future-predictor):
             deterministic per-timestep AE/noact selection. mask[t]=True ->
             add action_input_proj(action_input[t]); False -> learnable-only.
             Required to avoid GT future leakage (spec rule).
          2. `force_action_input`: Stage 2 GT generation : always AE.
          3. Stochastic (legacy Stage 1): per-sample AE at `action_input_rate`.

        At inference without a mask and no action_input, always uses the
        learnable token.

        Args:
            action_input: (B, T, 7) or (B, T, chunk, 7) normalized GT actions.
            override_n_steps: If set, use this as T instead of self.n_action_steps.
            force_action_input: If True, always use action_input (bypass stochastic/eval logic).
            per_slot_ae_mask: Optional (T,) or (B, T) boolean tensor gating
                per-slot AE injection. Requires `action_input` to be given.
        """
        if self.n_action_steps <= 0 or self.action_token is None:
            raise ValueError("Action tokens requested, but n_action_steps is 0.")
        T = override_n_steps if override_n_steps is not None else self.n_action_steps
        V = self.views_per_timestep
        expected_views = T * V
        if num_views != expected_views:
            raise ValueError(
                f"Expected {expected_views} views for action injection, got {num_views}."
            )

        # base (shared, like DA3 camera_token) + timestep: (B, T, D)
        tok = (self.action_token + self.action_timestep_embed[:, :T, :]).to(device=device, dtype=dtype)
        tok = tok.expand(batch_size, -1, -1).clone()

        # GT action conditioning (if provided)
        if action_input is not None:
            ac = action_input.to(device=device, dtype=dtype)
            if ac.ndim == 4:
                bsz, steps, chunk, dims = ac.shape
                if chunk != self.action_steps_per_token or dims != 7:
                    raise ValueError(
                        f"Expected action_input shape (B, T, {self.action_steps_per_token}, 7), "
                        f"got {tuple(ac.shape)}."
                    )
                ac = ac.reshape(bsz, steps, chunk * dims)
            elif ac.ndim == 3:
                if ac.shape[-1] != 7 * self.action_steps_per_token:
                    if self.action_steps_per_token != 1 or ac.shape[-1] != 7:
                        raise ValueError(
                            f"Expected action_input last dim {7 * self.action_steps_per_token}, "
                            f"got {ac.shape[-1]}."
                        )
            else:
                raise ValueError(f"Unsupported action_input shape: {tuple(ac.shape)}")
            action_cond = self.action_input_proj(ac)

            if per_slot_ae_mask is not None:
                mask = per_slot_ae_mask.to(device=device, dtype=dtype)
                if mask.dim() == 1:
                    mask = mask.view(1, -1, 1).expand(batch_size, T, 1)
                elif mask.dim() == 2:
                    if mask.shape[0] == 1 and batch_size > 1:
                        mask = mask.expand(batch_size, T)
                    mask = mask.view(mask.shape[0], mask.shape[1], 1)
                else:
                    raise ValueError(
                        f"per_slot_ae_mask must be 1D (T,) or 2D (B,T); got {tuple(mask.shape)}"
                    )
                if mask.shape[1] != T:
                    raise ValueError(f"per_slot_ae_mask T dim {mask.shape[1]} != T={T}")
                tok = tok + mask * action_cond
            elif force_action_input:
                # Stage 2 GT: always use action autoencoding
                tok = tok + action_cond
            elif self.training:
                # Stochastic: action_input_rate -> GT action, else -> learnable only
                use_action = torch.rand(batch_size, 1, 1, device=device) < self.action_input_rate
                tok = tok + use_action.float() * action_cond
            else:
                # Inference with action_input provided: use it deterministically
                tok = tok + action_cond

        # expand to per-view: (B, T*V, D)
        tok = tok.repeat_interleave(V, dim=1)

        return tok.unsqueeze(2)  # (B, T*V, 1, D)

    def build_learnable_action_seed_tokens(
        self,
        batch_size: int,
        steps: int,
        views_per_step: int,
        device: torch.device,
        dtype: torch.dtype,
        start_timestep: int = 0,
    ) -> torch.Tensor:
        """Build Stage-1 learnable action seeds for shallow propagation.

        Returns `(B, steps, views_per_step, D)` using the same base action token
        and timestep embedding that `_build_action_tokens()` inserts at
        `alt_start`, without adding GT action-conditioning.
        """
        if self.n_action_steps <= 0 or self.action_token is None:
            raise ValueError("Action seed tokens requested, but n_action_steps is 0.")
        start = int(start_timestep)
        end = start + int(steps)
        if start < 0 or end > self.action_timestep_embed.shape[1]:
            raise ValueError(
                f"Action seed timestep range [{start}, {end}) exceeds "
                f"available embeddings {self.action_timestep_embed.shape[1]}."
            )
        tok = (self.action_token + self.action_timestep_embed[:, start:end, :]).to(
            device=device,
            dtype=dtype,
        )
        tok = tok.expand(int(batch_size), -1, -1).clone()
        tok = tok[:, :, None, :].expand(-1, -1, int(views_per_step), -1).contiguous()
        return tok

    def _normalize_cat_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        trans = self.backbone.pretrained
        if tokens.shape[-1] == self.embed_dim:
            return trans.norm(tokens)
        if tokens.shape[-1] == self.hidden_size:
            return torch.cat(
                [tokens[..., : self.embed_dim], trans.norm(tokens[..., self.embed_dim :])], dim=-1
            )
        raise ValueError(f"Unexpected token shape: {tokens.shape}")

    def _prepare_input(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        b, s, _, h, w = images.shape
        return images, b, s, h, w

    def _run_backbone(
        self,
        images: torch.Tensor,
        target_layers: Sequence[int],
        stop_at_first: bool = False,
        inject_action: bool = False,
        cam_token: Optional[torch.Tensor] = None,
        action_input: Optional[torch.Tensor] = None,
        override_n_steps: Optional[int] = None,
        force_action_input: bool = False,
        per_slot_ae_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        images, batch_size, num_views, height, width = self._prepare_input(images)
        trans = self.backbone.pretrained

        current_x = trans.prepare_tokens_with_masks(images)
        pos, pos_nodiff = trans._prepare_rope(batch_size, num_views, height, width, current_x.device)
        local_x = None
        results: Dict[int, Dict[str, torch.Tensor]] = {}
        action_inserted = False

        # VGA-style two-stream state (activates only when the flag is on AND
        # action tokens get injected; otherwise this path is skipped).
        use_two_stream = bool(self.action_only_frame_attn) and bool(inject_action)
        img_x: Optional[torch.Tensor] = None   # (B, V, N_img, D): cam + reg + patches
        act_x: Optional[torch.Tensor] = None   # (B, V, 1, D) : one action token per view
        img_local: Optional[torch.Tensor] = None
        act_local: Optional[torch.Tensor] = None

        def _split_streams(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Position 1 carries the action token (inserted at alt_start).
            img = torch.cat([x[:, :, :1], x[:, :, 2:]], dim=2)
            act = x[:, :, 1:2]
            return img, act

        def _merge_streams(img: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
            return torch.cat([img[:, :, :1], act, img[:, :, 1:]], dim=2)

        for i, blk in enumerate(trans.blocks):
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff, pos

            # Skip DA3's reference view reordering (saddle_balanced); always use "first".
            # DA3 forward_features reorders at alt_start-1, and this path bypasses it.

            if trans.alt_start != -1 and i == trans.alt_start:
                camera_tokens = self._build_camera_tokens(
                    batch_size, num_views, current_x.device, current_x.dtype, cam_token=cam_token
                )
                current_x[:, :, 0] = camera_tokens
                if local_x is not None:
                    local_x[:, :, 0] = camera_tokens

                # Add temporal embedding to image patches (if enabled)
                if inject_action and self.temporal_embed is not None:
                    T = override_n_steps if override_n_steps is not None else self.n_action_steps
                    V = self.views_per_timestep
                    # temporal_embed: (1, T, D) → expand to (1, T*V, D)
                    t_emb = self.temporal_embed[:, :T, :].to(
                        device=current_x.device, dtype=current_x.dtype
                    )
                    t_emb = t_emb.unsqueeze(2).expand(-1, -1, V, -1).reshape(1, T * V, -1)
                    # Add to all tokens (cam + reg + patches) per view
                    current_x = current_x + t_emb.unsqueeze(2)
                    if local_x is not None:
                        local_x = local_x + t_emb.unsqueeze(2)

                if inject_action:
                    action_tokens = self._build_action_tokens(
                        batch_size,
                        num_views,
                        current_x.device,
                        current_x.dtype,
                        action_input=action_input,
                        override_n_steps=override_n_steps,
                        force_action_input=force_action_input,
                        per_slot_ae_mask=per_slot_ae_mask,
                    )
                    current_x = torch.cat(
                        [current_x[:, :, :1], action_tokens, current_x[:, :, 1:]], dim=2
                    )
                    if local_x is not None:
                        local_x = torch.cat(
                            [local_x[:, :, :1], action_tokens, local_x[:, :, 1:]], dim=2
                        )
                    if pos is not None:
                        act_pos = torch.zeros(
                            batch_size,
                            num_views,
                            1,
                            2,
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        pos = torch.cat([pos[:, :, :1], act_pos, pos[:, :, 1:]], dim=2)
                        pos_nodiff = torch.cat(
                            [pos_nodiff[:, :, :1], act_pos, pos_nodiff[:, :, 1:]], dim=2
                        )
                        # Re-bind after reassignment so current block sees updated pos
                        g_pos, l_pos = pos_nodiff, pos
                    action_inserted = True

                    # Initialize two-stream state by splitting the just-assembled sequence.
                    if use_two_stream:
                        img_x, act_x = _split_streams(current_x)
                        if local_x is not None:
                            img_local, act_local = _split_streams(local_x)

            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type, pos_emb = "global", g_pos
            else:
                attn_type, pos_emb = "local", l_pos

            if use_two_stream and i >= trans.alt_start and img_x is not None:
                if attn_type == "local":
                    # Image stream: standard per-view local attention.
                    # img_pos drops the action slot (index 1) that was inserted into pos.
                    img_pos_local = (
                        torch.cat([pos[:, :, :1], pos[:, :, 2:]], dim=2) if pos is not None else None
                    )
                    img_x = trans.process_attention(
                        img_x, blk, attn_type="local", pos=img_pos_local
                    )
                    # Action stream: reshape to (B, 1, V, D) so process_attention's
                    # "local" mode (which flattens view dim into batch) computes a
                    # single attention over all V action tokens together. No spatial
                    # RoPE for action tokens : action_timestep_embed already encodes
                    # temporal position internally.
                    act_grouped = act_x.transpose(1, 2).contiguous()  # (B, 1, V, D)
                    act_grouped = trans.process_attention(
                        act_grouped, blk, attn_type="local", pos=None
                    )
                    act_x = act_grouped.transpose(1, 2).contiguous()  # (B, V, 1, D)
                    img_local = img_x
                    act_local = act_x
                else:  # global
                    # Merge streams → standard global attention → split back.
                    merged = _merge_streams(img_x, act_x)
                    merged = trans.process_attention(
                        merged, blk, attn_type="global", pos=g_pos
                    )
                    img_x, act_x = _split_streams(merged)

                # Reassemble current_x (and local_x for cat_token) only when a
                # downstream consumer needs them: at OUT_LAYER extraction, or for
                # the baseline path we would have kept in sync.
                if i in target_layers:
                    current_x = _merge_streams(img_x, act_x)
                    if img_local is not None:
                        local_x = _merge_streams(img_local, act_local)
            else:
                current_x = trans.process_attention(
                    current_x, blk, attn_type=attn_type, pos=pos_emb
                )
                if attn_type == "local":
                    local_x = current_x

            if i in target_layers:
                effective_local = current_x if local_x is None else local_x
                raw_tokens = (
                    torch.cat([effective_local, current_x], dim=-1) if trans.cat_token else current_x
                )
                current_norm = trans.norm(current_x)
                results[i] = {
                    "raw": raw_tokens,
                    "current_norm": current_norm,
                    "action_injected": action_inserted,
                }
                if stop_at_first:
                    return results

        return results

    def _format_level_output(
        self,
        raw_tokens: torch.Tensor,
        current_norm: torch.Tensor,
        action_injected: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        extra_tokens = 2 if action_injected else 1
        start_idx = extra_tokens + self.backbone.pretrained.num_register_tokens
        patches = raw_tokens[:, :, start_idx:, :]
        camera_token = raw_tokens[:, :, 0, :]
        action_token = current_norm[:, :, 1, :] if action_injected else None
        return patches, camera_token, action_token, raw_tokens

    def _norm_feats_for_dpt(
        self,
        level_feats: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the final-block LayerNorm to the second half of cat_token feats.

        Replicates `depth_anything_3.model.dinov2.vision_transformer.get_intermediate_layers`
        (lines 385-390): when the DA3 backbone runs with `cat_token=True` (our
        DA3-Giant setup), the official path concatenates [local_out, current_x]
        along the feature dim and then applies `self.norm` ONLY to the second
        half before the DPT head consumes it. Our `_run_backbone` builds
        `raw_tokens = cat([effective_local, current_x], dim=-1)` without that
        norm, so every `dpt_head(...)` call site in this module used to feed
        out-of-distribution activations to the head and got noticeably softer /
        noisier depth. Centralizing the fix here guarantees parity with the
        official DA3 inference pipeline across all four DPT entry points
        (`decode_depth`, `encode_with_actions`+DPT decode, `propagate_and_decode`,
        `propagate_shallow_with_actions`). Confirmed on 2026-04-21 by
        side-by-side against `depth_anything_3.api.DepthAnything3.inference`.
        """
        trans = self.backbone.pretrained
        embed_dim = int(trans.embed_dim)
        cat_token = bool(getattr(trans, "cat_token", False))
        if not cat_token:
            return level_feats
        trans_norm = trans.norm
        out = []
        trans_norm_dtype = trans_norm.weight.dtype
        for p, c in level_feats:
            if p.shape[-1] == 2 * embed_dim:
                first = p[..., :embed_dim]
                # `trans.norm` weights track the surrounding module dtype (bf16
                # under DeepSpeed bf16). DPT callers upcast `p` to float32
                # before passing it in (see `_prepare_dpt_head_for_float_decode`
                # / the enclosing `torch.autocast(enabled=False)` block). Match
                # the input to the LayerNorm's own weight dtype so F.layer_norm
                # to avoid dtype mismatch errors, then cast back
                # to the rest of `p`'s dtype for cat.
                second = trans_norm(
                    p[..., embed_dim:].to(trans_norm_dtype)
                ).to(p.dtype)
                p = torch.cat([first, second], dim=-1)
            out.append((p, c))
        return out

    def decode_depth(
        self,
        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None,
        views_per_sequence: Optional[int] = None,
        frames_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Run DPT head on multi-level features to produce depth map.

        Args:
            features_per_level: list of (patches, cls_token) from encode_with_actions
                or encode_all_levels.  patches: (B*V, N, C), cls: (B*V, C).

        Returns:
            depth: (B*V, H, W) predicted depth.
        """
        # Delegate to decode_depth_full so the backward-compat path and the
        # full-dict path share the exact DPT call site.
        return self.decode_depth_full(
            features_per_level,
            batch_size=batch_size,
            views_per_sequence=views_per_sequence,
            frames_chunk_size=frames_chunk_size,
        )["depth"]

    def decode_depth_full(
        self,
        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: Optional[int] = None,
        views_per_sequence: Optional[int] = None,
        frames_chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run DPT head and return the full output dict.

        Keys returned (for DA3-Giant):
            - ``depth``: (B*V, H, W)
            - ``depth_conf``: (B*V, H, W)
            - ``ray``: (B*V, Hr, Wr, 6)
            - ``ray_conf``: (B*V, Hr, Wr)
        where ``(Hr, Wr)`` is the DPT head's ray resolution (128 for 224 input).
        """
        image_h = image_w = self.h_patches * self.PATCH_SIZE
        self._prepare_dpt_head_for_float_decode()
        with torch.autocast(device_type=features_per_level[0][0].device.type, enabled=False):
            normed = self._norm_feats_for_dpt(
                [(p.float(), c.float() if c is not None else c) for p, c in features_per_level]
            )
            if batch_size is not None:
                dpt_batch = int(batch_size)
                dpt_views = int(views_per_sequence) if views_per_sequence is not None else None
                if dpt_views is None:
                    flat = int(normed[0][0].shape[0])
                    if flat % dpt_batch != 0:
                        raise ValueError(
                            f"Cannot reshape DPT features with flat={flat} into batch_size={dpt_batch}."
                        )
                    dpt_views = flat // dpt_batch
            else:
                if normed and normed[0][0].ndim == 4:
                    dpt_batch = int(normed[0][0].shape[0])
                    dpt_views = int(normed[0][0].shape[1])
                else:
                    dpt_batch = 1
                    dpt_views = int(normed[0][0].shape[0]) if normed and normed[0][0].ndim == 3 else None

            feats_4d = []
            for p, c in normed:
                if p.ndim == 3:
                    if dpt_views is None:
                        raise ValueError(f"Cannot infer DPT view count from patch shape {tuple(p.shape)}.")
                    p = p.reshape(dpt_batch, dpt_views, p.shape[-2], p.shape[-1])
                    if c is not None:
                        c = c.reshape(dpt_batch, dpt_views, c.shape[-1])
                feats_4d.append((p, c) if c is not None else (p,))
            output = self.dpt_head(feats_4d, image_h, image_w, patch_start_idx=0)

        # DPT returns (B, S, ...) for the standard `(patches, cls)` contract.
        # Public decode_depth historically returns flattened `(B*S, ...)`.
        squeezed: Dict[str, torch.Tensor] = {}
        for k, v in output.items():
            if not hasattr(v, "dim"):
                squeezed[k] = v
            elif (
                dpt_views is not None
                and v.dim() >= 2
                and v.shape[0] == dpt_batch
                and v.shape[1] == dpt_views
            ):
                squeezed[k] = v.reshape(dpt_batch * dpt_views, *v.shape[2:])
            elif v.dim() >= 1 and v.shape[0] == 1:
                squeezed[k] = v.squeeze(0)
            else:
                squeezed[k] = v
        return squeezed

    def decode_camera(
        self,
        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Run camera decoder on last-level camera token to produce pose encoding.

        Args:
            features_per_level: list of (patches, cls_token).

        Returns:
            pose_enc: (B, V, 9) camera pose encoding [t(3), qvec(4), fov(2)],
            or None when cam_dec is unavailable.
        """
        cam_dec = self.da3_model.model.cam_dec
        if cam_dec is None:
            return None
        # Last level cls_token: (B*V, C) -> need (B, V, C)
        cls_last = features_per_level[-1][1]  # (B*V, C)
        # We don't know B vs V here, so return as (B*V, 1, 9) for flexibility
        bv = cls_last.shape[0]
        feat = cls_last.unsqueeze(1)  # (B*V, 1, C)
        # CameraDec.forward() internally calls feat.float() before fc_t/fc_qvec/fc_fov,
        # so the module must be in float32. Model-wide .to(bfloat16) (e.g. teacher viz
        # cleanup) can override __init__'s cam_dec.float(), so we force it here.
        cam_dec.float()
        pose_enc = cam_dec(feat)  # (B*V, 1, 9)
        return pose_enc.squeeze(1)  # (B*V, 9)

    @torch.no_grad()
    def encode_single(
        self, images: torch.Tensor, level: int = 0, return_cls: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, num_views = images.shape[:2]
        target_layer = self.out_layers[level]
        results = self._run_backbone(images, [target_layer], stop_at_first=True)
        level_output = results[target_layer]
        patches, cls_token, _, _ = self._format_level_output(
            level_output["raw"], level_output["current_norm"], action_injected=False
        )
        patches = patches.reshape(batch_size * num_views, patches.shape[2], patches.shape[3])
        cls_token = cls_token.reshape(batch_size * num_views, cls_token.shape[-1])
        if return_cls:
            return patches, cls_token
        return patches

    @torch.no_grad()
    def encode_all_levels(self, images: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, num_views = images.shape[:2]
        results = self._run_backbone(images, self.out_layers, stop_at_first=False)
        level_results = {}
        for layer_idx, level_output in results.items():
            level = self.out_layers.index(layer_idx)
            patches, cls_token, _, _ = self._format_level_output(
                level_output["raw"], level_output["current_norm"], action_injected=False
            )
            level_results[level] = (
                patches.reshape(batch_size * num_views, patches.shape[2], patches.shape[3]),
                cls_token.reshape(batch_size * num_views, cls_token.shape[-1]),
            )
        return level_results

    @torch.no_grad()
    def encode_all_levels_raw(self, images: torch.Tensor) -> List[torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        results = self._run_backbone(images, self.out_layers, stop_at_first=False)
        return [results[layer_idx]["raw"] for layer_idx in self.out_layers]

    @torch.no_grad()
    def encode_shallow_visual_slots(
        self,
        images: torch.Tensor,
        T: int,
        V: int,
        target_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Export DA3 pre-global visual tokens before `alt_start`.

        The returned layout preserves DA3's visual token order before action
        insertion: `[CLS, registers, patches]`. No camera/action tokens are
        inserted here; those are supplied when resuming from `alt_start`.
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, num_views = images.shape[:2]
        if num_views != int(T) * int(V):
            raise ValueError(f"Expected T*V={int(T) * int(V)} views, got {num_views}.")
        if target_layer is None:
            target_layer = self.shallow_target_layer
        target_layer = int(target_layer)
        alt_start = int(getattr(self.backbone.pretrained, "alt_start", -1))
        if alt_start >= 0 and int(target_layer) >= alt_start:
            raise ValueError(
                "encode_shallow_visual_slots must export a pre-global layer; "
                f"got target_layer={target_layer} with alt_start={alt_start}."
            )
        results = self._run_backbone(
            images,
            [target_layer],
            stop_at_first=True,
            inject_action=False,
        )
        raw = results[target_layer]["raw"]
        if raw.shape[-1] == self.hidden_size:
            visual = raw[..., : self.embed_dim]
        elif raw.shape[-1] == self.embed_dim:
            visual = raw
        else:
            raise ValueError(f"Unexpected shallow feature dim: {raw.shape[-1]}")
        return {
            "visual_tokens": visual.reshape(batch_size, int(T), int(V), visual.shape[2], self.embed_dim),
            "raw": raw,
            "layer": torch.tensor(target_layer, device=visual.device),
        }

    def encode_with_actions(
        self,
        images: torch.Tensor,
        action_input: Optional[torch.Tensor] = None,
        override_n_steps: Optional[int] = None,
        levels: Optional[List[int]] = None,
        force_action_input: bool = False,
        per_slot_ae_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[torch.Tensor]]:
        """Run DA3 with action-token injection and return raw per-level features.

        Args:
            action_input: (B, T, 7) or (B, T, chunk, 7) normalized GT actions for
                autoencoding (training only). During training, stochastically selects
                between action_input and learnable action_cond_token. At inference,
                always uses learnable token.
            override_n_steps: If set, overrides self.n_action_steps for this call.
                Use override_n_steps=1 to encode ref-only views (2 views -> 1 action step).
            levels: Which semantic levels to extract (0-3). Defaults to all [0,1,2,3].
                Use levels=[0] to stop early at layer 19, skipping blocks 20-39.
            force_action_input: If True, always use action_input for action tokens
                (bypasses stochastic selection). Use for Stage 2 GT generation.
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, num_views = images.shape[:2]

        if levels is None:
            target_layers = self.out_layers
            stop_early = False
        else:
            target_layers = [self.out_layers[l] for l in levels]
            stop_early = max(levels) < len(self.out_layers) - 1

        results = self._run_backbone(
            images,
            target_layers,
            stop_at_first=stop_early,
            inject_action=True,
            action_input=action_input,
            override_n_steps=override_n_steps,
            force_action_input=force_action_input,
            per_slot_ae_mask=per_slot_ae_mask,
        )

        features_per_level: List[Tuple[torch.Tensor, torch.Tensor]] = []
        raw_levels: List[torch.Tensor] = []
        action_tokens: List[torch.Tensor] = []

        for layer_idx in target_layers:
            level_output = results[layer_idx]
            patches, cls_token, action_token, raw_tokens = self._format_level_output(
                level_output["raw"],
                level_output["current_norm"],
                action_injected=True,
            )
            features_per_level.append(
                (
                    patches.reshape(batch_size * num_views, patches.shape[2], patches.shape[3]),
                    cls_token.reshape(batch_size * num_views, cls_token.shape[-1]),
                )
            )
            raw_levels.append(raw_tokens)
            if action_token is None:
                raise RuntimeError("Action token was requested but not produced.")
            action_tokens.append(action_token)

        return features_per_level, action_tokens[-1], raw_levels

    def _deep_prefix_lengths(
        self,
        step_valid_mask: Optional[torch.Tensor],
        batch_size: int,
        steps: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if step_valid_mask is None:
            return None
        valid = step_valid_mask.to(device=device, dtype=torch.bool)
        if valid.shape != (batch_size, steps):
            raise ValueError(
                f"step_valid_mask shape {tuple(valid.shape)} != {(batch_size, steps)}"
            )
        if bool(valid.all().item()):
            return None
        arange = torch.arange(steps, device=device).view(1, steps)
        lengths = valid.long().sum(dim=1)
        expected = arange < lengths.view(batch_size, 1)
        if not bool(torch.equal(valid, expected)):
            raise ValueError(
                "step_valid_mask must be prefix-valid for compact DA3 deep propagation."
            )
        return lengths

    def _scatter_compact_deep_result(
        self,
        full: Dict[str, torch.Tensor],
        part: Dict[str, torch.Tensor],
        batch_indices: torch.Tensor,
        part_steps: int,
        total_steps: int,
        views_per_step: int,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        flat_dst = (
            batch_indices[:, None, None] * (total_steps * views_per_step)
            + torch.arange(part_steps, device=batch_indices.device)[None, :, None] * views_per_step
            + torch.arange(views_per_step, device=batch_indices.device)[None, None, :]
        ).reshape(-1)
        part_flat = int(batch_indices.numel()) * int(part_steps) * int(views_per_step)
        full_flat = int(batch_size) * int(total_steps) * int(views_per_step)

        def scatter_flat_tensor(dst_key: str, tensor: torch.Tensor) -> None:
            part_batch = int(batch_indices.numel())
            part_seq = int(part_steps) * int(views_per_step)
            if tensor.ndim >= 2 and tensor.shape[0] == part_batch and tensor.shape[1] == part_seq:
                shape = [int(batch_size), int(total_steps) * int(views_per_step), *tensor.shape[2:]]
                out = full.get(dst_key)
                if out is None:
                    out = tensor.new_zeros(shape)
                else:
                    out = out.reshape(shape)
                seq_dst = (
                    torch.arange(part_steps, device=batch_indices.device)[:, None] * views_per_step
                    + torch.arange(views_per_step, device=batch_indices.device)[None, :]
                ).reshape(-1)
                batch_dst = batch_indices[:, None].expand(-1, part_seq).reshape(-1)
                seq_dst = seq_dst[None, :].expand(part_batch, -1).reshape(-1)
                full[dst_key] = out.index_put(
                    (batch_dst, seq_dst),
                    tensor.reshape(part_batch * part_seq, *tensor.shape[2:]),
                )
                return
            if tensor.ndim >= 1 and tensor.shape[0] == part_flat:
                shape = list(tensor.shape)
                shape[0] = full_flat
                out = full.get(dst_key)
                if out is None:
                    out = tensor.new_zeros(shape)
                full[dst_key] = out.index_copy(0, flat_dst, tensor)
                return
            if tensor.ndim >= 2 and tensor.shape[1] == part_flat:
                shape = list(tensor.shape)
                shape[1] = full_flat
                out = full.get(dst_key)
                if out is None:
                    out = tensor.new_zeros(shape)
                full[dst_key] = out.index_copy(1, flat_dst, tensor)
                return
            full[dst_key] = tensor

        for key, value in part.items():
            if key == "level_feats":
                existing = full.get(key)
                scattered_levels = []
                for level_idx, (patches, cls) in enumerate(value):
                    patch_shape = [full_flat, *patches.shape[2:]]
                    cls_shape = [full_flat, *cls.shape[2:]]
                    if existing is None:
                        full_patches = patches.new_zeros(patch_shape)
                        full_cls = cls.new_zeros(cls_shape)
                    else:
                        full_patches, full_cls = existing[level_idx]
                        full_patches = full_patches.reshape(full_flat, *full_patches.shape[2:])
                        full_cls = full_cls.reshape(full_flat, *full_cls.shape[2:])
                    part_patches = patches.reshape(part_flat, *patches.shape[2:])
                    part_cls = cls.reshape(part_flat, *cls.shape[2:])
                    full_patches = full_patches.index_copy(0, flat_dst, part_patches)
                    full_cls = full_cls.index_copy(0, flat_dst, part_cls)
                    scattered_levels.append(
                        (
                            full_patches.reshape(batch_size, total_steps * views_per_step, *patches.shape[2:]),
                            full_cls.reshape(batch_size, total_steps * views_per_step, *cls.shape[2:]),
                        )
                    )
                full[key] = scattered_levels
            elif isinstance(value, torch.Tensor):
                scatter_flat_tensor(key, value)
            else:
                full[key] = value
        return full

    def _run_deep_global_block_flex(
        self,
        x_4d: torch.Tensor,
        block: nn.Module,
        pos_4d: Optional[torch.Tensor],
        block_mask,
    ) -> torch.Tensor:
        """Apply one DA3 deep-stack global-attention block using
        flex_attention with a block-causal BlockMask.

        Replicates DA3's `Transformer.process_attention(global)` + the
        block's pre-norm attention + LayerScale + residual + FFN while
        substituting flex_attention(q, k, v, block_mask=...) for SDPA so
        the [B, H, L, S] attention score tensor is never materialized.

        Args:
            x_4d:    (B, total_view, token_count, C) : pre-attn input.
            block:   one DA3 transformer block (has .norm1, .attn, .ls1,
                     .norm2, .mlp, .ls2 sub-modules; matches DA3's
                     dinov2.layers.block.Block / NestedTensorBlock API).
            pos_4d:  (B, total_view, token_count, 2) RoPE pos or None.
            block_mask: flex_attention BlockMask covering the flattened
                     (B, total_view * token_count, C) seq.

        Returns: (B, total_view, token_count, C) : same shape as input.
        """
        if not _HAS_FLEX_ATTENTION:
            raise RuntimeError(
                "_run_deep_global_block_flex called without flex_attention."
            )
        b, s, n, c = x_4d.shape
        # Match process_attention's "b s n c -> b (s n) c" flatten.
        x = x_4d.reshape(b, s * n, c)
        pos_flat = pos_4d.reshape(b, s * n, -1) if pos_4d is not None else None

        # === attn sub-layer (pre-norm + attention + LayerScale + residual) ===
        h = block.norm1(x)
        attn = block.attn
        B_, N_, C_ = h.shape
        qkv = (
            attn.qkv(h)
            .reshape(B_, N_, 3, attn.num_heads, C_ // attn.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        if attn.rope is not None and pos_flat is not None:
            q = attn.rope(q, pos_flat)
            k = attn.rope(k, pos_flat)
        # flex_attention expects (B, H, L, D) : same as SDPA. Bool BlockMask
        # broadcasts across B and H.
        out = _flex_attention(q, k, v, block_mask=block_mask)
        out = out.transpose(1, 2).reshape(B_, N_, C_)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        x = x + block.ls1(out)

        # === FFN sub-layer (pre-norm + MLP + LayerScale + residual) ===
        x = x + block.ls2(block.mlp(block.norm2(x)))

        return x.reshape(b, s, n, c)

    def _propagate_shallow_with_actions_impl(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        decode_visuals: bool = True,
        dpt_chunk_size: Optional[int] = None,
        gradient_checkpointing: bool = False,
        return_multi_level: bool = False,
        step_valid_mask: Optional[torch.Tensor] = None,
        deep_temporal_causal_mask: bool = False,
        profile: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Resume DA3 from shallow visual tokens plus supplied action seeds.

        Args:
            visual_tokens: (B, steps, V, 1+R+P, D) in `[CLS, registers, patches]`
                order, normally from `encode_shallow_visual_slots()` or the
                gam predictor.
            action_tokens: (B, steps, V, D) action seeds inserted before block 13.
        """
        _cuda_profile_mark(profile, "deep_start")
        b, steps, v_count, n_visual, dim = visual_tokens.shape
        if dim != self.embed_dim:
            raise ValueError(f"Expected visual dim {self.embed_dim}, got {dim}.")
        if action_tokens.shape != (b, steps, v_count, dim):
            raise ValueError(
                f"action_tokens shape {tuple(action_tokens.shape)} != {(b, steps, v_count, dim)}"
            )
        prefix_lengths = self._deep_prefix_lengths(
            step_valid_mask=step_valid_mask,
            batch_size=b,
            steps=steps,
            device=visual_tokens.device,
        )
        if prefix_lengths is not None:
            if int(prefix_lengths.max().item()) <= 0:
                prefix_lengths = torch.full_like(prefix_lengths, steps)
            compact_result: Dict[str, torch.Tensor] = {}
            for length_tensor in torch.unique(prefix_lengths, sorted=True):
                sub_steps = int(length_tensor.item())
                if sub_steps <= 0:
                    continue
                batch_indices = torch.nonzero(prefix_lengths == sub_steps, as_tuple=False).flatten()
                part = self._propagate_shallow_with_actions_impl(
                    visual_tokens.index_select(0, batch_indices)[:, :sub_steps],
                    action_tokens.index_select(0, batch_indices)[:, :sub_steps],
                    decode_visuals=decode_visuals,
                    dpt_chunk_size=dpt_chunk_size,
                    gradient_checkpointing=gradient_checkpointing,
                    return_multi_level=return_multi_level,
                    step_valid_mask=None,
                    deep_temporal_causal_mask=deep_temporal_causal_mask,
                )
                compact_result = self._scatter_compact_deep_result(
                    compact_result,
                    part,
                    batch_indices=batch_indices,
                    part_steps=sub_steps,
                    total_steps=steps,
                    views_per_step=v_count,
                    batch_size=b,
                )
            return compact_result

        num_register_tokens = int(self.backbone.pretrained.num_register_tokens)
        num_patches = n_visual - 1 - num_register_tokens
        if num_patches <= 0:
            raise ValueError(f"Invalid shallow visual token count: {n_visual}")
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not square.")

        total_view = int(steps) * int(v_count)
        bv = b * total_view
        backbone_dtype = next(self.backbone.parameters()).dtype
        current_x = visual_tokens.reshape(b, total_view, n_visual, dim).to(backbone_dtype).clone()
        local_x = current_x.clone()
        action_x = action_tokens.reshape(b, total_view, 1, dim).to(backbone_dtype)

        trans = self.backbone.pretrained
        image_h = image_w = grid_size * self.PATCH_SIZE
        pos_all, pos_nodiff_all = trans._prepare_rope(
            b, total_view, image_h, image_w, current_x.device
        )

        if trans.alt_start == -1:
            raise RuntimeError("gam propagation requires DA3 alternating attention.")

        camera_tokens = self._build_camera_tokens(
            b, total_view, current_x.device, current_x.dtype
        )
        current_x[:, :, 0] = camera_tokens
        local_x[:, :, 0] = camera_tokens

        if self.temporal_embed is not None:
            t_emb = self.temporal_embed[:, :steps, :].to(
                device=current_x.device, dtype=current_x.dtype
            )
            t_emb = t_emb.unsqueeze(2).expand(-1, -1, v_count, -1).reshape(1, total_view, -1)
            current_x = current_x + t_emb.unsqueeze(2)
            local_x = local_x + t_emb.unsqueeze(2)

        current_x = torch.cat([current_x[:, :, :1], action_x, current_x[:, :, 1:]], dim=2)
        local_x = torch.cat([local_x[:, :, :1], action_x, local_x[:, :, 1:]], dim=2)
        if pos_all is not None:
            act_pos = torch.zeros(
                b, total_view, 1, 2,
                device=pos_all.device, dtype=pos_all.dtype,
            )
            pos_all = torch.cat([pos_all[:, :, :1], act_pos, pos_all[:, :, 1:]], dim=2)
            pos_nodiff_all = torch.cat(
                [pos_nodiff_all[:, :, :1], act_pos, pos_nodiff_all[:, :, 1:]], dim=2
            )

        token_count = n_visual + 1
        patch_start = 2 + num_register_tokens
        channels = self.embed_dim * 2
        level_feats = []
        start_block = int(trans.alt_start)
        use_checkpoint = bool(gradient_checkpointing) and torch.is_grad_enabled()
        # Optional strict timestep-block-causal mask for the deep global
        # attention. Without it, action token at timestep t can attend to
        # predicted obs tokens at timesteps > t (the predictor itself is
        # causal while bidirectional global attention re-mixes timesteps).
        # When enabled, each token is constrained to attend only to tokens
        # at timesteps <= its own. Local within-timestep attention is
        # unaffected. See docs/architecture.md "GAM AR FuturePredictor"
        # for the leakage discussion this fixes.
        #
        # Implementation note: an SDPA-with-additive-mask path materializes
        # the full [B, H, L, S] attention score tensor (~800 MB per global
        # block at H=8 V=2 K=258), which OOMs at 8 GPUs. FlexAttention with
        # a sparse BlockMask avoids that materialization. We require flex
        # to keep this strictly causal at production scale.
        global_block_mask = None
        if bool(deep_temporal_causal_mask) and int(steps) > 1:
            if not _HAS_FLEX_ATTENTION:
                raise RuntimeError(
                    "predictor.deep_temporal_causal_mask=true requires "
                    "torch.nn.attention.flex_attention; upgrade PyTorch."
                )
            # Cache BlockMask by (steps, v_count, token_count, device).
            # Rebuilding every forward is cheap; a stable
            # BlockMask helps the compiled flex_attention reuse its cache.
            cache_key = (int(steps), int(v_count), int(token_count), str(current_x.device))
            cached = self._deep_flex_block_mask_cache.get(cache_key)
            if cached is None:
                seq_len = int(total_view) * int(token_count)
                mask_mod = _make_deep_block_causal_mask_mod(
                    token_count=int(token_count),
                    v_count=int(v_count),
                )
                cached = _create_block_mask(
                    mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=current_x.device,
                )
                self._deep_flex_block_mask_cache[cache_key] = cached
            global_block_mask = cached
        _cuda_profile_mark(profile, "deep_prep_done")
        for i, blk in enumerate(trans.blocks):
            if i < start_block:
                continue
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff_all, pos_all
            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type, pos_emb = "global", g_pos
            else:
                attn_type, pos_emb = "local", l_pos
            use_flex = attn_type == "global" and global_block_mask is not None
            if use_checkpoint and current_x.requires_grad:
                if use_flex:
                    current_x = torch_checkpoint(
                        lambda x, block=blk, pos=pos_emb, mask=global_block_mask: self._run_deep_global_block_flex(
                            x, block, pos, mask
                        ),
                        current_x,
                        use_reentrant=False,
                    )
                else:
                    current_x = torch_checkpoint(
                        lambda x, block=blk, kind=attn_type, pos=pos_emb: trans.process_attention(
                            x, block, attn_type=kind, pos=pos
                        ),
                        current_x,
                        use_reentrant=False,
                    )
            else:
                if use_flex:
                    current_x = self._run_deep_global_block_flex(
                        current_x, blk, pos_emb, global_block_mask
                    )
                else:
                    current_x = trans.process_attention(
                        current_x, blk, attn_type=attn_type, pos=pos_emb
                    )
            if attn_type == "local":
                local_x = current_x
            # Capture multi-level features when either consumer requests them:
            #   * decode_visuals -> DPT depth/RGB head needs them downstream
            #   * return_multi_level -> Path B deep feature distillation
            if (decode_visuals or return_multi_level) and i in self.out_layers:
                cs = current_x.reshape(bv, token_count, -1)
                ls = local_x.reshape(bv, token_count, -1)
                level_feats.append(
                    (
                        torch.cat([ls[:, patch_start:], cs[:, patch_start:]], dim=-1).reshape(
                            b, total_view, num_patches, channels
                        ),
                        torch.cat([ls[:, 0], cs[:, 0]], dim=-1).reshape(
                            b, total_view, channels
                        ),
                    )
                )
        _cuda_profile_mark(profile, "deep_blocks_done")

        result: Dict[str, torch.Tensor] = {}
        if decode_visuals and len(level_feats) >= 4:
            self._prepare_dpt_head_for_float_decode()
            with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
                feats_float = self._norm_feats_for_dpt(
                    [(p.float(), c.float()) for p, c in level_feats]
                )
                dpt_kwargs = {"patch_start_idx": 0}
                if dpt_chunk_size is not None:
                    dpt_kwargs["chunk_size"] = max(1, int(dpt_chunk_size))
                dpt_output = self.dpt_head(feats_float, image_h, image_w, **dpt_kwargs)
            for key in ["depth", "depth_conf", "ray", "ray_conf", "rgb"]:
                if key in dpt_output and dpt_output[key] is not None:
                    result[key] = dpt_output[key]
        _cuda_profile_mark(profile, "deep_dpt_done")

        cs_final = current_x.reshape(bv, token_count, -1)
        result["action_tokens"] = trans.norm(cs_final[:, 1])
        _cuda_profile_mark(profile, "deep_final_done")
        # Optional: expose deep multi-level (OUT_LAYERS) features for
        # teacher-vs-student feature distillation (Path B feature reg).
        # Each entry is (patches[B,V,N,2C], cls[B,V,2C]) where 2C is
        # local+current concatenation matching the DPT input format and
        # teacher_da3.encode_all_levels(...) output format.
        if return_multi_level:
            result["level_feats"] = level_feats
        return result

    @torch.no_grad()
    def propagate_shallow_with_actions(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        decode_visuals: bool = True,
        step_valid_mask: Optional[torch.Tensor] = None,
        deep_temporal_causal_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self._propagate_shallow_with_actions_impl(
            visual_tokens,
            action_tokens,
            decode_visuals=decode_visuals,
            step_valid_mask=step_valid_mask,
            deep_temporal_causal_mask=deep_temporal_causal_mask,
        )

    def propagate_shallow_with_actions_grad(
        self,
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        decode_visuals: bool = True,
        dpt_chunk_size: int = 1,
        gradient_checkpointing: bool = False,
        return_multi_level: bool = False,
        step_valid_mask: Optional[torch.Tensor] = None,
        deep_temporal_causal_mask: bool = False,
        profile: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._propagate_shallow_with_actions_impl(
            visual_tokens,
            action_tokens,
            decode_visuals=decode_visuals,
            dpt_chunk_size=dpt_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            return_multi_level=return_multi_level,
            step_valid_mask=step_valid_mask,
            deep_temporal_causal_mask=deep_temporal_causal_mask,
            profile=profile,
        )

    def _propagate_and_decode_impl(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor = None,
        total_view: int = 1,
        dpt_chunk_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Propagate Level-0 features through remaining DA3 blocks and decode outputs.

        DA3 exports output-layer features as `[local_x, current_x]` when
        `cat_token=True`. A `D`-dim input is therefore a pre-global `local_x`
        seed and must replay the first output layer. A `2D` input already
        carries native `[local_x, current_x]` for Level 0 and resumes after it.
        """
        bv, num_patches, channels = level0_features.shape
        batch_size, num_views = bv // total_view, total_view
        grid_size = int(num_patches ** 0.5)
        embed_dim = self.embed_dim
        if channels not in (embed_dim, embed_dim * 2):
            raise ValueError(
                f"Expected Level-0 feature dim {embed_dim} or {embed_dim * 2}, got {channels}."
            )
        dpt_channels = embed_dim * 2

        if level0_cls is None:
            level0_cls = torch.zeros(
                bv, channels, device=level0_features.device, dtype=level0_features.dtype
            )

        backbone_dtype = next(self.backbone.parameters()).dtype
        level0_features = level0_features.to(backbone_dtype)
        level0_cls = level0_cls.to(backbone_dtype)

        feats = torch.cat([level0_cls.unsqueeze(1), level0_features], dim=1)
        feats_4d = feats.reshape(batch_size, num_views, num_patches + 1, channels)

        pre_global_seed = channels == embed_dim
        if pre_global_seed:
            current_x = feats_4d
            local_x = feats_4d
        else:
            current_x = feats_4d[..., embed_dim:]
            local_x = feats_4d[..., :embed_dim]

        trans = self.backbone.pretrained
        image_h = image_w = grid_size * self.PATCH_SIZE
        pos_all, pos_nodiff_all = trans._prepare_rope(
            batch_size, num_views, image_h, image_w, level0_features.device
        )

        token_count = num_patches + 1
        level_feats = []
        if not pre_global_seed:
            cs = current_x.reshape(bv, token_count, -1)
            ls = local_x.reshape(bv, token_count, -1)
            level_feats.append(
                (
                    torch.cat([ls[:, 1:], cs[:, 1:]], dim=-1).reshape(batch_size, num_views, num_patches, dpt_channels),
                    torch.cat([ls[:, 0], cs[:, 0]], dim=-1).reshape(batch_size, num_views, dpt_channels),
                )
            )

        start_block = self.out_layers[0] if pre_global_seed else self.out_layers[0] + 1
        for i, blk in enumerate(trans.blocks):
            if i < start_block:
                continue
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff_all, pos_all
            if trans.alt_start != -1 and i == trans.alt_start:
                camera_tokens = self._build_camera_tokens(
                    batch_size, num_views, current_x.device, current_x.dtype
                )
                current_x[:, :, 0] = camera_tokens
            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type, pos_emb = "global", g_pos
            else:
                attn_type, pos_emb = "local", l_pos
            current_x = trans.process_attention(current_x, blk, attn_type=attn_type, pos=pos_emb)
            if attn_type == "local":
                local_x = current_x
            if i in self.out_layers:
                cs = current_x.reshape(bv, token_count, -1)
                ls = local_x.reshape(bv, token_count, -1)
                level_feats.append(
                    (
                        torch.cat([ls[:, 1:], cs[:, 1:]], dim=-1).reshape(
                            batch_size, num_views, num_patches, dpt_channels
                        ),
                        torch.cat([ls[:, 0], cs[:, 0]], dim=-1).reshape(
                            batch_size, num_views, dpt_channels
                        ),
                    )
                )

        if len(level_feats) < 4:
            return {}

        self._prepare_dpt_head_for_float_decode()
        with torch.autocast(device_type=level0_features.device.type, enabled=False):
            feats_float = self._norm_feats_for_dpt(
                [(p.float(), c.float()) for p, c in level_feats]
            )
            dpt_kwargs = {"patch_start_idx": 0}
            if dpt_chunk_size is not None:
                dpt_kwargs["chunk_size"] = max(1, int(dpt_chunk_size))
            output = self.dpt_head(feats_float, image_h, image_w, **dpt_kwargs)

        result = {}
        for key in ["depth", "depth_conf", "rgb"]:
            if key in output and output[key] is not None:
                result[key] = output[key]
        return result

    @torch.no_grad()
    def propagate_and_decode(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor = None,
        total_view: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Monitor path: propagate Level-0 features and decode depth without gradients."""
        return self._propagate_and_decode_impl(level0_features, level0_cls, total_view)

    def propagate_and_decode_grad(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor = None,
        total_view: int = 1,
        dpt_chunk_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Training path: differentiable version used by optional depth supervision."""
        return self._propagate_and_decode_impl(
            level0_features,
            level0_cls,
            total_view,
            dpt_chunk_size=dpt_chunk_size,
        )

    def _propagate_and_predict_impl(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor,
        level0_action: torch.Tensor,
        total_view: int = 1,
        action_head: Optional[nn.Module] = None,
        cond_num: int = 2,
        decode_visuals: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Propagate Level-0 features (with action token) through remaining DA3 blocks.

        Returns action predictions from action head and, when requested, depth/RGB
        maps from the DPT head.

        Supports both full `2D` input (local+current) and `D` input. A `D` input
        is the Level-0 pre-global local stream, so propagation replays the first
        output layer. A full `2D` input already contains the post-output-layer
        current stream and resumes at the following block.

        Args:
            level0_features: (BV, 256, C) patch features at Level 0. C=2D or D.
            level0_cls: (BV, C) CLS/camera token at Level 0.
            level0_action: (BV, C) action token at Level 0.
            total_view: number of views (V).
            action_head: frozen ActionHeadV2 instance (optional).
            cond_num: number of conditioning views (for action extraction).
            decode_visuals: if False, skip DPT depth/RGB decoding for faster
                closed-loop action rollout.

        Returns:
            Dict with optionally 'depth'/'rgb', 'action_pred', and
            'action_tokens'.
        """
        bv, num_patches, feat_dim = level0_features.shape
        batch_size = bv // total_view
        num_views = total_view
        grid_size = int(num_patches ** 0.5)
        embed_dim = self.embed_dim

        backbone_dtype = next(self.backbone.parameters()).dtype
        level0_features = level0_features.to(backbone_dtype)
        level0_cls = level0_cls.to(backbone_dtype)
        level0_action = level0_action.to(backbone_dtype)

        # Assemble sequence: [CLS | action | patches]
        feats = torch.cat([
            level0_cls.unsqueeze(1),      # (BV, 1, C)
            level0_action.unsqueeze(1),   # (BV, 1, C)
            level0_features,              # (BV, 256, C)
        ], dim=1)  # (BV, 258, C)

        channels = embed_dim * 2  # local+current, always needed for DPT output format
        feats_4d = feats.reshape(batch_size, num_views, feats.shape[1], feat_dim)

        pre_global_seed = feat_dim == embed_dim
        if pre_global_seed:
            # local_x only: Level-0 seed before the first output layer. Initialize
            # current_x from the same seed so DA3 can recreate the current stream.
            local_x = feats_4d
            current_x = feats_4d.clone()
        else:
            # Full 2D: split into local + current
            local_x = feats_4d[..., :embed_dim]
            current_x = feats_4d[..., embed_dim:]

        trans = self.backbone.pretrained
        image_h = image_w = grid_size * self.PATCH_SIZE
        pos_all, pos_nodiff_all = trans._prepare_rope(
            batch_size, num_views, image_h, image_w, level0_features.device
        )

        # Action token needs zero-position in rope (same as injection)
        if pos_all is not None:
            act_pos = torch.zeros(
                batch_size, num_views, 1, 2,
                device=pos_all.device, dtype=pos_all.dtype,
            )
            pos_all = torch.cat([pos_all[:, :, :1], act_pos, pos_all[:, :, 1:]], dim=2)
            pos_nodiff_all = torch.cat(
                [pos_nodiff_all[:, :, :1], act_pos, pos_nodiff_all[:, :, 1:]], dim=2
            )

        token_count = num_patches + 2  # CLS + action + patches
        level_feats = []

        if decode_visuals and not pre_global_seed:
            # Collect native Level 0 features for DPT. For a D-d pre-global
            # seed, Level 0 is collected after replaying the first output layer.
            cs = current_x.reshape(bv, token_count, -1)
            ls = local_x.reshape(bv, token_count, -1)
            # Skip action token (pos 1) for DPT features. DPT applies its own
            # LayerNorm, so match native DA3 raw `[local_x, current_x]` export.
            level_feats.append(
                (
                    torch.cat([ls[:, 2:], cs[:, 2:]], dim=-1).reshape(
                        batch_size, num_views, num_patches, channels
                    ),
                    torch.cat([ls[:, 0], cs[:, 0]], dim=-1).reshape(
                        batch_size, num_views, channels
                    ),
                )
            )

        start_block = self.out_layers[0] if pre_global_seed else self.out_layers[0] + 1
        for i, blk in enumerate(trans.blocks):
            if i < start_block:
                continue
            if i < trans.rope_start or trans.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos, l_pos = pos_nodiff_all, pos_all
            if trans.alt_start != -1 and i == trans.alt_start:
                camera_tokens = self._build_camera_tokens(
                    batch_size, num_views, current_x.device, current_x.dtype
                )
                current_x[:, :, 0] = camera_tokens
            if trans.alt_start != -1 and i >= trans.alt_start and i % 2 == 1:
                attn_type, pos_emb = "global", g_pos
            else:
                attn_type, pos_emb = "local", l_pos
            current_x = trans.process_attention(
                current_x, blk, attn_type=attn_type, pos=pos_emb
            )
            if attn_type == "local":
                local_x = current_x
            if decode_visuals and i in self.out_layers:
                cs = current_x.reshape(bv, token_count, -1)
                ls = local_x.reshape(bv, token_count, -1)
                level_feats.append(
                    (
                        torch.cat([ls[:, 2:], cs[:, 2:]], dim=-1).reshape(
                            batch_size, num_views, num_patches, channels
                        ),
                        torch.cat([ls[:, 0], cs[:, 0]], dim=-1).reshape(
                            batch_size, num_views, channels
                        ),
                    )
                )

        result: Dict[str, torch.Tensor] = {}

        # DPT depth decoding (same as propagate_and_decode)
        if decode_visuals and len(level_feats) >= 4:
            self._prepare_dpt_head_for_float_decode()
            with torch.autocast(device_type=level0_features.device.type, enabled=False):
                feats_float = self._norm_feats_for_dpt(
                    [(p.float(), c.float()) for p, c in level_feats]
                )
                dpt_output = self.dpt_head(
                    feats_float, image_h, image_w, patch_start_idx=0
                )
            for key in ["depth", "depth_conf", "ray", "ray_conf", "rgb"]:
                if key in dpt_output and dpt_output[key] is not None:
                    result[key] = dpt_output[key]

        # Action token extraction at final layer
        cs_final = current_x.reshape(bv, token_count, -1)
        action_normed = trans.norm(cs_final[:, 1])  # action at pos 1, (BV, D)
        result["action_tokens"] = action_normed

        if action_head is not None:
            # Reshape: (B, V, D) -> take ext views of future timesteps
            action_bv = action_normed.reshape(batch_size, total_view, -1)
            ext_indices = list(range(cond_num, total_view, 2))
            action_per_step = action_bv[:, ext_indices]  # (B, T_future, D)
            action_pred = action_head(action_per_step)
            result["action_pred"] = action_pred

        return result

    @torch.no_grad()
    def propagate_and_predict(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor,
        level0_action: torch.Tensor,
        total_view: int = 1,
        action_head: Optional[nn.Module] = None,
        cond_num: int = 2,
        decode_visuals: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Monitor/eval path: propagate Level-0 slots without gradients."""
        return self._propagate_and_predict_impl(
            level0_features,
            level0_cls,
            level0_action,
            total_view=total_view,
            action_head=action_head,
            cond_num=cond_num,
            decode_visuals=decode_visuals,
        )

    def propagate_and_predict_grad(
        self,
        level0_features: torch.Tensor,
        level0_cls: torch.Tensor,
        level0_action: torch.Tensor,
        total_view: int = 1,
        action_head: Optional[nn.Module] = None,
        cond_num: int = 2,
        decode_visuals: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Training path: differentiable Level-0 slot propagation."""
        return self._propagate_and_predict_impl(
            level0_features,
            level0_cls,
            level0_action,
            total_view=total_view,
            action_head=action_head,
            cond_num=cond_num,
            decode_visuals=decode_visuals,
        )
