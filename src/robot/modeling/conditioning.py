"""Text and proprioception conditioning modules for GLD-Robot.

`TextConditioner` dispatches between CLIP and T5 frozen encoders. CLIP is the
historical default and is kept bit-identical to the prior single-backend
implementation (the `.clip` / `.tokenizer` attributes are preserved for the
legacy DiT call site in `train_robot.py:1736`). T5 is added so the predictor
can swap to FastWAM-style language conditioning without changing callers.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List


class TextConditioner(nn.Module):
    """Frozen text encoder (CLIP or T5) with learnable projection.

    Args:
        encoder_type: "clip" (default) or "t5".
        clip_model: HuggingFace CLIP model name (used when encoder_type="clip").
        t5_model: HuggingFace T5 model name (used when encoder_type="t5").
            Recommended: "google-t5/t5-base" (768d, 12L) for parity with CLIP-L
            text dim. "google/flan-t5-large" (1024d) for bigger language
            capacity. "google/t5-v1_1-xxl" (4096d) for FastWAM scale (heavy).
        proj_dim: Output projection dimension (matches DiT encoder_hidden_size
            for the legacy pooled forward path; predictor uses `encode_tokens`
            which bypasses this projection).
    """

    def __init__(
        self,
        encoder_type: str = "clip",
        clip_model: str = "openai/clip-vit-large-patch14",
        t5_model: str = "google-t5/t5-base",
        proj_dim: int = 768,
        cache_token_embeddings: bool = False,
        cache_max_entries: int = 0,
        cache_device: str = "cpu",
    ):
        super().__init__()

        encoder_type = str(encoder_type).strip().lower()
        if encoder_type == "clip":
            from transformers import CLIPTextModel, CLIPTokenizer
            self.encoder = CLIPTextModel.from_pretrained(clip_model)
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            self.hidden_size = int(self.encoder.config.hidden_size)
            # Backward-compat alias: legacy DiT call site (train_robot.py:1736)
            # accesses `text_conditioner.clip(...)` directly.
            self.clip = self.encoder
        elif encoder_type == "t5":
            from transformers import T5EncoderModel, AutoTokenizer
            self.encoder = T5EncoderModel.from_pretrained(t5_model)
            self.tokenizer = AutoTokenizer.from_pretrained(t5_model)
            self.hidden_size = int(self.encoder.config.d_model)
            self.clip = None
        else:
            raise ValueError(
                f"Unsupported TextConditioner.encoder_type={encoder_type!r}. "
                "Expected 'clip' or 't5'."
            )

        self.encoder_type = encoder_type
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self.hidden_size, proj_dim)
        self.cache_token_embeddings = bool(cache_token_embeddings)
        self.cache_max_entries = max(0, int(cache_max_entries))
        self.cache_device = str(cache_device).strip().lower()
        self._token_cache = OrderedDict()

    def _cache_storage_device(self):
        if self.cache_device in {"cuda", "gpu", "device"}:
            return self.proj.weight.device
        return torch.device("cpu")

    def set_token_cache(
        self,
        *,
        enabled: bool,
        max_entries: int,
        cache_device: str = "cpu",
    ) -> None:
        self.cache_token_embeddings = bool(enabled)
        self.cache_max_entries = max(0, int(max_entries))
        self.cache_device = str(cache_device).strip().lower()
        self._token_cache.clear()

    def _tokenize(self, text_list: List[str], pad_to: int, padding: str = "max_length"):
        kwargs = dict(
            return_tensors="pt",
            padding=padding,
            truncation=True,
            max_length=pad_to,
        )
        return self.tokenizer(text_list, **kwargs).to(self.proj.weight.device)

    def forward(self, text_list: List[str]) -> torch.Tensor:
        """Encode text descriptions to a single pooled (B, proj_dim) embedding.

        Used by the legacy DiT path. CLIP uses `pooler_output`; T5 emits
        masked-mean over `last_hidden_state` (T5 has no pooler head).
        """
        tokens = self._tokenize(text_list, pad_to=77, padding=True)

        with torch.no_grad():
            if self.encoder_type == "clip":
                feat = self.encoder(**tokens).pooler_output  # (B, hidden_size)
            else:
                last_hidden = self.encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).last_hidden_state  # (B, L, hidden_size)
                mask = tokens["attention_mask"].to(last_hidden.dtype).unsqueeze(-1)
                feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.proj(feat.to(self.proj.weight.dtype))

    def encode_tokens(self, text_list: List[str], pad_to: int = 77) -> Dict[str, torch.Tensor]:
        """Return per-token last_hidden_state (for token-prepend / cross-attn).

        The caller is responsible for projecting hidden_size -> predictor d_model.

        Returns:
            dict with `last_hidden_state` (B, pad_to, hidden_size) and
            `attention_mask` (B, pad_to) bool.
        """
        if self.cache_token_embeddings and self.cache_max_entries > 0:
            return self._encode_tokens_cached(text_list, pad_to=pad_to)

        tokens = self._tokenize(text_list, pad_to=pad_to, padding="max_length")
        with torch.no_grad():
            if self.encoder_type == "clip":
                last_hidden = self.encoder(**tokens).last_hidden_state
            else:
                last_hidden = self.encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).last_hidden_state
        return {
            "last_hidden_state": last_hidden,
            "attention_mask": tokens["attention_mask"].bool(),
        }

    def _encode_tokens_cached(self, text_list: List[str], pad_to: int) -> Dict[str, torch.Tensor]:
        device = self.proj.weight.device
        storage_device = self._cache_storage_device()
        keys = [(self.encoder_type, int(pad_to), str(text)) for text in text_list]
        missing_texts = []
        missing_keys = []

        for key in keys:
            if key in self._token_cache:
                self._token_cache.move_to_end(key)
            else:
                missing_keys.append(key)
                missing_texts.append(key[2])

        if missing_texts:
            tokens = self._tokenize(missing_texts, pad_to=pad_to, padding="max_length")
            with torch.no_grad():
                if self.encoder_type == "clip":
                    encoded = self.encoder(**tokens).last_hidden_state
                else:
                    encoded = self.encoder(
                        input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"],
                    ).last_hidden_state
            masks = tokens["attention_mask"].bool()
            for idx, key in enumerate(missing_keys):
                self._token_cache[key] = (
                    encoded[idx].detach().to(storage_device),
                    masks[idx].detach().to(storage_device),
                )
                self._token_cache.move_to_end(key)
            while len(self._token_cache) > self.cache_max_entries:
                self._token_cache.popitem(last=False)

        last_hidden = []
        attention_mask = []
        for key in keys:
            hidden_i, mask_i = self._token_cache[key]
            self._token_cache.move_to_end(key)
            last_hidden.append(hidden_i.to(device=device, non_blocking=True))
            attention_mask.append(mask_i.to(device=device, non_blocking=True))
        return {
            "last_hidden_state": torch.stack(last_hidden, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }


class ProprioConditioner(nn.Module):
    """Proprioception MLP: encodes robot state to conditioning vector.

    Default: eef_pos(3) + eef_quat(4) + gripper_qpos(2) = 9D.

    Args:
        proprio_dim: Input proprioception dimension.
        hidden_dim: MLP hidden dimension.
        out_dim: Output dimension (matches DiT encoder_hidden_size).
    """

    def __init__(self, proprio_dim: int = 9, hidden_dim: int = 256, out_dim: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception.

        Args:
            proprio: (B, proprio_dim) robot state vector.

        Returns:
            (B, out_dim) proprioception embedding.
        """
        return self.mlp(proprio)
