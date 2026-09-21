"""Stateless observed-history inference, preserving the complete causal prefix."""
from __future__ import annotations
from contextlib import nullcontext
import numpy as np
from PIL import Image
import torch
from .bundle import load_bundle
from .contract import AGIBOT_KEY, embodiment_for_key
from .geometry import recover_targets
from .normalization import Normalizer


def prepare_images(images, views):
    """RGB uint8 H,V,height,width,3 -> float32 H,V,3,224,224, Pillow bilinear."""
    x = np.asarray(images)
    if x.dtype != np.uint8 or x.ndim != 5 or x.shape[1] != 3 or x.shape[-1] != 3:
        raise ValueError('Expected RGB uint8 images(H,3,height,width,3)')
    result = np.zeros((len(x), 3, 3, 224, 224), dtype=np.float32)
    for t in range(len(x)):
        for v in range(3):
            if views[t,v]:
                im = Image.fromarray(x[t,v]).resize((224,224), Image.Resampling.BILINEAR)
                result[t,v] = np.asarray(im).transpose(2,0,1).astype(np.float32)/255.
    return torch.from_numpy(result)


class MobileGAMPolicy:
    @classmethod
    def from_pretrained(cls, bundle, *, device='cuda', text_model_path=None):
        """Load a local exported bundle; optionally use a local copy of pinned T5."""
        from robot.modeling.da3_giant_encoder import DA3GiantEncoder
        from robot.modeling.future_predictor import build_future_predictor
        from robot.modeling.action_head_v2 import ActionHeadV2
        from transformers import AutoTokenizer, T5EncoderModel
        config, stats, weights = load_bundle(bundle)
        student = DA3GiantEncoder(ckpt_path=None, initialize_from_checkpoint=False,
                                 freeze_backbone=True, **config['backbone'])
        predictor = build_future_predictor(config['predictor'])
        head_config = dict(config['action_head']);head_config.pop('type', None)
        head = ActionHeadV2(n_views=3, **head_config)
        for name, module in (('student_da3',student),('future_predictor',predictor),('action_head',head)):
            module.load_state_dict(weights[name], strict=True)
            module.eval().requires_grad_(False).to(device)
        del weights
        text_id = str(text_model_path) if text_model_path else config['text_model']
        kwargs = {} if text_model_path else {'revision':config['text_revision']}
        tokenizer = AutoTokenizer.from_pretrained(text_id, **kwargs)
        text_encoder = T5EncoderModel.from_pretrained(text_id, **kwargs).eval().requires_grad_(False).to(device)
        obj = cls()
        obj.config, obj.device = config, torch.device(device)
        obj.student, obj.predictor, obj.head = student, predictor, head
        obj.tokenizer, obj.text_encoder = tokenizer, text_encoder
        obj.action_normalizer, obj.state_normalizer = Normalizer(stats['action']), Normalizer(stats['state'])
        obj._text_cache = {}
        return obj

    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()

    def _language(self, instruction):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Provide a nonempty instruction; text is not silently rewritten')
        if instruction not in self._text_cache:
            tokens = self.tokenizer([instruction], return_tensors='pt', padding='max_length',
                                    truncation=True, max_length=self.config['text_length']).to(self.device)
            # Frozen T5 inference runs in fp32 outside the policy autocast region.
            output = self.text_encoder(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'])
            if len(self._text_cache) >= 32:
                self._text_cache.pop(next(iter(self._text_cache)))
            self._text_cache[instruction] = (output.last_hidden_state, tokens['attention_mask'].bool())
        return self._text_cache[instruction]

    @torch.inference_mode()
    def predict(self, *, images, states, instruction, stats_key=AGIBOT_KEY,
                previous_actions=None, previous_action_mask=None, view_mask=None,
                anchor_poses=None):
        """Predict the current C16 chunk from H1..4 real observed anchors.

        states: raw(H,16). previous_actions: raw(H,16,16), one previous chunk
        per observed anchor, in THAT previous chunk's own reference frame.
        previous_action_mask: bool(H,16,16); unavailable entries become zero
        AFTER normalization. Omitting history is allowed only for episode-start H1.
        anchor_poses: current measured flange poses(2,7), xyz+xyzw, AgiBot only.
        """
        embodiment = embodiment_for_key(stats_key)
        h = len(images)
        if not 1 <= h <= 4:
            raise ValueError('Supply 1..4 real observed anchors; do not fabricate padded frames')
        views = np.ones((h,3), dtype=bool) if view_mask is None else np.asarray(view_mask)
        if views.dtype != np.bool_ or views.shape != (h,3) or not views.any(axis=1).all():
            raise ValueError('view_mask must be bool(H,3), with at least one camera per anchor')
        state = torch.as_tensor(np.asarray(states),dtype=torch.float32)
        if state.shape != (h,16) or not torch.isfinite(state).all():
            raise ValueError('Expected finite raw states(H,16)')
        if embodiment == 'oxe' and torch.count_nonzero(state[:,7:]):
            raise ValueError('OXE state must occupy 0:7 with zero padding 7:16')
        if previous_actions is None:
            if h != 1 or previous_action_mask is not None:
                raise ValueError('Missing previous chunks are allowed only for episode-start H1')
            previous = torch.zeros(h,16,16); keep = torch.zeros(h,16,16,dtype=torch.bool)
        else:
            previous = torch.as_tensor(np.asarray(previous_actions),dtype=torch.float32)
            raw_keep = np.asarray(previous_action_mask)
            if raw_keep.dtype != np.bool_ or raw_keep.shape != (h,16,16):
                raise ValueError('Provide explicit bool previous_action_mask(H,16,16)')
            keep = torch.from_numpy(raw_keep.copy())
            if previous.shape != (h,16,16) or not torch.isfinite(previous).all():
                raise ValueError('Expected finite raw previous_actions(H,16,16)')
        if embodiment == 'oxe':
            absent = list(range(7))+[14,15]
            if torch.count_nonzero(previous[...,absent]) or keep[...,absent].any():
                raise ValueError('OXE previous actions must be zero/masked outside slots 7:14')
        state_norm = self.state_normalizer.transform(state,stats_key).to(self.device)[None]
        previous_norm = self.action_normalizer.transform(previous,stats_key)
        previous_norm = torch.where(keep,previous_norm,0.).to(self.device)[None]
        rgb = prepare_images(images, views).to(self.device).reshape(1,h*3,3,224,224)
        rgb = (rgb-self.student.encoder_mean.float())/self.student.encoder_std.float()
        valid = torch.from_numpy(views.copy()).to(self.device)[None]
        language, language_mask = self._language(instruction)
        with self._autocast():
            shallow = self.student.encode_shallow_visual_slots(rgb,T=h,V=3)['visual_tokens']
            predicted = self.predictor(past_visual_tokens=shallow,proprio=state_norm[:,-1],
                proprio_history=state_norm,past_action_history=previous_norm,
                lang_feats=language,lang_padding_mask=language_mask,view_valid_mask=valid)
            visual = predicted['predicted_next_visual_tokens']*valid[...,None,None]
            action_tokens = predicted['predicted_action_tokens']*valid[...,None]
            refined = self.student.propagate_shallow_with_actions(visual,action_tokens,
                decode_visuals=False,deep_temporal_causal_mask=True)
            tokens = refined['action_tokens'].reshape(1,h,3,-1)*valid[...,None]
            # Preserve the full observed prefix in both predictor and deep stack.
            normalized = self.head(tokens,view_mask=valid)[0,-1].float()
        actions = self.action_normalizer.transform(normalized,stats_key,inverse=True)
        if not torch.isfinite(actions).all():
            raise RuntimeError('Model returned nonfinite actions')
        if embodiment == 'oxe':
            actions[:,list(range(7))+[14,15]] = 0.
        result = {'actions':actions.cpu().numpy(),'normalized_actions':normalized.cpu().numpy(),
                  'history_length':h,'stats_key':stats_key}
        if anchor_poses is not None:
            if embodiment != 'agibot':
                raise ValueError('Bimanual anchor recovery requires AgiBot statistics')
            result.update(recover_targets(result['actions'],anchor_poses))
        return result
