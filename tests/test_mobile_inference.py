"""Numerical and boundary tests; run in a CPU/compute Python environment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from gam.mobile.geometry import (recover_targets, encode_targets, joint_state,
    agibot_measured_closure, pad_oxe_actions, pad_oxe_state)
from gam.mobile.normalization import Normalizer
from gam.mobile.policy import MobileGAMPolicy, prepare_images
from gam.mobile.bundle import export_checkpoint, load_bundle
from gam.mobile.contract import AGIBOT_KEY, release_config


def stats():
    return {'norm_mode':'q01_q99','eps':1e-8,'stats_by_key':{AGIBOT_KEY:
        {'q01':[-2.]*16,'q99':[4.]*16,'mask':[True]*6+[False]+[True]*6+[False]*3}}}


def training_config():
    return {'dataset':{'action_schema':'agibot_oxe_eef_relative_right16_v1',
        'state_schema':'left_joint7_gripper_right_joint7_gripper_v1','action_frame':'eef_relative',
        'chunk_size':16,'image_size':[224,224],'secret_private_path':'/private/do-not-export'},
        'stage_1':{},'proprioception':{'proprio_dim':16},
        'da3_finetune':{'n_views':3,'n_action_steps':5},
        'predictor':{'enabled':True,'type':'shallow12_ar','shared_padded_proprio':True,
          'H_choices':[1,2,3,4],'deep_temporal_causal_mask':True,'language_encoder_type':'t5',
          't5_model':'google-t5/t5-base','d_model':1024,'depth':12,'num_heads':16,
          'ffn_ratio':4.,'dropout':0.,'num_patches_per_view':256,'num_register_tokens':0,
          'use_language':True,'language_dim':768,'language_len':77,'condition_mode':'concat','input_proj_norm':'ln'},
        'action_head':{'type':'mlp_resnet','n_dims':16,'chunk_size':16,'input_dim':1536,
          'hidden_dim':1536,'num_blocks':2,'pool_mode':'mean','chunk_position_encoding':'none'}}


class GeometryTests(unittest.TestCase):
    def test_fixed_anchor_rotation_and_inverse(self):
        anchor=np.zeros((2,7));anchor[:,:3]=[[1.,2.,3.],[-2.,1.,4.]]
        anchor[:,3:]=Rotation.from_euler('z',[90.,-90.],degrees=True).as_quat()
        a=np.zeros((16,16));a[:,0]=.2;a[:,7]=.3;a[:,5]=.1;a[:,12]=-.2
        a[:,6]=.4;a[:,13]=.8;a[:,14]=1.2;a[:,15]=-.6
        out=recover_targets(a,anchor)
        np.testing.assert_allclose(out['flange_poses'][0,:,:3],[[1.,2.2,3.],[-2.,.7,4.]],atol=1e-6)
        # Every identical target stays identical; no repeated integration.
        np.testing.assert_allclose(out['flange_poses'][0],out['flange_poses'][-1])
        encoded=encode_targets(out['flange_poses'],anchor,out['gripper_closure'],out['base_velocity'])
        np.testing.assert_allclose(encoded,a,atol=1e-6)

    def test_distinct_chunk_anchors_near_pi(self):
        rng=np.random.default_rng(42)
        anchor=np.c_[rng.normal(size=(2,3)),Rotation.random(2,random_state=rng).as_quat()]
        a=rng.normal(size=(16,16))*.1;a[:,3:6]=0.;a[:,3]=np.pi-1e-5
        first=recover_targets(a,anchor); shifted=anchor.copy();shifted[:,:3]+=[1,2,3]
        second=recover_targets(a,shifted)
        np.testing.assert_allclose(second['flange_poses'][...,:3]-first['flange_poses'][...,:3],
                                   np.broadcast_to([1,2,3],(16,2,3)),atol=1e-6)
        np.testing.assert_allclose(encode_targets(first['flange_poses'],anchor,
             first['gripper_closure'],first['base_velocity']),a,atol=2e-6)

    def test_state_slots_and_calibration(self):
        x=joint_state(np.arange(7),np.arange(7)+10,.2,.8)
        np.testing.assert_equal(x[:7],np.arange(7));np.testing.assert_equal(x[8:15],np.arange(7)+10)
        np.testing.assert_allclose(x[[7,15]],[.2,.8])
        np.testing.assert_allclose(agibot_measured_closure([35,77.5,120]),[0,.5,1])
        with self.assertRaises(ValueError):joint_state([0]*6,[0]*7,0,1)
        bad=np.zeros((2,7))
        with self.assertRaises(ValueError):recover_targets(np.zeros((16,16)),bad)

    def test_oxe_padding(self):
        state=pad_oxe_state(np.arange(7));action=pad_oxe_actions(np.arange(7))
        np.testing.assert_equal(state[:7],np.arange(7));self.assertEqual(state[7:].sum(),0)
        np.testing.assert_equal(action[7:14],np.arange(7));self.assertEqual(action[:7].sum()+action[14:].sum(),0)


class NormalizationTests(unittest.TestCase):
    def test_roundtrip_identity_mask_and_missing_actions(self):
        n=Normalizer(stats());x=torch.randn(2,16,16)
        normalized=n.transform(x,AGIBOT_KEY)
        torch.testing.assert_close(n.transform(normalized,AGIBOT_KEY,inverse=True),x)
        torch.testing.assert_close(normalized[...,[6,13,14,15]],x[...,[6,13,14,15]])
        self.assertFalse(torch.equal(n.transform(torch.zeros_like(x),AGIBOT_KEY),torch.zeros_like(x)))
        with self.assertRaises(KeyError):n.transform(x,'unknown')

    def test_constant_dimensions(self):
        s=stats();s['stats_by_key'][AGIBOT_KEY]['q99'][0]=-2.
        x=torch.ones(16);n=Normalizer(s)
        self.assertEqual(n.transform(x,AGIBOT_KEY)[0],1.)


class ContractTests(unittest.TestCase):
    def test_whitelist_and_incompatible_checkpoints(self):
        cfg=training_config();out=release_config(cfg,7000)
        self.assertNotIn('private',json.dumps(out));self.assertEqual(out['predictor']['proprio_dim'],16)
        self.assertEqual(out['backbone']['action_steps_per_token'],16)
        for section,key,value in [('predictor','shared_padded_proprio',False),
            ('predictor','single_arm_proprio_dim',7),('dataset','action_frame','base'),
            ('predictor','deep_temporal_causal_mask',False),('action_head','chunk_size',8)]:
            bad=copy.deepcopy(cfg);bad[section][key]=value
            with self.assertRaises(ValueError):release_config(bad,7000)

    def test_export_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw={'config':training_config(),'train_steps':7000,
                **{k:{'_orig_mod.weight':torch.ones(2)} for k in ('student_da3','future_predictor','action_head')},
                'action_normalizer':stats(),'proprio_normalizer':stats(),'optimizer':{'private':'omitted'}}
            torch.save(raw,p/'source.pt')
            export_checkpoint(p/'source.pt',p/'bundle',text_revision='a'*40)
            cfg,norm,weights=load_bundle(p/'bundle')
            self.assertEqual(set(weights['student_da3']),{'weight'})
            self.assertNotIn('optimizer',weights)
            with self.assertRaises(FileExistsError):export_checkpoint(p/'source.pt',p/'bundle',text_revision='a'*40)
            (p/'bundle/config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):load_bundle(p/'bundle')


class PipelineTests(unittest.TestCase):
    def test_deep_attention_matches_qk_and_value_dtypes(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from robot.modeling import da3_giant_encoder as encoder

        # Reproduce eager QK normalization promoting bf16 inputs to fp32.
        identity = torch.nn.Identity()
        attention = SimpleNamespace(
            qkv=lambda x: torch.cat([x, x, x], dim=-1), num_heads=1,
            q_norm=lambda x: x.float(), k_norm=lambda x: x.float(),
            rope=None, proj=identity, proj_drop=identity,
        )
        block = SimpleNamespace(norm1=identity, attn=attention, ls1=identity,
                                norm2=identity, mlp=identity, ls2=identity)

        def check_attention(q, k, v, *, block_mask):
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertEqual(k.dtype, v.dtype)
            return v

        x = torch.ones(1, 2, 3, 4, dtype=torch.bfloat16)
        with patch.object(encoder, '_HAS_FLEX_ATTENTION', True), \
                patch.object(encoder, '_flex_attention', check_attention):
            out = encoder.DA3GiantEncoder._run_deep_global_block_flex(
                None, x, block, None, None)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def fake_policy(self):
        p=MobileGAMPolicy();p.device=torch.device('cpu');p.config={'text_length':77}
        p.action_normalizer=p.state_normalizer=Normalizer(stats())
        class Encoder:
            encoder_mean=torch.zeros(1,1,3,1,1);encoder_std=torch.ones(1,1,3,1,1)
            def encode_shallow_visual_slots(self,images,T,V):
                self.h=T;return {'visual_tokens':torch.ones(1,T,V,1,2)}
            def propagate_shallow_with_actions(self,visual,action,**kw):
                self.deep_h=visual.shape[1];self.kw=kw
                return {'action_tokens':action}
        p.student=Encoder()
        def predictor(**kw):
            p.inputs=kw;h=kw['proprio_history'].shape[1]
            tokens=torch.arange(h).reshape(1,h,1,1).expand(1,h,3,2).float()
            return {'predicted_next_visual_tokens':kw['past_visual_tokens'],'predicted_action_tokens':tokens}
        p.predictor=predictor
        p.head=lambda tokens,view_mask: tokens.mean(dim=(2,3))[...,None,None].expand(1,tokens.shape[1],16,16)
        p._language=lambda instruction:(torch.zeros(1,77,768),torch.ones(1,77,dtype=torch.bool))
        return p

    def test_complete_h4_prefix_masking_and_current_slot(self):
        p=self.fake_policy();images=np.zeros((4,3,12,10,3),dtype=np.uint8)
        view=np.ones((4,3),dtype=bool);view[:,2]=False
        out=p.predict(images=images,states=np.zeros((4,16)),instruction='pick',
            previous_actions=np.ones((4,16,16)),previous_action_mask=np.zeros((4,16,16),bool),view_mask=view)
        self.assertEqual(p.student.deep_h,4);self.assertTrue(p.student.kw['deep_temporal_causal_mask'])
        self.assertEqual(torch.count_nonzero(p.inputs['past_action_history']),0)
        # Last slot3, two valid views, one masked: mean is2 with this fake head.
        np.testing.assert_equal(out['normalized_actions'],np.full((16,16),2.))
        self.assertFalse(p.inputs['view_valid_mask'][...,2].any())

    def test_missing_history_rejected_and_rgb_contract(self):
        p=self.fake_policy()
        with self.assertRaises(ValueError):p.predict(images=np.zeros((2,3,224,224,3),np.uint8),states=np.zeros((2,16)),instruction='pick')
        image=np.zeros((1,3,224,224,3),np.uint8);image[...,0]=255
        tensor=prepare_images(image,np.array([[True,False,True]]))
        self.assertEqual(tensor[0,0,0,0,0],1.);self.assertEqual(tensor[0,0,2,0,0],0.)
        self.assertEqual(tensor[0,1].sum(),0.)
        with self.assertRaises(ValueError):prepare_images(image.astype(float),np.ones((1,3),bool))


if __name__=='__main__':unittest.main()
