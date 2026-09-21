#!/usr/bin/env python3
"""Run a mobile GAM bundle on an explicit observed-history NPZ; write predictions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--instruction',required=True)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--stats-key',default='agibot_eef_relative_joint16')
    p.add_argument('--text-model-path',type=Path)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists; choose a new file')
    import numpy as np
    from gam.mobile import MobileGAMPolicy
    with np.load(a.input,allow_pickle=False) as data:
        allowed={'images','states','previous_actions','previous_action_mask','view_mask','anchor_poses'}
        if not {'images','states'}<=set(data.files) or not set(data.files)<=allowed:
            p.error('NPZ requires images/states and accepts only documented history fields')
        inputs={key:data[key] for key in data.files}
    policy=MobileGAMPolicy.from_pretrained(a.bundle,device=a.device,text_model_path=a.text_model_path)
    result=policy.predict(**inputs,instruction=a.instruction,stats_key=a.stats_key)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as stream:np.savez_compressed(stream,**result)
    print(f"Saved actions{result['actions'].shape} to {a.output}")


if __name__=='__main__':main()
