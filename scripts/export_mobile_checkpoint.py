#!/usr/bin/env python3
"""Export an owned/trusted training checkpoint; does not upload weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--text-revision',required=True,help='Exact 40-character google-t5/t5-base commit used in training')
    p.add_argument('--trust-training-checkpoint',action='store_true',help='Acknowledge that this owned training .pt uses pickle')
    args=p.parse_args()
    if not args.trust_training_checkpoint:
        p.error('Export accepts only trusted training files; pass --trust-training-checkpoint')
    from gam.mobile.bundle import export_checkpoint
    export_checkpoint(args.checkpoint,args.output,text_revision=args.text_revision)
    print(f'Exported portable bundle: {args.output}')


if __name__=='__main__':main()
