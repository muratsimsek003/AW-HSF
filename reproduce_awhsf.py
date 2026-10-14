#!/usr/bin/env python3
"""Portable, independently implemented AW-HSF reproduction.

Usage example:
  python code/reproduce_awhsf.py --images-root /path/to/extracted \
    --manifests-root evidence --checkpoint /path/to/mobilenet_v3_small-047dcff4.pth \
    --out /path/to/new-results --curves

The image root must contain paths matching the relative filenames in the provided
curated manifests (e.g., data/skin/acne/acne/acne1.png). The Kaggle images are NOT
redistributed. Subjects in the skin dataset are not identified; pHash groups are
only image-similarity-based leakage controls.
"""
from __future__ import annotations
import argparse,hashlib,sys,json,logging
from pathlib import Path
import pandas as pd,numpy as np,torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
sys.path.insert(0,str(Path(__file__).resolve().parent))
import awhsf_scratch as model

EXPECTED='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 ap.add_argument('--images-root',type=Path,required=True)
 ap.add_argument('--manifests-root',type=Path,required=True)
 ap.add_argument('--checkpoint',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--datasets',choices=['skin','botox'],nargs='+',default=['skin','botox'])
 ap.add_argument('--curves',action='store_true')
 ap.add_argument('--num-threads',type=int,default=2)
 args=ap.parse_args()
 logging.basicConfig(level=logging.INFO)
 torch.set_num_threads(args.num_threads)
 sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
 if sha!=EXPECTED: raise ValueError('Checkpoint SHA-256 mismatch. Expected '+EXPECTED+'; got '+sha)

 class PretrainedLocal:
  def __init__(self,device,pretrained=True):
   if not pretrained:raise ValueError('This study requires pretrained MobileNetV3-Small weights')
   self.device=device
   net=mobilenet_v3_small(weights=None)
   net.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True),strict=True)
   self.net=net.features.eval().to(device)
   for p in self.net.parameters():p.requires_grad_(False)
   self.avg=nn.AdaptiveAvgPool2d(1)
   self.tx=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                               transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
  @torch.inference_mode()
  def batch(self,ims,batch_size=32):
   v=[]
   for k in range(0,len(ims),batch_size):
    x=torch.stack([self.tx(im) for im in ims[k:k+batch_size]]).to(self.device)
    v.append(self.avg(self.net(x)).flatten(1).cpu().numpy())
   return np.concatenate(v).astype(np.float32)
 model.FrozenDeep=PretrainedLocal
 args.out.mkdir(parents=True,exist_ok=True)
 for which in args.datasets:
  manifest=args.manifests_root/which/'curated_manifest_relative.csv'
  df=pd.read_csv(manifest)
  df['path']=df['path'].map(lambda x:str(args.images_root/x))
  for _,r in df.iterrows():
   p=Path(r['path'])
   if not p.exists():raise FileNotFoundError(str(p))
   if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError(f'Image changed since audit: {p}')
  feats=model.make_features(df,which,args.out/'feature_cache',aug_count=14 if which=='botox' else 0,
                            pretrained=True,device='cpu',batch_size=24)
  scores=model.trial(df,feats,which,args.out,seeds=model.SEEDS,lambdas=model.LAMBDAS,
                     folds=5,max_epochs=100,device='cpu')
  print(which,scores.groupby('method')[['macro_f1','ECE_10','Brier_multiclass']].mean().to_string())
  if args.curves:
   model.run_learning_curve(df,feats,which,args.out,seeds=[42,7,123],epochs=80,device='cpu')
 print('Complete:',args.out)
if __name__=='__main__':main()
