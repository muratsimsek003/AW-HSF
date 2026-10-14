import sys, os, time, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
ROOT=Path('/mnt/data/AW_HSF_Tamamlanan')
sys.path.insert(0,str(ROOT/'submission_update/code'))
import awhsf_scratch as m
CKPT=Path('/mnt/data/mobilenet_v3_small-047dcff4.pth')
assert hashlib.sha256(CKPT.read_bytes()).hexdigest()=='047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f', 'Model weights SHA mismatch'
torch.set_num_threads(2)
class LocalFrozenDeep:
 def __init__(self,device,pretrained=True):
  self.device=device
  self.net=mobilenet_v3_small(weights=None)
  self.net.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True))
  self.net=self.net.features.eval().to(device)
  for p in self.net.parameters():p.requires_grad_(False)
  self.avg=nn.AdaptiveAvgPool2d(1)
  self.tx=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
 @torch.inference_mode()
 def batch(self,ims,batch_size=32):
  vectors=[]
  for off in range(0,len(ims),batch_size):
   x=torch.stack([self.tx(im) for im in ims[off:off+batch_size]]).to(self.device)
   vectors.append(self.avg(self.net(x)).flatten(1).cpu().numpy())
  return np.concatenate(vectors).astype(np.float32)
m.FrozenDeep=LocalFrozenDeep
ap=argparse.ArgumentParser()
ap.add_argument('--datasets',nargs='+',default=['skin','botox']);ap.add_argument('--curves',action='store_true');ap.add_argument('--mode',choices=['full','pilot'],default='full')
a=ap.parse_args()
out=Path('/mnt/data/AW_HSF_Yeniden_Egitim_20261014');out.mkdir(exist_ok=True)
for which in a.datasets:
 df=pd.read_csv(ROOT/f'analysis/{which}_curated_manifest.csv')
 df['path']=df.path.map(lambda x:str(ROOT/x))
 assert df.path.map(lambda x:Path(x).exists()).all()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==hs for p,hs in zip(df.path,df.sha256))
 print(f'{which}: {len(df)} verified original images; {df.group.nunique()} groups; labels {df.label.value_counts().to_dict()}',flush=True)
 start=time.time()
 features=m.make_features(df,which,out/'feature_cache',aug_count=14 if which=='botox' else 0,pretrained=True,device='cpu',batch_size=24)
 print(f'{which} features: h{features["h"].shape} d{features["d"].shape}; elapsed={time.time()-start:.1f}s',flush=True)
 (out/which).mkdir(exist_ok=True,parents=True)
 df.drop(columns=['path']).to_csv(out/which/'curated_public_manifest_no_local_paths.csv',index=False)
 if a.mode=='pilot':
  seeds=[42,7];lambdas=[0.,.05];folds=3;epochs=60
 else:
  seeds=m.SEEDS;lambdas=m.LAMBDAS;folds=5;epochs=100
 s=m.trial(df,features,which,out,seeds=seeds,lambdas=lambdas,folds=folds,max_epochs=epochs,device='cpu')
 print(f'{which} full trial done {time.time()-start:.1f}s',flush=True)
 print(s.to_string(index=False),flush=True)
 if a.curves:
  m.run_learning_curve(df,features,which,out,seeds=[42,7,123],epochs=80,device='cpu')
  print(f'{which} curves done {time.time()-start:.1f}s',flush=True)
print('ALL DONE',flush=True)
