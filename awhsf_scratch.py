#!/usr/bin/env python3
"""From-scratch, prospective AW-HSF reproducibility experiment.

Downloads the TWO datasets cited in the manuscript via KaggleHub/Kaggle; no
original experiment code, trained weights, or predictions are used. This is
an independent implementation, not a claim of numerical replication.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import random
import shutil
import tarfile
import zipfile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

LOG = logging.getLogger("AWHSF")
SEEDS = [42, 7, 123, 2024, 91]
LAMBDAS = [0.0, .005, .01, .025, .05, .10, .20]
DATASETS = {
    "botox": "trainingdatapro/botox-injections-before-and-after",
    "skin": "ranvijaybalbir/acne-wrinkles-spots-classification",
}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def download_dataset(which: str, cache_dir: Path) -> Path:
    slug = DATASETS[which]; dest = cache_dir / which
    if dest.exists() and len(list(dest.rglob("*"))) > 1:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    errors=[]
    try:
        import kagglehub
        remote=Path(kagglehub.dataset_download(slug, output_dir=str(dest)))
        LOG.info("KaggleHub completed: %s", remote)
        return remote
    except Exception as exc: errors.append("kagglehub: " + repr(exc))
    try:
        subprocess.run([sys.executable,"-m","kaggle","datasets","download","-d",slug,"-p",str(dest),"--unzip"],check=True)
        LOG.info("Kaggle CLI completed: %s", dest)
        return dest
    except Exception as exc: errors.append("kaggle CLI: " + repr(exc))
    raise RuntimeError(f"Could not download {slug}.\n" + "\n".join(errors) +
                       "\nSet --botox-dir/--skin-dir if the official archive was extracted manually.")


def image_paths(root: Path):
    return sorted((p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file()),key=str)



def unpack_nested_archives(root: Path):
    """Kaggle ZIP may contain nested before.tar.gz / after.tar.gz archives."""
    for p in list(root.rglob('*')):
        if not p.is_file(): continue
        if not (p.name.lower().endswith(('.tar.gz','.tgz','.zip'))): continue
        marker=p.parent/('.extracted_'+p.name.replace('.','_'))
        if marker.exists(): continue
        LOG.info('Unpacking nested archive: %s',p)
        if p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as zf:
                for member in zf.infolist():
                    dest=(p.parent/member.filename).resolve()
                    if not dest.is_relative_to(p.parent.resolve()):raise ValueError('Unsafe zip path')
                    if member.file_size>1_000_000_000:raise ValueError('Unexpectedly large archive entry')
                zf.extractall(p.parent)
        else:
            with tarfile.open(p,'r:gz') as tf:
                tf.extractall(p.parent,filter='data')
        marker.write_text('ok')


def assign_botox(root: Path) -> pd.DataFrame:
    paths=image_paths(root); out=[]
    for p in paths:
        parts=[part.lower() for part in p.parts]
        before=any(part in ("before", "pre") for part in parts)
        after=any(part in ("after", "post") for part in parts)
        if before == after: continue
        label=0 if before else 1
        name=p.stem.lower()
        # Paired images are documented as using the same file stem.
        out.append({"path":str(p), "label":label, "subject":name})
    df=pd.DataFrame(out)
    if df.empty: raise ValueError("Could not locate before/after images. Check dataset directory.")
    counts=df.groupby("subject")["label"].agg(["size","nunique"])
    if not np.all((counts["size"]==2)&(counts["nunique"]==2)):
        raise ValueError("Before/after files are not uniquely paired by filename. Resolve using CSV person ID before proceeding. "
                         + str(counts.to_dict("index")))
    return df


def assign_skin(root: Path) -> pd.DataFrame:
    images=image_paths(root); out=[]
    aliases={"acne":0,"spots":1,"spot":1,"wrinkles":2,"wrinkle":2}
    for p in images:
        matches=[]
        for part in p.parent.parts:
            tag=part.lower().replace("_", " ").replace("-", " ").strip()
            for a,v in aliases.items():
                if tag==a or tag.startswith(a+" ") or tag.endswith(" "+a): matches.append(v)
        if len(set(matches)) != 1: continue
        out.append({"path":str(p),"label":matches[0],"subject":None})
    df=pd.DataFrame(out)
    if df.empty or df.label.nunique()!=3:
        raise ValueError("Expected class folders for acne / spots / wrinkles under dataset root. "
                         "Use a directory structure with these class names.")
    return df


def phash(p: str) -> int:
    with Image.open(p) as im:
        gray=np.array(ImageOps.grayscale(im).resize((32,32)),dtype=np.float32)
    dct=cv2.dct(gray)[:8,:8]
    median=np.median(dct.reshape(-1)[1:])
    value=0
    for bit in (dct>median).reshape(-1): value=(value<<1)|int(bit)
    return value


def audit_and_group(df: pd.DataFrame, which: str, max_phash_distance: int=4):
    df=df.copy().reset_index(drop=True)
    df["sha256"]=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in df.path]
    df["phash"]=[format(phash(p),"016x") for p in df.path]
    hs=[int(h,16) for h in df.phash]; n=len(df)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def unite(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj: parent[rj]=ri
    pairs=[]; crosslabel=[]
    for i in range(n):
        for j in range(i+1,n):
            d=(hs[i]^hs[j]).bit_count()
            if d<=max_phash_distance:
                pairs.append((i,j,d))
                if df.label.iloc[i]!=df.label.iloc[j]: crosslabel.append((i,j,d))
                if which=="skin": unite(i,j)
    if which=="skin":
        # These are near-duplicate-image clusters, NOT independently verified subject IDs.
        df["group"]=[f"phash_{find(i)}" for i in range(n)]
    else:
        df["group"]=[f"subject_{s}" for s in df.subject]
    if which=="botox":
        for i,j,d in pairs:
            if df.subject.iloc[i] != df.subject.iloc[j] and d<=1:
                raise ValueError("Possible duplicate photographs across independent subject IDs; manual audit required: " + df.path.iloc[i] + " vs " + df.path.iloc[j])
    stats={"dataset":which,"n_images":n,"class_counts":{str(k):int(v) for k,v in df.label.value_counts().items()},
           "n_groups":int(df.group.nunique()),"exact_duplicate_pairs":sum(df.sha256.duplicated()),
           "phash_near_pairs":len(pairs),"cross_label_near_pairs":len(crosslabel),
           "grouping":"documented subject ID" if which=="botox" else "image near-duplicate proxy; NO verified subject IDs",
           "note":"phash proximity cannot verify that two different photographs depict the same person"}
    return df,stats


def handcrafted(im: Image.Image, side: int) -> np.ndarray:
    img=np.asarray(ImageOps.fit(im.convert("RGB"),(side,side),method=Image.Resampling.BILINEAR))
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h=hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2),
          block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    lbp=local_binary_pattern(gray,8,1,"uniform")
    lbph=np.histogram(lbp,bins=26,range=(0,26),density=True)[0].astype(np.float32)
    glcm=graycomatrix((gray//4).astype(np.uint8),distances=[1,2,4],
                     angles=[0,np.pi/4,np.pi/2,3*np.pi/4],levels=64,symmetric=True,normed=True)
    g=np.concatenate([graycoprops(glcm,prop).ravel() for prop in ["contrast","dissimilarity","homogeneity","energy","correlation"]]).astype(np.float32)
    gray_f=gray.astype(np.float32)/255.0
    gabor=[]
    for frequency in [.1,.2,.3]:
        for theta in [0,np.pi/4,np.pi/2,3*np.pi/4]:
            real,imag=cv2.getGaborKernel((9,9),3,theta,1/frequency,.5,psi=0,ktype=cv2.CV_32F),None
            res=cv2.filter2D(gray_f,-1,real)
            gabor.extend([float(res.mean()),float(res.std())])
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[4,2],[0,180,0,256]).ravel().astype(np.float32)
    hist/=max(float(hist.sum()),1)
    return np.nan_to_num(np.concatenate([h,lbph,g,np.asarray(gabor,dtype=np.float32),hist]),
                         nan=0,posinf=0,neginf=0).astype(np.float32)


def augment(im: Image.Image, rng: np.random.Generator, index:int) -> Image.Image:
    im=im.convert("RGB")
    if index%3==0: im=ImageOps.mirror(im)
    angle=float(rng.choice([-12,-6,6,12]))
    im=im.rotate(angle, Image.Resampling.BILINEAR)
    im=ImageEnhance.Brightness(im).enhance(float(rng.uniform(.80,1.20)))
    im=ImageEnhance.Contrast(im).enhance(float(rng.uniform(.80,1.20)))
    w,h=im.size; keep=float(rng.uniform(.85,.97)); cw,ch=max(1,int(w*keep)),max(1,int(h*keep))
    x=int(rng.integers(0,w-cw+1)); y=int(rng.integers(0,h-ch+1))
    return im.crop((x,y,x+cw,y+ch)).resize((w,h),Image.Resampling.BILINEAR)


class FrozenDeep:
    def __init__(self, device, pretrained=True):
        self.device=device
        if not pretrained: LOG.warning("NO pretrained weights: functionality test only; not a scientific result")
        weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.net=mobilenet_v3_small(weights=weights).features.eval().to(device)
        for param in self.net.parameters(): param.requires_grad_(False)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.tx=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                    transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    @torch.inference_mode()
    def batch(self,ims,batch_size=32):
        vectors=[]
        for off in range(0,len(ims),batch_size):
            x=torch.stack([self.tx(im) for im in ims[off:off+batch_size]]).to(self.device)
            vectors.append(self.avg(self.net(x)).flatten(1).cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)


def make_features(df,which,cache_dir,aug_count=14,pretrained=True,device="cpu",batch_size=32):
    side=160 if which=="botox" else 128
    key=f"{which}_n{len(df)}_aug{aug_count}_pt{int(pretrained)}_r2"
    target=cache_dir/(key+".npz")
    if target.exists():
        x=np.load(target,allow_pickle=False)
        if int(x['n_original'])==len(df):
            LOG.info("Loaded feature cache %s",target)
            return {k:x[k] for k in x.files}
    net=FrozenDeep(device,pretrained)
    hand=[];deep=[];source=[];augflag=[]
    rng=np.random.default_rng(20261014)
    batch=[]; hb=[]; sid=[]; aid=[]
    def flush():
        nonlocal batch,hb,sid,aid
        if not batch: return
        hvec=np.asarray(hb,dtype=np.float32); dvec=net.batch(batch,batch_size=batch_size)
        hand.extend(hvec);deep.extend(dvec);source.extend(sid);augflag.extend(aid)
        batch=[];hb=[];sid=[];aid=[]
    for i,path in enumerate(tqdm(df.path,desc="Extracting original and train-only augmentation features")):
        with Image.open(path) as original: im=ImageOps.exif_transpose(original).convert('RGB')
        ims=[im]+([augment(im,rng,k) for k in range(aug_count)] if which=="botox" else [])
        for k,example in enumerate(ims):
            batch.append(example);hb.append(handcrafted(example,side));sid.append(i);aid.append(int(k!=0))
            if len(batch)>=batch_size: flush()
    flush()
    payload=dict(h=np.asarray(hand,np.float32),d=np.asarray(deep,np.float32),source=np.asarray(source),
                 augmented=np.asarray(augflag,np.int8),n_original=np.asarray(len(df)))
    np.savez_compressed(target,**payload)
    return payload


def ece(y,prob,bins=10):
    prob=np.clip(np.asarray(prob),1e-9,1)
    confidence=prob.max(1); correct=(prob.argmax(1)==y)
    edges=np.linspace(0,1,bins+1); total=0.
    for i in range(bins):
        mask=(confidence>=edges[i]) & (confidence <= edges[i+1] if i==bins-1 else confidence<edges[i+1])
        if mask.any(): total+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return float(total)

def brier(y,p):
    Y=np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p-Y)**2,axis=1)))

def calc_metrics(y,p):
    pred=p.argmax(1); result={"n":int(len(y)),"acc":float(accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "ECE_10":ece(y,p,10),"Brier_multiclass":brier(y,p),
      "NLL":float(log_loss(y,p,labels=np.arange(p.shape[1])))}
    try:
        result["AUROC"]=(float(roc_auc_score(y,p[:,1])) if p.shape[1]==2
                           else float(roc_auc_score(y,p,multi_class="ovr",average="macro")))
    except ValueError: result["AUROC"]=None
    return result


def softmax(logits):
    z=logits-logits.max(1,keepdims=True)
    ex=np.exp(z);return ex/ex.sum(1,keepdims=True)


class Calibration:
    def __init__(self,kind): self.kind=kind
    def fit(self,p,y):
        self.nclass=p.shape[1]
        self.eps=1e-6
        logits=np.log(np.clip(p,self.eps,1))
        if self.kind=='none': return self
        if self.kind=='temperature':
            res=minimize_scalar(lambda t:log_loss(y,softmax(logits/np.exp(t)),labels=np.arange(self.nclass)),
                                bounds=(-3,3),method='bounded')
            self.temp=float(np.exp(res.x))
        elif self.kind in ('platt','isotonic'):
            self.models=[]
            for j in range(self.nclass):
                targets=(y==j).astype(int)
                if len(np.unique(targets))<2:
                    self.models.append(None); continue
                if self.kind=='platt':
                    obj=LogisticRegression(C=1.,max_iter=1000).fit(logits[:,[j]],targets)
                else:
                    obj=IsotonicRegression(out_of_bounds='clip').fit(p[:,j],targets)
                self.models.append(obj)
        return self
    def predict(self,p):
        if self.kind=='none': return p
        if self.kind=='temperature': return softmax(np.log(np.clip(p,self.eps,1))/self.temp)
        out=[];logits=np.log(np.clip(p,self.eps,1))
        for j,model in enumerate(self.models):
            if model is None: out.append(p[:,j])
            elif self.kind=='platt': out.append(model.predict_proba(logits[:,[j]])[:,1])
            else: out.append(model.predict(p[:,j]))
        out=np.clip(np.column_stack(out),1e-6,1)
        return out/out.sum(1,keepdims=True)


class GateNet(nn.Module):
    def __init__(self,nh,nd,nc,dropout=0.0):
        super().__init__()
        def projection(n):
            return nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,128),nn.LayerNorm(128))
        self.hp=projection(nh);self.dp=projection(nd)
        self.gate=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,2))
        self.classifier=nn.Linear(128,nc)
    def forward(self,h,d):
        hv=self.hp(h);dv=self.dp(d)
        weights=torch.softmax(self.gate(torch.cat([hv,dv],dim=1)),dim=1)
        mixed=weights[:,0:1]*hv+weights[:,1:2]*dv
        return self.classifier(mixed),weights


def fit_gate(fit_h,fit_d,fit_y,stop_h,stop_d,stop_y,nc,lam,seed,device,max_epochs=100):
    seed_everything(seed)
    model=GateNet(fit_h.shape[1],fit_d.shape[1],nc).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-3)
    trainset=TensorDataset(torch.from_numpy(fit_h),torch.from_numpy(fit_d),torch.tensor(fit_y,dtype=torch.long))
    loader=DataLoader(trainset,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    valid_h=torch.from_numpy(stop_h).to(device);valid_d=torch.from_numpy(stop_d).to(device)
    best=float('inf');best_state=None;patience=0
    for epoch in range(max_epochs):
        model.train()
        for h,d,y in loader:
            h,d,y=h.to(device),d.to(device),y.to(device)
            logits,w=model(h,d)
            entropy=-(w*torch.log(w.clamp(min=1e-8))).sum(1).mean()
            loss=nn.functional.cross_entropy(logits,y)-lam*entropy
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            logits,_=model(valid_h,valid_d)
            vloss=float(nn.functional.cross_entropy(logits,torch.tensor(stop_y,device=device)).item())
        if vloss<best-1e-5:
            best=vloss;patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: patience+=1
        if patience>=15: break
    model.load_state_dict(best_state)
    return model, {"epochs":int(epoch+1),"inner_validation_NLL":best}

@torch.inference_mode()
def predict_gate(model,h,d,device):
    model.eval();logits,w=model(torch.from_numpy(h).to(device),torch.from_numpy(d).to(device))
    return torch.softmax(logits,dim=1).cpu().numpy(), w.mean(0).cpu().numpy()


def get_parts(y,groups,train_idx,seed,grouped):
    ysub=y[train_idx];gsub=groups[train_idx]
    def one_split(idx, frac, seed_local):
        ys=y[idx]; gs=groups[idx]
        if grouped:
            spl=GroupShuffleSplit(n_splits=1,test_size=frac,random_state=seed_local)
            a,b=next(spl.split(idx,ys,gs))
        else:
            spl=StratifiedShuffleSplit(n_splits=1,test_size=frac,random_state=seed_local)
            a,b=next(spl.split(idx,ys))
        return idx[a],idx[b]
    fitval,cal=one_split(train_idx,.2,seed+10)
    fit,val=one_split(fitval,.25,seed+20)
    for a,b in [(fit,val),(fit,cal),(fitval,cal)]:
        assert set(groups[a]).isdisjoint(set(groups[b])) if grouped else len(set(a)&set(b))==0
    return fit,val,cal


def train_test_scalers(H,D,source,augflag,fit_original,val_original,cal_original,test_original):
    trainrow=np.flatnonzero(np.isin(source,fit_original))
    originalrow=np.flatnonzero(augflag==0)
    idx_by_source={int(source[i]):i for i in originalrow}
    get_rows=lambda original:np.asarray([idx_by_source[int(i)] for i in original],dtype=int)
    sh=StandardScaler().fit(H[trainrow]);sd=StandardScaler().fit(D[trainrow])
    def trans(rows):
        h=normalize(sh.transform(H[rows])).astype(np.float32)
        d=normalize(sd.transform(D[rows])).astype(np.float32)
        return h,d
    return trans(trainrow), trans(get_rows(val_original)),trans(get_rows(cal_original)),trans(get_rows(test_original)),source[trainrow]


def trial(df, features,which, output_dir, seeds=SEEDS,lambdas=LAMBDAS,folds=5,max_epochs=100,device='cpu'):
    y=df.label.to_numpy(np.int64);groups=df.group.to_numpy();nc=int(y.max())+1
    H=features['h'];D=features['d'];source=features['source'];augflag=features['augmented']
    outrows=[];selection=[]; records=[]
    # Fixed pHash-cluster grouping is a conservative anti-near-duplicate precaution
    # on the skin dataset; it DOES NOT prove person independence.
    grouped=True
    if df.group.nunique() < folds or min(df.label.value_counts()) < folds:
        raise ValueError("Cannot form safe five-fold split for this dataset")
    for seed in seeds:
        split=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
        seed_probs={}
        for fold,(outer_train,outer_test) in enumerate(split.split(np.zeros(len(y)),y,groups)):
            assert set(groups[outer_train]).isdisjoint(set(groups[outer_test]))
            fit,val,cal=get_parts(y,groups,outer_train,seed+fold*100,grouped=grouped)
            # test = strictly original images; no augmented images enter evaluation or calibration
            (fit_h,fit_d),(val_h,val_d),(cal_h,cal_d),(te_h,te_d),fit_sources = train_test_scalers(
                H,D,source,augflag,fit,val,cal,outer_test)
            yf=y[fit_sources];yv=y[val];yc=y[cal];yt=y[outer_test]
            for code in ('D1','F1'):
                def xy(h,d):return d if code=='D1' else np.concatenate([h,d],1)
                # No fitting on calibration, validation or outer test entries.
                base=LogisticRegression(max_iter=1500,C=1.,random_state=seed)
                base.fit(xy(fit_h,fit_d),yf)
                pcal=base.predict_proba(xy(cal_h,cal_d));pte=base.predict_proba(xy(te_h,te_d))
                if len(base.classes_)!=nc: raise ValueError("Some classes absent from train partition")
                for ck in ('none','temperature','platt','isotonic'):
                    calmodel=Calibration(ck).fit(pcal,yc)
                    pp=calmodel.predict(pte)
                    for j,i in enumerate(outer_test): records.append((seed,fold,code+'_'+ck,int(i),int(yt[j]),list(map(float,pp[j]))))
            candidate_models={};candidate_scores={};candidate_weights={}
            for lam in lambdas:
                model,history=fit_gate(fit_h,fit_d,yf,val_h,val_d,yv,nc,float(lam),seed+fold,device,max_epochs)
                pc,_=predict_gate(model,cal_h,cal_d,device)
                pt,w=predict_gate(model,te_h,te_d,device)
                pval,_=predict_gate(model,val_h,val_d,device)
                candidate_models[lam]=(pc,pt)
                candidate_scores[lam]=log_loss(yv,pval,labels=np.arange(nc))
                candidate_weights[lam]=list(map(float,w))
                tag='P3_lambda_'+str(lam).replace('.','p')
                for j,i in enumerate(outer_test): records.append((seed,fold,tag,int(i),int(yt[j]),list(map(float,pt[j]))))
                outrows.append(dict(dataset=which,seed=seed,fold=fold,lambda_=lam,
                                    validation_NLL=candidate_scores[lam],epochs=history['epochs'],
                                    gate_weight_deep=float(w[1]),gate_weight_hand=float(w[0]),
                                    **calc_metrics(yt,pt)))
            # Lambda selected entirely from inner validation data, never outer test.
            winner=min(candidate_scores,key=candidate_scores.get)
            pcal,ptest=candidate_models[winner]
            selection.append(dict(seed=seed,fold=fold,selected_lambda=winner,validation_NLL=candidate_scores[winner],
                                  n_fit_original=len(fit),n_validation_original=len(val),n_calibration_original=len(cal),
                                  n_test_original=len(outer_test)))
            for ck in ('none','temperature','platt','isotonic'):
                chosen=Calibration(ck).fit(pcal,yc).predict(ptest)
                for j,i in enumerate(outer_test): records.append((seed,fold,'P3_nested_'+ck,int(i),int(yt[j]),list(map(float,chosen[j]))))
            LOG.info("%s seed=%d fold=%d: chosen lambda %.3f; outer originals=%d",which,seed,fold,winner,len(outer_test))
        LOG.info("%s seed %d complete",which,seed)
    out=output_dir/which;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(outrows).to_csv(out/'lambda_fold_results.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'nested_lambda_selection.csv',index=False)
    recdf=pd.DataFrame(records,columns=['seed','fold','method','image_id','label','probabilities'])
    recdf['probabilities']=recdf.probabilities.apply(json.dumps)
    recdf.to_csv(out/'original_image_oof.csv',index=False)
    summary=[]
    for (seed,method),g in recdf.groupby(['seed','method']):
        # exact one OOF prediction for each ORIGINAL per method per seed
        if len(g)!=len(y) or g.image_id.nunique()!=len(y):
            raise AssertionError(f"Out-of-fold coverage error for {seed} {method}")
        gg=g.sort_values('image_id');prob=np.array([json.loads(v) for v in gg.probabilities]);truth=gg.label.to_numpy()
        summary.append(dict(seed=seed,method=method,**calc_metrics(truth,prob)))
    sd=pd.DataFrame(summary);sd.to_csv(out/'seed_level_metrics.csv',index=False)
    agg=sd.groupby('method').agg({k:['mean','std'] for k in ('acc','macro_f1','ECE_10','Brier_multiclass','NLL','AUROC')})
    agg.to_csv(out/'summary_seed_mean_sd.csv')
    compute_cluster_ci(recdf,groups,out)
    return sd


def compute_cluster_ci(recdf,groups,out,nboot=600):
    # Correctly paired at the prediction level and clustered over persons (BoNT)
    # or pHash near-duplicate clusters (skin).  Seed correlation is preserved by
    # averaging seed-level paired deltas WITHIN each resample, not by pretending
    # there are N*seed independent observations.
    seeds=sorted(recdf.seed.unique()); methods=sorted(recdf.method.unique())
    pivot={}
    for (s,m),frame in recdf.groupby(['seed','method']):
        frame=frame.sort_values('image_id')
        pivot[(s,m)]=(frame.label.to_numpy(),np.asarray([json.loads(v) for v in frame.probabilities]))
    baseline=['D1_none','D1_temperature','F1_none','F1_temperature']
    targets=['P3_nested_none','P3_nested_temperature']
    keys=np.unique(groups);indices=[np.flatnonzero(groups==g) for g in keys]
    rng=np.random.default_rng(20261014)
    rows=[]
    for target in targets:
        for other in baseline:
            if not all((s,target) in pivot and (s,other) in pivot for s in seeds):continue
            ds={k:[] for k in ('ECE_10','Brier_multiclass')}
            obs={k:[] for k in ds}
            for s in seeds:
                yt,pt=pivot[(s,target)];yb,pb=pivot[(s,other)]
                assert np.array_equal(yt,yb)
                for k in ds:
                    fn={'ECE_10':lambda y,p:ece(y,p),'Brier_multiclass':brier,
                        }[k]
                    obs[k].append(fn(yt,pt)-fn(yb,pb))
            for rep in range(nboot):
                choice=rng.integers(0,len(keys),len(keys))
                idx=np.concatenate([indices[i] for i in choice])
                for k in ds:
                    fn={'ECE_10':lambda y,p:ece(y,p),'Brier_multiclass':brier,
                        }[k]
                    ds[k].append(np.mean([fn(pivot[(s,target)][0][idx],pivot[(s,target)][1][idx])-
                                          fn(pivot[(s,other)][0][idx],pivot[(s,other)][1][idx]) for s in seeds]))
            for k in ds:
                ci=np.percentile(ds[k],[2.5,97.5])
                rows.append({'proposed':target,'comparator':other,'metric':k,
                             'delta_proposed_minus_comparator':np.mean(obs[k]),
                             'CI95_low':ci[0],'CI95_high':ci[1],
                             'bootstrap_cluster_unit':'subject' if out.name=='botox' else 'image pHash-near-duplicate cluster',
                             'n_bootstrap':nboot})
    pd.DataFrame(rows).to_csv(out/'paired_cluster_bootstrap.csv',index=False)



def run_learning_curve(df,features,which,out,seeds,epochs,device):
    """Within-dataset sample-size sensitivity, NOT a universal image-count threshold.

    Fixed lambda 0.05 is precommitted. Use the same original-only grouped
    outer testing to compare F1 vs P3 at distinct sample sizes. Do not infer a
    cross-dataset minimum number of images from this analysis.
    """
    results=[]
    original_count=len(df)
    if which=='skin':
        sizes=[100,200,300,400,500,600]
    else:
        sizes=[12,16,20,23]  # subjects, each has 2 original images
    class_members={k:np.flatnonzero(df.label.to_numpy()==k) for k in sorted(df.label.unique())}
    for n in sizes:
        if which=='skin' and n>len(df): continue
        if which=='botox' and n>df.subject.nunique(): continue
        rng=np.random.default_rng(102+int(n))
        if which=='skin':
            # Class-balanced subset; for near-duplicate clusters, stratified
            # grouping remains mandatory at the subsequent CV stage.
            target=[n//3+(1 if k<n%3 else 0) for k in range(3)]
            chosen=np.concatenate([rng.choice(class_members[k],size=target[k],replace=False) for k in range(3)])
        else:
            groups=np.unique(df.group)
            keep=set(rng.choice(groups,size=n,replace=False))
            chosen=np.flatnonzero(df.group.isin(keep).to_numpy())
        chosen=np.sort(chosen)
        newdf=df.iloc[chosen].reset_index(drop=True)
        old_source=features['source']; old_aug=features['augmented']
        mask=np.isin(old_source,chosen)
        remap={int(idx):k for k,idx in enumerate(chosen)}
        subfeat={'h':features['h'][mask],'d':features['d'][mask],
                 'source':np.array([remap[int(i)] for i in old_source[mask]],dtype=int),
                 'augmented':old_aug[mask],'n_original':np.array(len(chosen))}
        # Fixed lambda under 3-fold outer CV; repeated across seeds.
        foldnum=3
        label=f'{which}_curve_{n}'
        summary=trial(newdf,subfeat,label,out,seeds=seeds,lambdas=[.05],folds=foldnum,
                      max_epochs=epochs,device=device)
        for _,r in summary.iterrows():
            if r['method'] not in ['P3_nested_none','F1_none']: continue
            results.append({'dataset':which,'N_original_images':len(newdf),'N_subjects':newdf.group.nunique(),
                            'seed':int(r['seed']),'method':r['method'],'macro_f1':r['macro_f1'],
                            'ECE_10':r['ECE_10'],'Brier_multiclass':r['Brier_multiclass']})
        LOG.info('%s learning curve %s complete',which,n)
    path=out/which/'learning_curve_original_only.csv'
    pd.DataFrame(results).to_csv(path,index=False)
    LOG.info('Saved %s',path)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--datasets',nargs='+',choices=['botox','skin'],default=['skin','botox'])
    ap.add_argument('--data-dir',type=Path,default=Path('data'))
    ap.add_argument('--botox-dir',type=Path,help='Previously downloaded official Kaggle Botox dataset folder')
    ap.add_argument('--skin-dir',type=Path,help='Previously downloaded official Kaggle Acne Wrinkles Spots dataset folder')
    ap.add_argument('--out',type=Path,default=Path('results'))
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--lambdas',type=float,nargs='+',default=LAMBDAS)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--botox-aug',type=int,default=14,help='46 originals + 14 variants/original = 690 rows')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--no-pretrained',action='store_true',help='ONLY for smoke-tests, not publication results')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--learning-curve',action='store_true',help='Optional additional grouped learning-curve runs; increases runtime')
    ap.add_argument('--curve-only',action='store_true',help='Run learning-curve analysis only (skip full main trial)')
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    args.out.mkdir(parents=True,exist_ok=True);args.data_dir.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    with open(args.out/'run_config.json','w') as f:
        json.dump({**vars(args),'data_dir':str(args.data_dir),'botox_dir':str(args.botox_dir),
                  'skin_dir':str(args.skin_dir),'out':str(args.out),'device_resolved':device},f,indent=2)
    for which in args.datasets:
        local = args.botox_dir if which=='botox' else args.skin_dir
        root=local if local else download_dataset(which,args.data_dir)
        unpack_nested_archives(root)
        df=assign_botox(root) if which=='botox' else assign_skin(root)
        df, audit=audit_and_group(df,which)
        (args.out/which).mkdir(exist_ok=True)
        df.to_csv(args.out/which/'dataset_manifest_AUDIT_LOCAL_PATHS.csv',index=False)
        (args.out/which/'dataset_audit.json').write_text(json.dumps(audit,indent=2))
        LOG.info('Audit: %s',audit)
        if audit['cross_label_near_pairs']:
            LOG.warning('Near-duplicate pairs across LABELS: inspect before paper submission')
        if which=='skin':
            LOG.warning('Dataset has NO verified subject metadata. Only duplicate-cluster grouping can be claimed.')
        features=make_features(df,which,args.data_dir,aug_count=args.botox_aug if which=='botox' else 0,
                               pretrained=not args.no_pretrained,device=device,batch_size=args.batch_size)
        if not args.curve_only:
            summary=trial(df,features,which,args.out,seeds=args.seeds,lambdas=args.lambdas,
                          folds=args.folds,max_epochs=args.epochs,device=device)
            print('\n',which,'SEED LEVEL RESULTS')
            print(summary.to_string(index=False))
        if args.learning_curve or args.curve_only:
            run_learning_curve(df,features,which,args.out,args.seeds[:3],args.epochs,device)
    LOG.info('All finished. Outputs in %s',args.out.resolve())

if __name__=='__main__':main()
