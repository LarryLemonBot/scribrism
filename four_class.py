"""Local four-class inference with no vendor code execution.

Model provenance: wasitaigeneratedcom/ai-text-detector-small at revision
f1795c86806e6838d4afa33d0b1427f8430c9615, whose publisher declares lineage
from Desklib's detector and Microsoft's DeBERTa-v3-large:
https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615

Local implementation/adaptations: pinned offline configuration and safetensors
loading, strict tensor names, overlapping complete-token coverage, explicit
four-class semantics, fixed document/region rules and a reusable local cache.
These changes and their receipts do not import publisher accuracy claims.
See THIRD_PARTY.md and the original model notices for attribution and terms.
"""
import os,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
import review
from model_store import model_root
MODEL_ID='wasitaigeneratedcom/ai-text-detector-small'
REVISION='f1795c86806e6838d4afa33d0b1427f8430c9615'
LABELS=['human','ai','ai_edited','humanized']

def classify(distributions,config):
    if not distributions or any(len(p)!=4 or any(not math.isfinite(x) or x<0 or x>1 for x in p) or abs(sum(p)-1)>1e-5 for p in distributions):
        raise ValueError('Invalid probability vectors.')
    n=len(distributions)
    means=[sum(p[k] for p in distributions)/n for k in range(4)]
    three=[means[0],means[1]+means[3],means[2]]
    winner=max(range(3),key=lambda i:three[i])
    label=['human','ai','mixed'][winner] if three[winner]>=config['minimum_score'] else 'uncertain'
    regions=[sum(p[0]>=config['region_confidence'] for p in distributions)/n,
             sum(p[1]+p[3]>=config['region_confidence'] for p in distributions)/n]
    if config['region_override'] and min(regions)>=config['mixed_min_fraction']:
        label='mixed'
    return {'label':label,'mean_class_scores':dict(zip(LABELS,means)),
            'confident_human_fraction':regions[0],'confident_ai_fraction':regions[1],
            'calibrated_authorship_probability':None}

class Engine:
    def __init__(self,config):
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        import torch
        from transformers import AutoConfig,AutoModel,AutoTokenizer
        from safetensors.torch import load_file
        torch.set_num_threads(4);torch.manual_seed(0)
        self.torch=torch;self.config=config
        folder=model_root(ROOT)/REVISION
        manifest=review.read_json(folder/'manifest.local.json')
        if manifest['revision']!=REVISION or manifest['model_id']!=MODEL_ID:
            raise ValueError('Model identity mismatch.')
        for name,item in manifest['files'].items():
            if review.file_sha(folder/name)!=item['sha256']:
                raise ValueError('Model artifact checksum mismatch: '+name)
        self.model_sha=manifest['files']['model.safetensors']['sha256']
        cfg=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        if [cfg.id2label[i] for i in range(4)]!=LABELS:
            raise ValueError('Model label order differs from the pinned scoring contract.')
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=AutoModel.from_config(cfg,trust_remote_code=False)
                self.classifier=torch.nn.Linear(cfg.hidden_size,4)
            def forward(self,ids,mask):
                h=self.model(input_ids=ids,attention_mask=mask).last_hidden_state
                m=mask.unsqueeze(-1).to(h.dtype)
                return self.classifier((h*m).sum(1)/m.sum(1).clamp_min(1))
        self.model=Classifier()
        self.model.load_state_dict(load_file(str(folder/'model.safetensors')),strict=True)
        self.model.eval()
        self.tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)

    def score(self,text):
        c=self.config;words=len(text.split())
        if not 80<=words<=30000:raise ValueError('Outside declared word limits.')
        enc=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        ids,offsets=enc['input_ids'],enc['offset_mapping'];windows=[]
        sums=[[0.]*4 for _ in ids];visits=[0]*len(ids)
        for a,b in review.window_ranges(len(ids),c['window_tokens'],c['window_tokens']//2):
            inp=self.tokenizer.prepare_for_model(ids[a:b],add_special_tokens=True,return_attention_mask=True,
                 return_token_type_ids=False,return_tensors='pt',truncation=False)
            inp={k:v.unsqueeze(0) if v.ndim==1 else v for k,v in inp.items()}
            with self.torch.inference_mode():
                probs=self.torch.softmax(self.model(inp['input_ids'],inp['attention_mask']),dim=-1)[0].tolist()
            windows.append({'token_start':a,'token_end':b,'char_start':offsets[a][0],
                            'char_end':offsets[b-1][1],'scores':dict(zip(LABELS,probs))})
            for i in range(a,b):
                visits[i]+=1
                for k in range(4):sums[i][k]+=probs[k]
        if not visits or not all(visits):raise ValueError('Incomplete coverage.')
        scores=[[x/n for x in p] for p,n in zip(sums,visits)]
        return {'words':words,'tokens':len(ids),'token_coverage':1.,'windows':windows,
                'token_distributions':scores,**classify(scores,c)}
