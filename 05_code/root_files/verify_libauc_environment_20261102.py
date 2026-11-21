"""Validate pinned LibAUC core loss/optimizer contracts on synthetic data only."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
import libauc
from libauc.losses import AUCMLoss,AveragePrecisionLoss
from libauc.optimizers import PESG,SOAP

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'02_experiments/NCKUH_DMSA_metric_optimization_20261102'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wheel=ROOT/'03_data_models/third_party/libauc_wheels/libauc-2.0.1-py3-none-any.whl'
    sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert sha=='416039daded4a79e5a88b4cc843a7a18efc64b1e254512804984f4de058ba31f'
    torch.manual_seed(1)
    x=torch.randn(8,4)
    y=torch.tensor([0.,1.,0.,1.,0.,1.,0.,1.]).reshape(-1,1)
    indices=torch.arange(8)
    checks=[]
    for name in ['AUCM_PESG','AP_SOAP']:
        model=torch.nn.Linear(4,1)
        initial=torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        if name=='AUCM_PESG':
            loss_fn=AUCMLoss(version='v2',device='cpu')
            optimizer=PESG(model.parameters(),loss_fn=loss_fn,lr=.01,mode='sgd',device='cpu',verbose=False)
        else:
            loss_fn=AveragePrecisionLoss(data_len=8,device='cpu')
            optimizer=SOAP(model.parameters(),lr=.001,device='cpu',verbose=False)
        losses=[]
        for _ in range(3):
            optimizer.zero_grad()
            score=torch.sigmoid(model(x))
            loss=loss_fn(score,y) if name=='AUCM_PESG' else loss_fn(score,y,indices)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        final=torch.cat([p.detach().flatten() for p in model.parameters()])
        displacement=float((final-initial).square().sum())
        assert displacement>0
        checks.append(dict(method=name,finite_forward_backward_and_optimizer_steps=True,
            parameter_squared_displacement=displacement,synthetic_losses=losses))
    paths={name:inspect.getsourcefile(obj) for name,obj in [('AUCMLoss',AUCMLoss),('APLoss',AveragePrecisionLoss),('PESG',PESG),('SOAP',SOAP)]}
    report=dict(version=importlib.metadata.version('libauc'),module_path=libauc.__file__,torch=torch.__version__,
        wheel_sha256=sha,official_wheel_sha_verified=True,
        checks=checks,source_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()},
        clinical_models_trained=False,clinical_data_used=False,
        scope='Core AUC/AP losses and optimizers only; optional graph/data/trainer dependencies were not installed or validated',
        environment=str(ROOT/'.venvs/libauc'),system_python_unchanged=True,
        references=['https://pypi.org/project/libauc/2.0.1/','https://docs.libauc.org/api/libauc.losses.html',
                    'https://arxiv.org/abs/2012.03173','https://arxiv.org/abs/2104.08736'])
    (OUT/'environment_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(version=report['version'],core_contracts_passed=True,clinical_models_trained=False)),flush=True)


if __name__=='__main__':
    main()
