"""FP32 instance training, deterministic grid AP selection and resumable state."""
import argparse
from dataclasses import replace
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .config import ExperimentConfig
from data.instance_data import InstancePatches, collate, rle_encode
from .instance_model import InstanceModel, validate_config
from .instance_metrics import InstanceEvaluator
from .instance_curves import render_run


def optimizer_for(model, cfg):
    trunk = {id(p) for p in model.encoder.trunk.parameters()}
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in trunk],lr=cfg.learning_rate),
              dict(params=[p for p in model.parameters() if p.requires_grad and id(p) in trunk],lr=cfg.encoder_learning_rate)]
    return torch.optim.AdamW(groups,weight_decay=cfg.weight_decay)


def checkpoint(path, model, optimizer, scheduler, epoch, best, stale, history):
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               epoch=epoch,best=best,stale=stale,history=history,torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def restore(path, model, optimizer=None, scheduler=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state


@torch.no_grad()
def validate(model, loader, device, run_dir=None, epoch=None):
    model.eval(); evaluator=InstanceEvaluator();saturation=0; max_targets=0; samples=[]
    for images, targets in loader:
        for image,target,pred in zip(images,targets,model.predict(images.to(device))):
            valid=~target['ignore']
            gt=[rle_encode(m & valid) for m in target['masks']]
            dt=[(rle_encode((p>.5)&valid),s) for p,s in zip(pred['probabilities'],pred['scores']) if ((p>.5)&valid).any()]
            evaluator.add(valid.shape,gt,dt)
            saturation+=pred['query_saturation'];max_targets=max(max_targets,len(gt))
            spaced=all(target['case'] != t['case'] or abs(target['left']-t['left'])+abs(target['top']-t['top'])>=512 for _,t,_ in samples)
            if len(samples)<4 and len(gt)>=3 and spaced:
                samples.append((image.clone(),target,pred))
    if run_dir is not None:
        from .instance_qualitative import render_samples
        render_samples(run_dir,epoch,samples)
    return dict(**evaluator.compute(),saturated_tiles=saturation,max_targets=max_targets)


def train(cfg, config_path, resume=False, overfit_steps=0):
    validate_config(cfg)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)
    dataset=InstancePatches(cfg)
    valset=InstancePatches(cfg,'val')
    model=InstanceModel(cfg).to(cfg.device).float()
    optimizer=optimizer_for(model,cfg)
    steps=math.ceil(math.ceil(len(dataset)/cfg.batch_size)/cfg.accumulation_steps)
    warmup=cfg.lr_warmup_iters or max(50,min(500,steps))
    total=steps*cfg.epochs
    schedule=lambda s: (cfg.lr_warmup_start_factor+(1-cfg.lr_warmup_start_factor)*s/warmup) if s < warmup else max(0.,1-(s-warmup)/max(1,total-warmup))**cfg.lr_power
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    if overfit_steps:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        return overfit(model,dataset,optimizer,cfg,overfit_steps)
    run=cfg.run_dir
    run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not resume:
        raise FileExistsError(f'{run}/last.pt exists; use --resume')
    (run/'config.yaml').write_bytes(Path(config_path).read_bytes())
    start,best,stale,history=0,-1.,0,[]
    if resume:
        state=restore(run/'last.pt',model,optimizer,scheduler)
        start,best,stale,history=state['epoch']+1,state['best'],state['stale'],state['history']
        render_run(run,history)
    val_loader=DataLoader(valset,batch_size=cfg.batch_size,num_workers=cfg.num_workers,collate_fn=collate,
                          multiprocessing_context='spawn' if cfg.num_workers else None)
    for epoch in range(start,cfg.epochs):
        dataset.epoch=epoch
        # Epoch-seeded shuffling makes resume independent of earlier validation loader use.
        loader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,
                          collate_fn=collate,multiprocessing_context='spawn' if cfg.num_workers else None,
                          generator=torch.Generator().manual_seed(cfg.seed+epoch))
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0;max_targets=0;started=time.monotonic()
        for step,(images,targets) in enumerate(loader):
            group_size=min(cfg.accumulation_steps,len(loader)-(step//cfg.accumulation_steps)*cfg.accumulation_steps)
            loss=model.loss(model(images.to(cfg.device)),targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss epoch={epoch},step={step}')
            (loss/group_size).backward();loss_sum+=float(loss.detach())
            max_targets=max(max_targets,max(len(t['masks']) for t in targets))
            if (step+1)%cfg.accumulation_steps==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip,error_if_nonfinite=True)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        # Freeze sampling RNG for repeatable validation while preserving training RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(cfg.device).startswith('cuda') else []):
            torch.manual_seed(cfg.seed)
            metrics=validate(model,val_loader,cfg.device,run,epoch+1)
        improved=metrics['mask_AP'] > best+cfg.early_stopping_min_delta
        if improved:best=metrics['mask_AP'];stale=0
        else:stale+=1
        row=dict(epoch=epoch+1,loss=loss_sum/len(loader),seconds=time.monotonic()-started,
                 train_max_targets=max_targets,peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,**metrics)
        history.append(row)
        (run/'history.json').write_text(json.dumps(history,indent=2))
        render_run(run,history)
        checkpoint(run/'last.pt',model,optimizer,scheduler,epoch,best,stale,history)
        if improved:
            torch.save(dict(model=model.state_dict(),epoch=epoch,metrics=metrics),run/'best.pt')
        print(json.dumps(row),flush=True)
        if epoch+1>=cfg.min_epochs and cfg.early_stopping_patience and stale>=cfg.early_stopping_patience:break
    (run/'training_complete.json').write_text(json.dumps(dict(best_mask_AP=best,epochs=len(history))))


def overfit(model,dataset,optimizer,cfg,steps):
    # Choose two nonempty native patches deterministically, keep augmentation fixed.
    batch=[]
    for i in range(len(dataset)):
        item=dataset[i]
        if len(item[1]['masks']):batch.append(item)
        if len(batch)==cfg.batch_size:break
    if len(batch)!=cfg.batch_size:raise ValueError('Insufficient nonempty patches for overfit')
    images,targets=collate(batch);images=images.to(cfg.device)
    model.train();losses=[]
    for step in range(steps):
        torch.manual_seed(cfg.seed)  # fixed sampling makes the loss trend interpretable
        optimizer.zero_grad(set_to_none=True)
        loss=model.loss(model(images),targets)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip,error_if_nonfinite=True)
        optimizer.step();losses.append(float(loss.detach()))
        print(json.dumps(dict(step=step+1,loss=losses[-1],grad_norm=float(norm))),flush=True)
    if not np.isfinite(losses).all() or np.mean(losses[-3:]) >= np.mean(losses[:3])*.9:
        raise RuntimeError(f'Overfit failed to reduce loss by 10%: {losses}')
    out=cfg.run_dir/'overfit';out.mkdir(parents=True,exist_ok=True)
    # Verify a checkpoint round trip with the same architecture/state and next RNG draws.
    sched=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    checkpoint(out/'resume_test.pt',model,optimizer,sched,0,0.,0,[])
    before=next(model.parameters()).detach().clone()
    restore(out/'resume_test.pt',model,optimizer,sched)
    if not torch.equal(before,next(model.parameters())):raise RuntimeError('Checkpoint round trip failed')
    (out/'resume_test.pt').unlink()
    result=dict(losses=losses,passed=True,input_shape=list(images.shape),targets=[len(t['masks']) for t in targets],
                peak_memory_bytes=torch.cuda.max_memory_allocated(),checkpoint_roundtrip=True)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--overfit-steps',type=int,default=0)
    a=p.parse_args();train(ExperimentConfig.from_yaml(a.config),a.config,a.resume,a.overfit_steps)
