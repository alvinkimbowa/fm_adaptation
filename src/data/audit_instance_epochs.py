import concurrent.futures
import json
from fm_adaptation.config import ExperimentConfig
from data.instance_data import InstancePatches

def audit(epoch):
    import cv2
    cv2.setNumThreads(1)
    cfg=ExperimentConfig.from_yaml('configs/convnextt_m2f_ft_aug_p256_red_ours.yaml')
    ds=InstancePatches(cfg);ds.epoch=epoch
    maximum=0
    for i in range(len(ds)):
        _,t=ds[i];maximum=max(maximum,len(t['masks']))
    return dict(epoch=epoch+1,max_instances=maximum)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description="Audit deterministic training crops over all configured epochs")
    parser.add_argument('--start-epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    output=Path('results/instance_preflight/epoch_audit.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    results=json.loads(output.read_text())[:args.start_epoch] if args.start_epoch and output.exists() else []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(audit,range(args.start_epoch,args.epochs)):
            results.append(row);print(json.dumps(row),flush=True)
            open('results/instance_preflight/epoch_audit.json','w').write(json.dumps(results,indent=2))
