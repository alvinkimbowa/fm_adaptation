import numpy as np
import pytest
import torch
from data.instance_data import affine, clip_runs, patch_targets, red_image, rasterize, rle_encode
from fm_adaptation.instance_model import layer_loss, sample_indices
from fm_adaptation.instance_metrics import global_rle, InstanceEvaluator
from pycocotools import mask as mu


def fiber(points, id='a'):
    return dict(id=id,points=points)


def test_crossing_and_touching_keep_identity():
    f=[fiber([[0,32],[63,32]]),fiber([[32,0],[32,63]],'b'),fiber([[32,32],[50,50]],'c')]
    t=patch_targets(f,affine(0,0,64),64)
    assert len(t['masks'])==3
    assert t['masks'][:,32,32].all()


def test_reentry_is_separate():
    p=np.array([[10,10],[80,10],[80,50],[10,50]])
    runs=clip_runs(p,64)
    assert len(runs)==2
    t=patch_targets([fiber(p)],affine(0,0,64),64)
    assert len(t['masks'])==2 and t['source_ids']==['a','a']


def test_short_original_kept_fragment_ignored_positive_precedence():
    t=patch_targets([fiber([[4,10],[8,10]]),fiber([[-50,20],[5,20]],'b'),
                     fiber([[3,0],[3,63]],'c')],affine(0,0,64),64)
    assert len(t['masks'])==2
    assert t['ignore'][20,0] and not t['ignore'][20,3]
    assert not (t['ignore'] & t['masks'].any(0)).any()


def test_empty_overflow_and_rotation():
    assert patch_targets([],affine(0,0,64),64)['masks'].shape==(0,64,64)
    with pytest.raises(ValueError,match='overflow'):
        patch_targets([fiber([[2,2],[50,2]]),fiber([[2,4],[50,4]])],affine(0,0,64),64,queries=1)
    import cv2
    p=np.array([[10,20],[50,20]])
    m=affine(0,0,64,90,True,False)
    source=rasterize(p,(64,64)).astype(np.uint8)
    warped=cv2.warpAffine(source,m,(64,64),flags=cv2.INTER_NEAREST)
    target=patch_targets([fiber(p)],m,64)
    assert np.array_equal(warped,target['masks'][0])


def test_red_and_global_rle():
    gray=np.arange(64,dtype=np.uint8).reshape(8,8)
    rgb=red_image(gray)
    assert np.array_equal(rgb[...,0],gray) and not rgb[...,1:].any()
    local=gray>40
    actual=mu.decode(global_rle(rle_encode(local),(3,4),(20,20)))
    expected=np.zeros((20,20),bool);expected[4:12,3:11]=local
    assert np.array_equal(actual,expected)


def test_thin_support_ignore_loss_and_gradients():
    t=patch_targets([fiber([[2,2],[25,2]]),fiber([[4,15],[25,15]],'b')],affine(0,0,32),32)
    cls=torch.randn(4,2,requires_grad=True)
    masks=torch.randn(4,8,8,requires_grad=True)
    value=layer_loss(cls,masks,t,64)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(masks.grad).all() and torch.isfinite(cls.grad).all()
    truth=torch.tensor(t['masks'])
    indices=sample_indices(truth,~torch.tensor(t['ignore']),torch.randn(4,32,32),64)
    assert truth.flatten(1)[:,indices].any(1).all()
    t['ignore'][:]=True;t['masks']=np.zeros((0,32,32),bool)
    assert layer_loss(cls,masks,t).item()==0


def test_perfect_ap():
    t=rle_encode(rasterize(np.array([[2,2],[25,25]]),(32,32)))
    ev=InstanceEvaluator();ev.add((32,32),[t],[(t,.9)])
    result=ev.compute()
    assert result['mask_AP']==pytest.approx(1.) and result['mask_AR']==pytest.approx(1.)


def test_stitch_continuation_crossing_and_ambiguity():
    from fm_adaptation.instance_stitch import compatible,stitch
    a=np.zeros((64,64),np.float32);a[30:33,:]=.9
    b=a.copy()
    pa=dict(tile=0,origin=(0,0),probability=a,score=.9)
    pb=dict(tile=1,origin=(32,0),probability=b,score=.8)
    assert compatible(pa,pb)
    result,audit=stitch([pa,pb],(64,96))
    assert len(result)==1 and mu.decode(result[0]['rle'])[31].all()
    cross=np.zeros_like(a);cross[:,30:33]=.9
    pc=dict(tile=1,origin=(32,0),probability=cross,score=.8)
    assert not compatible(pa,pc)
    assert len(stitch([pa,pc],(64,96))[0])==2
    pd=dict(tile=1,origin=(32,0),probability=b.copy(),score=.7)
    result,audit=stitch([pa,pb,pd],(64,96))
    assert len(result)==3 and audit['ambiguous_matches']>0


def test_instance_configs_native_and_semantic_default():
    from fm_adaptation.config import ExperimentConfig
    from fm_adaptation.instance_model import validate_config
    for name in ['m2f_inj_ft_aug_ours','convnextt_m2f_ft_aug_p256_red_ours']:
        cfg=ExperimentConfig.from_yaml('configs/'+name+'.yaml');validate_config(cfg)
        assert cfg.patching.stride==128 and cfg.patching.patches_per_case==64
    assert ExperimentConfig.from_yaml('configs/dinov3_convnextt_upernet_ft_aug_p512_red_sci301.yaml').task=='semantic'


def test_ignored_pixels_do_not_affect_matched_mask_loss():
    target=dict(masks=np.zeros((1,32,32),bool),ignore=np.zeros((32,32),bool))
    target['masks'][0,4:6,2:20]=True;target['ignore'][20:]=True
    logits=torch.randn(1,32,32,requires_grad=True);cls=torch.zeros(1,2,requires_grad=True)
    torch.manual_seed(123);first=layer_loss(cls,logits,target,256)
    modified=logits.detach().clone();modified[:,20:]+=100
    torch.manual_seed(123);second=layer_loss(cls,modified,target,256)
    assert torch.allclose(first,second)
    first.backward();assert not logits.grad[:,20:].any()


def test_fusing_disconnected_same_tile_pieces_uses_union():
    from fm_adaptation.instance_stitch import fuse
    a=np.zeros((64,64),np.float32);a[20,2:25]=.9
    b=np.zeros_like(a);b[40,2:25]=.9
    predictions=[dict(tile=0,origin=(0,0),probability=a,score=.9),dict(tile=0,origin=(0,0),probability=b,score=.9)]
    output=fuse(predictions,[0,1],(64,64))
    assert np.array_equal(mu.decode(output['rle']),(a>.5)|(b>.5))


def test_checkpoint_restores_parameters_optimizer_and_rng(tmp_path):
    from fm_adaptation.instance_training import checkpoint,restore
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.1)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:.9**s)
    model(torch.ones(1,2)).sum().backward();opt.step();sched.step()
    expected={k:v.clone() for k,v in model.state_dict().items()}
    checkpoint(tmp_path/'last.pt',model,opt,sched,2,.4,3,[])
    rng=torch.rand(3)
    with torch.no_grad():
        for p in model.parameters():p.add_(100)
    state=restore(tmp_path/'last.pt',model,opt,sched)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in expected.items())
    assert torch.equal(torch.rand(3),rng) and state['epoch']==2 and state['stale']==3
    assert opt.param_groups[0]['lr']==pytest.approx(.09)


def test_disconnected_local_runs_reconnect_through_neighbor():
    from fm_adaptation.instance_stitch import stitch
    a=np.zeros((64,64),np.float32);a[10,4:64]=.9
    b=np.zeros_like(a);b[50,4:64]=.9
    c=np.zeros_like(a);c[10,:50]=.9;c[50,:50]=.9;c[10:51,49]=.9
    predictions=[dict(tile=0,origin=(0,0),probability=a,score=.9),
                 dict(tile=0,origin=(0,0),probability=b,score=.9),
                 dict(tile=1,origin=(32,0),probability=c,score=.9)]
    instances,audit=stitch(predictions,(64,96))
    assert len(instances)==1
    mask=mu.decode(instances[0]['rle'])
    assert mask[10,4:81].all() and mask[50,4:81].all()


def test_contained_short_duplicate_merges():
    from fm_adaptation.instance_stitch import stitch
    a=np.zeros((64,64),np.float32);a[20,40:45]=.9
    b=np.zeros_like(a);b[20,8:13]=.9
    p=[dict(tile=0,origin=(0,0),probability=a,score=.9),dict(tile=1,origin=(32,0),probability=b,score=.9)]
    assert len(stitch(p,(64,96))[0])==1


def test_short_fiber_diagnostics_perfect_prediction():
    from fm_adaptation.instance_metrics import fiber_diagnostics
    rle=rle_encode(rasterize(np.array([[2,2],[8,2]]),(16,16)))
    d=fiber_diagnostics([dict(id='short',length=6)],[rle],[dict(rle=rle,score=.9)])
    assert d['short_recall_iou50']==1 and d['missed_source_fibers_iou50']==[]
