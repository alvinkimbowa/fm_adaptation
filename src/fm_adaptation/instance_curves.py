"""Refresh each instance run's loss/AP learning curves after every epoch."""
import json
from pathlib import Path


def render_run(run_dir, history=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run_dir=Path(run_dir)
    if history is None:
        history=json.loads((run_dir/'history.json').read_text())
    if not history:return None
    epochs=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(epochs,[r['loss'] for r in history],color='#2563a5',label='Training loss')
    axes[0].set(title='Training set loss',xlabel='Epoch',ylabel='Classification + BCE + Dice loss')
    axes[1].plot(epochs,[r['mask_AP'] for r in history],color='#2563a5',label='Mask AP (IoU 0.50–0.95)')
    axes[1].plot(epochs,[r['mask_AP50'] for r in history],color='#68737d',linestyle='--',label='Mask AP50')
    axes[1].set(title='Deterministic validation-grid mask AP',xlabel='Epoch',ylabel='Average precision (0–1)',ylim=(0,None))
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_xlim(.5,max(2,epochs[-1])+.5)
    fig.suptitle(f'{run_dir.parent.name} · fold {run_dir.name.removeprefix("fold_")}')
    fig.text(.5,-.02,'Source: per-epoch history.json · native 256 px patches · includes auxiliary decoder losses',ha='center',fontsize=8)
    path=run_dir/'history.png';tmp=run_dir/'history.tmp.png'
    fig.savefig(tmp,dpi=150,bbox_inches='tight');plt.close(fig);tmp.replace(path)
    return path


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+',type=Path);a=p.parse_args()
    for run in a.run_dirs:print(render_run(run))
