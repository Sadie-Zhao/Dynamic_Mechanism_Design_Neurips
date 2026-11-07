"""Plot archived post-distillation training revenue; no simulation or optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'budget/figures/distillation_revenue_compact.pdf')
    args=parser.parse_args()
    rows=json.loads((ROOT/'budget/results/distillation_trajectories.json').read_text(encoding='utf-8'))
    byunit={r['unit']:r for r in rows}
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7,4.25),sharex=True)
    for ax,(setting,teacher) in zip(axes.flat,[('e1','B'),('e1','C'),('e2','B'),('e2','C')]):
        for budget,color in [(1000,'#1875ab')]:
            r=byunit[f'{setting}_{teacher}toA_distill{budget}']
            y=np.asarray(r['training_evidence']['scalar_logs']['revenue'])
            assert len(y)==budget
            smoothed=np.asarray([np.mean(y[max(0,i-24):i+1]) for i in range(len(y))])
            ax.plot(np.arange(1,len(y)+1),smoothed,color=color,lw=1.25,label=f'{budget}+{budget}')
        ax.set_title(f'{setting}: {teacher} → A',fontsize=10,pad=4)
        ax.grid(alpha=.15);ax.set_xlim(0,1000)
    for ax in axes[:,0]:ax.set_ylabel('Training revenue')
    for ax in axes[-1,:]:ax.set_xlabel('Acceptance-training epoch')
    fig.tight_layout(rect=(0,0,1,1),h_pad=1.3,w_pad=1.5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight')
    fig.savefig(args.output.with_suffix('.png'),dpi=180,bbox_inches='tight')
    print(args.output)

if __name__=='__main__':main()
