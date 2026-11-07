"""Render frozen training revenues with sample-SD bands; no experiment execution."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'budget/results/data_and_provenance.json'
OUT = ROOT / 'budget/figures'
OUT.mkdir(exist_ok=True)
data = json.loads(SOURCE.read_text())
colors = {'A': '#0072B2', 'A_wide': '#E69F00', 'B': '#009E73', 'C': '#CC79A7'}
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(8, 3.05))
x = np.arange(25, 1501)
aggregates = []
for ax, setting, title in zip(axes, ['e1', 'e2'], ['(a) e1', '(b) e2']):
    for variant, color in colors.items():
        rows = sorted([r for r in data['selected_trials'] if r['setting'] == setting and r['variant'] == variant], key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == [1, 2, 3]
        y = np.stack([np.convolve(r['revenue'], np.ones(25)/25, mode='valid') for r in rows])
        mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        original = next(a for a in data['aggregates'] if a['setting'] == setting and a['variant'] == variant and a['metric'] == 'revenue')
        np.testing.assert_allclose(mean, original['mean'], atol=1e-12, rtol=0)
        assert np.isfinite(sd).all()
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.16, linewidth=0)
        ax.plot(x, mean, color=color, lw=1.65, label=variant)
        aggregates.append(dict(setting=setting, variant=variant, epoch=x.tolist(), mean=mean.tolist(), sample_sd=sd.tolist()))
    ax.set_title(title, pad=9)
    ax.set_xlim(0, 1500)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xlabel('Training epoch')
    ax.grid(axis='y', alpha=.3, linewidth=.5)
axes[0].set_ylabel('Training revenue')
axes[0].set_ylim(.92, 2.32)
axes[0].set_yticks([1., 1.4, 1.8, 2.2])
axes[1].set_ylim(1.42, 3.75)
axes[1].set_yticks([1.5, 2., 2.5, 3., 3.5])
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.52, 1.015), prop={'family': 'DejaVu Sans', 'size': 10})
fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.77, wspace=.25)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'budget_revenue_compact.{ext}', dpi=220, bbox_inches='tight', pad_inches=.04)
(OUT / 'plot_data.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), smoothing_window=25, band='mean +/- sample standard deviation across seeds (ddof=1)', aggregates=aggregates)))
print('Verified unchanged means for all 8 curves; generated sample-SD bands from the same 24 frozen runs.')
