import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIORS = {
    'map_only':              dict(label='Map only',                color='#2a78d6', marker='o'),
    'building_type':         dict(label='Building type',           color='#eb6834', marker='s'),
    'building_type_cuisine': dict(label='Building type + cuisine', color='#1baf7a', marker='^'),
    'history_any':           dict(label='Any restaurant history',  color='#eda100', marker='D'),
    'history_cuisine':       dict(label='Same-cuisine history',    color='#e87ba4', marker='v'),
    'everything':            dict(label='Everything',              color='#008300', marker='P'),
}
CONTROLS = {'control_units': 'Control: units', 'control_year': 'Control: year built'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
EPS_TICKS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
BOROUGHS = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
SPACING = ['<25 ft', '25-50 ft', '50-100 ft', '100+ ft']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'lines.linewidth': 1.6, 'lines.markersize': 4.5,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})


def eps_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks(EPS_TICKS)
    ax.set_xticklabels([f'{e:g}' for e in EPS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel('ε (ft$^{-1}$, log scale)')


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png', dpi=200)
    plt.close(fig)


def fig_single_query(res, out):
    df = pd.read_csv(res / 'single_query.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    ax = axes[0]
    a = df[df.group == 'all']
    for name, st in PRIORS.items():
        s = a[a.prior == name].sort_values('epsilon')
        ax.plot(s.epsilon, s.reid_rate * 100, color=st['color'], marker=st['marker'], label=st['label'])
        ax.fill_between(s.epsilon, s.reid_ci_low * 100, s.reid_ci_high * 100, color=st['color'], alpha=0.15, linewidth=0)
    for name, label in CONTROLS.items():
        s = a[a.prior == name].sort_values('epsilon')
        ax.plot(s.epsilon, s.reid_rate * 100, color=MUTED, linestyle=':', linewidth=1, label=label)
    ax.set_title('All test records', loc='left')
    ax.set_ylabel('Re-identification rate (%)')
    ax.legend(frameon=False, loc='upper left')
    ax = axes[1]
    for name in ['map_only', 'building_type', 'history_cuisine']:
        st = PRIORS[name]
        for group, ls, suffix in [('continuing', '-', 'continuing'), ('new', '--', 'new')]:
            s = df[(df.group == group) & (df.prior == name)].sort_values('epsilon')
            ax.plot(s.epsilon, s.reid_rate * 100, color=st['color'], marker=st['marker'], linestyle=ls,
                    markerfacecolor=st['color'] if group == 'continuing' else 'white',
                    label=f"{st['label']}, {suffix}")
    ax.set_title('Continuing vs new restaurants', loc='left')
    ax.legend(frameon=False, loc='upper left')
    for ax in axes:
        eps_axis(ax)
        ax.set_ylim(0, 100)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, 'fig_single_query')


def fig_multi_record(res, out, eps=0.01):
    df = pd.read_csv(res / 'multi_record.csv', dtype={'n_linked': str})
    df = df[(df.epsilon == eps) & (df.group == 'all') & (df.grouping == 'n_bucket')]
    order = ['2', '3', '4', '5-8']
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(order))
    for name, offset in [('map_only', -0.08), ('history_cuisine', 0.08)]:
        st = PRIORS[name]
        s = df[df.prior == name].set_index('n_linked').reindex(order)
        for col, lo, hi, ls, mfc, tag in [
            ('reid_one_record_avg', 'one_record_ci_low', 'one_record_ci_high', '--', 'white', 'one record (average)'),
            ('reid_rate', 'reid_ci_low', 'reid_ci_high', '-', st['color'], 'all records linked'),
        ]:
            y = s[col].values * 100
            ax.errorbar(x + offset, y, yerr=[y - s[lo].values * 100, s[hi].values * 100 - y], color=st['color'],
                        marker=st['marker'], linestyle=ls, markerfacecolor=mfc, capsize=2.5, elinewidth=1,
                        label=f"{st['label']}: {tag}")
    counts = df[df.prior == 'map_only'].set_index('n_linked').reindex(order)['n_restaurants'].fillna(0).astype(int)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}\n({c:,})' for b, c in zip(order, counts)])
    ax.set_xlabel('Inspections per restaurant (restaurants)')
    ax.set_ylabel('Re-identification rate (%)')
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7)
    save(fig, out, 'fig_multi_record')


def fig_utility(res, out):
    sc = pd.read_csv(res / 'utility_score.csv').sort_values('epsilon')
    cu = pd.read_csv(res / 'utility_cuisine.csv').sort_values('epsilon')
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    a = axes[0]
    a.plot(sc.epsilon, sc.rmse, color='#2a78d6', marker='o', label='RMSE')
    a.plot(sc.epsilon, sc.mae, color='#eb6834', marker='s', label='MAE')
    a.set_ylabel('Error in tract mean score (points)')
    a.set_title('Inspection score distortion', loc='left')
    a = axes[1]
    a.plot(cu.epsilon, cu.tvd_mean, color='#2a78d6', marker='o', label='Mean TVD')
    a.plot(cu.epsilon, cu.tvd_median, color='#eb6834', marker='s', label='Median TVD')
    a.set_ylabel('Total variation distance')
    a.set_title('Cuisine composition distortion', loc='left')
    for a in axes:
        eps_axis(a)
        a.set_ylim(0, None)
        a.legend(frameon=False, loc='upper right')
    fig.tight_layout(w_pad=2.5)
    save(fig, out, 'fig_utility')


def fig_tradeoff(res, out):
    sq = pd.read_csv(res / 'single_query.csv')
    sq = sq[sq.group == 'all']
    sc = pd.read_csv(res / 'utility_score.csv').set_index('epsilon')
    cu = pd.read_csv(res / 'utility_cuisine.csv').set_index('epsilon')
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for a, (util, col, xlabel) in zip(axes, [(sc, 'rmse', 'Inspection score distortion (RMSE, points)'),
                                             (cu, 'tvd_mean', 'Cuisine composition distortion (mean TVD)')]):
        for name in ['map_only', 'building_type', 'history_cuisine']:
            st = PRIORS[name]
            s = sq[sq.prior == name].sort_values('epsilon')
            xv = util.loc[s.epsilon, col].values
            a.plot(xv, s.reid_rate * 100, color=st['color'], marker=st['marker'], label=st['label'])
            if name == 'history_cuisine':
                for e, xi, yi in zip(s.epsilon, xv, s.reid_rate * 100):
                    a.annotate(f'ε={e:g}', (xi, yi), xytext=(4, 4), textcoords='offset points', fontsize=6.5, color=MUTED)
        a.set_xlabel(xlabel)
        a.set_ylim(0, 100)
    axes[0].set_ylabel('Re-identification rate (%)')
    axes[1].legend(frameon=False, loc='upper right')
    fig.tight_layout(w_pad=2)
    save(fig, out, 'fig_tradeoff')


def fig_spacing(res, out, eps=0.01):
    cells = pd.read_csv(res / 'spacing_by_borough.csv')
    std = pd.read_csv(res / 'spacing_standardized.csv')
    c = cells[cells.epsilon == eps].copy()
    c['hits'] = c.records * c.reid_rate
    city = c.groupby(['prior', 'spacing_bin'])[['hits', 'records']].sum()
    city['rate'] = city.hits / city.records
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    for name in ['map_only', 'building_type', 'history_any', 'history_cuisine']:
        st = PRIORS[name]
        s = city.loc[name].reindex(SPACING)
        ax.plot(range(len(SPACING)), s.rate * 100, color=st['color'], marker=st['marker'], label=st['label'])
    ax.set_xticks(range(len(SPACING)))
    ax.set_xticklabels(SPACING)
    ax.set_xlabel('Distance from restaurant lot to nearest other lot')
    ax.set_ylabel('Re-identification rate (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)
    ax.set_title(f'All boroughs, ε = {eps:g}', loc='left')
    ax.legend(frameon=False, loc='upper left')
    ax = axes[1]
    s = std[(std.epsilon == eps) & (std.prior == 'map_only')].set_index('borough').reindex(BOROUGHS)
    y = np.arange(len(BOROUGHS))[::-1]
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.plot([r.actual * 100, r.at_citywide_spacing_mix * 100], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(s.actual * 100, y, color='#2a78d6', s=30, zorder=2, label='Actual')
    ax.scatter(s.at_citywide_spacing_mix * 100, y, color='white', edgecolor='#2a78d6', s=30, zorder=3,
               label='With citywide spacing mix')
    ax.set_yticks(y)
    ax.set_yticklabels(BOROUGHS)
    ax.set_xlabel('Map-only re-identification rate (%)')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'Borough gap, ε = {eps:g}', loc='left')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    fig.tight_layout(w_pad=2)
    save(fig, out, 'fig_spacing')


if __name__ == '__main__':
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    fig_single_query(res, out)
    fig_multi_record(res, out)
    fig_utility(res, out)
    fig_tradeoff(res, out)
    fig_spacing(res, out)
    print('figures written to', out)
