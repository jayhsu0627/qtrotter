"""Figures for the notebook and write-up. Color encodes formula order; line style encodes what
is plotted; marker shape repeats the order so identity never rests on color alone."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ORDER_COLOR = {1: "#2a78d6", 2: "#eb6834", 4: "#1baf7a"}
ORDER_MARKER = {1: "o", 2: "s", 4: "^"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9.5, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 5.5, "figure.dpi": 110,
    })


def _order_handles(orders):
    return [Line2D([], [], color=ORDER_COLOR[o], marker=ORDER_MARKER[o], label=f"order {o}")
            for o in orders]


def _style_handles(entries):
    return [Line2D([], [], color=INK_2, linestyle=ls, marker=mk, label=lab)
            for ls, mk, lab in entries]


def _log2_steps(ax, steps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(int(s)) for s in steps])
    ax.minorticks_off()


def bounds_vs_measured(noiseless, model, n, times, save=None):
    """Figure 1: spectral-norm Trotter error and both bounds versus r."""
    style()
    fig, axes = plt.subplots(1, len(times), figsize=(4.1 * len(times), 3.5), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), times):
        g = noiseless[(noiseless.model == model) & (noiseless.n == n) & (noiseless.t == t)]
        for order, go in g.groupby("order"):
            c = ORDER_COLOR[order]
            ax.plot(go.r, go.spectral_err, color=c, marker=ORDER_MARKER[order])
            if order <= 2:
                ax.plot(go.r, go.commutator_bound, color=c, linestyle="--")
            ax.plot(go.r, go.norm_bound, color=c, linestyle=":")
        _log2_steps(ax, sorted(g.r.unique())[::2])
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1e3)
        ax.set_title(f"{model.upper()}, n = {n}, t = {t:g}")
        ax.set_xlabel("Trotter steps r")
    np.atleast_1d(axes)[0].set_ylabel("‖U(t) − S(t/r)ʳ‖  (spectral norm)")
    handles = _order_handles(sorted(noiseless.order.unique())) + _style_handles(
        [("-", None, "measured"), ("--", None, "commutator bound"), (":", None, "norm bound")])
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def bound_looseness(noiseless, min_r=16, save=None):
    """Bound / measured error versus n, median over r >= min_r and both times."""
    style()
    d = noiseless[(noiseless.r >= min_r) & noiseless.spectral_err.notna()].copy()
    d["commutator"] = d.commutator_bound / d.spectral_err
    d["norm"] = d.norm_bound / d.spectral_err
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 3.3), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), models):
        g = d[d.model == model].groupby(["order", "n"])[["commutator", "norm"]].median()
        for order in sorted(g.index.get_level_values(0).unique()):
            go = g.loc[order]
            c, mk = ORDER_COLOR[order], ORDER_MARKER[order]
            if go.commutator.notna().any():
                ax.plot(go.index, go.commutator, color=c, marker=mk, linestyle="--")
            ax.plot(go.index, go.norm, color=c, marker=mk, linestyle=":")
        ax.axhline(1, color=INK_2, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(sorted(d.n.unique()))
        ax.set_title(model.upper())
        ax.set_xlabel("qubits n")
    np.atleast_1d(axes)[0].set_ylabel("bound ÷ measured error")
    handles = _order_handles(sorted(d.order.unique())) + _style_handles(
        [("--", None, "commutator bound"), (":", None, "norm bound")])
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def noisy_tradeoff(noisy, n, times, metric="trace_dist", save=None):
    """Figure 2: total error under device noise and the noiseless Trotter error, versus r."""
    style()
    labels = {"trace_dist": "trace distance to exact state", "infidelity": "infidelity"}
    fig, axes = plt.subplots(1, len(times), figsize=(4.1 * len(times), 3.5), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), times):
        g = noisy[(noisy.n == n) & (noisy.t == t)]
        for order, go in g.groupby("order"):
            go = go.sort_values("r")
            c, mk = ORDER_COLOR[order], ORDER_MARKER[order]
            ax.plot(go.r, go[metric], color=c, marker=mk)
            ax.plot(go.r, go[f"trotter_{metric}"], color=c, linestyle="--")
            best = go.loc[go[metric].idxmin()]
            ax.plot(best.r, best[metric], marker="o", markersize=14, markerfacecolor="none",
                    markeredgecolor=c, markeredgewidth=1.6, zorder=5)
        _log2_steps(ax, [1, 2, 4, 8, 16, 32])
        ax.set_ylim(0, 1.02)
        ax.set_title(f"n = {n}, t = {t:g}")
        ax.set_xlabel("Trotter steps r")
    np.atleast_1d(axes)[0].set_ylabel(labels.get(metric, metric))
    handles = _order_handles(sorted(noisy.order.unique())) + _style_handles(
        [("-", None, "with device noise"), ("--", None, "noiseless (Trotter only)"),
         ("", "o", "measured optimum r*")])
    handles[-1].set(markersize=11, markerfacecolor="none")
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def predicted_vs_measured(summary, save=None):
    """Predicted r* against measured r*, one panel per predictor."""
    style()
    predictors = [("r_knee_munro", "Knee & Munro"), ("r_avtandilyan_pogosov",
                  "Avtandilyan & Pogosov"), ("r_semi_empirical", "semi-empirical (this work)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    lim = (0.8, 80)
    for ax, (col, title) in zip(axes, predictors):
        ax.plot(lim, lim, color=INK_2, linewidth=0.8)
        for order, g in summary.groupby("order"):
            g = g[g[col].notna()]
            if g.empty:
                continue
            jitter = np.exp(rng.uniform(-0.06, 0.06, len(g)))  # separates tied integer r*
            ax.scatter(g.r_measured * jitter, g[col], color=ORDER_COLOR[order],
                       marker=ORDER_MARKER[order], s=30, edgecolor=SURFACE, linewidth=0.8)
        low = summary[summary.order <= 2]  # orders all three predictors cover
        ratio = low[col] / low.r_measured
        ax.set_title(f"{title}\nmedian predicted ÷ measured: {np.median(ratio):.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("measured r*")
    axes[0].set_ylabel("predicted r*")
    fig.legend(handles=[Line2D([], [], color=ORDER_COLOR[o], marker=ORDER_MARKER[o],
                               linestyle="", label=f"order {o}")
                        for o in sorted(summary.order.unique())],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
