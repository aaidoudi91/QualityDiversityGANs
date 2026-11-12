""" Analysis of the VAE-init vs baseline study: figures and tables. """

import argparse
import json
import os
import textwrap
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter
from torch.func import jacrev, vmap
from torchvision import datasets

from gan.evaluation import SmallCNN, to_images01
from gan.model import Generator

RESULTS, RUNS, CACHE, DATA = "results", "runs", "eval_cache", "data"
VAE_CKPT = "checkpoints/vae/G_init.pth"
OUT, TAB = "figures", "tables"
S, FS = 1.0, 9  # figure scale, base font size

CONDS = [
    dict(key="base", label="Baseline", short="Baseline", runs=["baseline_s0"],
         color="#1f77b4", marker="o", vae=False),
    dict(key="reg", label=r"D dropout + $\beta_1$=0.5", short="D-reg (dropout 0.25, beta1 0.5)",
         runs=["drop_b05_s0"], color="#2ca02c", marker="s", vae=False),
    dict(key="vae", label="VAE init", short="VAE init", runs=["vae_init_s0"],
         color="#d62728", marker="^", vae=True),
    dict(key="vae_reg", label=r"VAE init + D dropout + $\beta_1$=0.5", short="VAE init + D-reg",
         runs=["vae_drop_b05_s0"], color="#ff7f0e", marker="D", vae=True),
]
CK = {c["key"]: c for c in CONDS}


# ----------------------------------------------------------------------------- helpers
def setup_style(fs):
    plt.rcParams.update({
        "font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs, "legend.fontsize": fs - 1,
        "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42,
    })


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{name}.[pdf|png]")


def load_hist(run):
    p = os.path.join(RESULTS, run, "history.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def cond_hist(cond):
    hs = []
    for r in cond["runs"]:
        df = load_hist(r)
        if df is None:
            print(f"[warn] missing {RESULTS}/{r}/history.csv")
            continue
        df = df.copy()
        df["run"] = r
        hs.append(df)
    return pd.concat(hs, ignore_index=True) if hs else None


def agg(df, col):
    g = df.groupby("epoch")[col].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def vae_cost_epochs():
    p = os.path.join(RESULTS, "vae_pretrain", "config.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    return cfg.get("steps", 50000) / np.ceil(60000 / cfg.get("batch_size", 64))


def fmt(m, s, n, d):
    if np.isnan(m):
        return "n/a"
    return f"{m:.{d}f} ± {s:.{d}f}" if n > 1 else f"{m:.{d}f}"


def stat(df, metric, ep, d):
    sub = df.loc[df.epoch == ep, metric].dropna()
    if sub.empty:
        return "n/a"
    return fmt(sub.mean(), sub.std(ddof=1) if len(sub) > 1 else 0.0, len(sub), d)


def write_table(df, name):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("±", "$\\pm$")
    lines = ["\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}", "\\toprule",
             " & ".join(esc(c) if "$" not in str(c) else str(c) for c in df.columns) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TAB, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n== {name} ==")
    print(df.to_string(index=False))


def epochs_to_stay_below(epochs, fid, thr):
    """First epoch after which FID stays <= thr (log-linear interpolation between evaluations)."""
    e, f = np.asarray(epochs, float), np.asarray(fid, float)
    above = np.where(f > thr)[0]
    if len(above) == 0:
        return 0.0
    j = above[-1]
    if j == len(f) - 1:
        return np.nan
    f0, f1 = np.log(f[j]), np.log(f[j + 1])
    return e[j] + (f0 - np.log(thr)) / (f0 - f1) * (e[j + 1] - e[j])


def load_G(path):
    G = Generator(g_output_dim=784)
    sd = torch.load(path, map_location="cpu")
    G.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    return G.eval()


def load_clf():
    clf = SmallCNN()
    clf.load_state_dict(torch.load(os.path.join(CACHE, "mnist_cnn.pth"), map_location="cpu"))
    return clf.eval()


@torch.no_grad()
def class_freq(G, clf, n=10000, seed=0, bs=1000):
    gen = torch.Generator().manual_seed(seed)
    preds = []
    for _ in range(0, n, bs):
        z = torch.randn(bs, 100, generator=gen)
        preds.append(clf(to_images01(G(z), "png_clamp")).argmax(1))
    return torch.bincount(torch.cat(preds), minlength=10).float().numpy() / n


def latent_sensitivity(G, n=64, seed=0):
    """Mean squared Jacobian column norm of G w.r.t. each latent dimension, averaged over z ~ N(0, I)."""
    z = torch.randn(n, 100, generator=torch.Generator().manual_seed(seed))
    J = vmap(jacrev(G))(z)  # (n, 784, 100)
    return J.pow(2).sum(1).mean(0).detach().numpy()


def latent_stats(s):
    p = np.sort(s)[::-1]
    cum = np.cumsum(p) / p.sum()
    return cum, p.sum() ** 2 / (p ** 2).sum(), int(np.searchsorted(cum, 0.9) + 1)


# ----------------------------------------------------------------------------- figures
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(7.4 * S, 2.8 * S))
    metrics = [("fid", "(a) FID (lower is better)"), ("precision", "(b) Precision"), ("recall", "(c) Recall")]
    for ax, (m, title) in zip(axes, metrics):
        for c in CONDS:
            df = cond_hist(c)
            if df is None:
                continue
            g = agg(df, m)
            ax.plot(g.epoch, g["mean"], color=c["color"], marker=c["marker"], ms=3, lw=1.4, label=c["label"])
            if g["count"].max() > 1:
                ax.fill_between(g.epoch, g["mean"] - g["std"], g["mean"] + g["std"], color=c["color"], alpha=0.2)
            if c["vae"] and (g.epoch == 0).any():
                ax.scatter([0], [g.loc[g.epoch == 0, "mean"].iloc[0]], marker="*", s=70, color=c["color"], zorder=5)
        ax.set_xlabel("GAN epoch")
        ax.set_title(title, loc="left")
        if m == "fid":
            ax.set_yscale("log")
            ax.set_yticks([20, 30, 50, 100, 200, 300])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", ls="", color="k", ms=8))
    labels.append("epoch 0 = VAE decoder alone")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, "fig1_convergence")


def fig2():
    fig, axs = plt.subplots(1, 3, figsize=(7.4 * S, 2.8 * S))
    ax = axs[0]
    for c in CONDS:
        df = cond_hist(c)
        if df is None:
            continue
        g = agg(df.dropna(subset=["d_loss"]), "d_loss")
        ax.plot(g.epoch, g["mean"], color=c["color"], marker=c["marker"], ms=3, lw=1.4)
    ax.axhline(2 * np.log(2), color="gray", ls=":", lw=1)
    ax.text(100, 2 * np.log(2) + 0.03, "chance", ha="right", va="bottom", fontsize=FS - 2, color="gray")
    ax.set_xlabel("GAN epoch")
    ax.set_ylabel("D loss (real + fake)")
    ax.set_title("(a) Discriminator loss", loc="left")

    ax = axs[1]
    for key in ("reg", "vae_reg"):
        c = CK[key]
        p = os.path.join(RESULTS, c["runs"][0], "steps.csv")
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p)
        ax.plot(s.step, s.d_real, color=c["color"], ls="-", lw=1.4)
        ax.plot(s.step, s.d_fake, color=c["color"], ls="--", lw=1.4)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean D output")
    ax.set_title("(b) First steps (D-reg runs)", loc="left")

    ax = axs[2]
    gens = [("VAE decoder (before GAN)", VAE_CKPT, "#7f7f7f")]
    gens += [(c["short"], os.path.join(RUNS, c["runs"][0], "G_final.pth"), c["color"]) for c in CONDS]
    rows = []
    for name, path, col in gens:
        if not os.path.exists(path):
            print(f"[warn] missing {path}")
            continue
        cum, pr, n90 = latent_stats(latent_sensitivity(load_G(path)))
        ax.plot(np.arange(1, len(cum) + 1), cum, color=col, lw=1.4)
        rows.append({"Generator": name, "Effective dims (participation ratio)": f"{pr:.1f}",
                     "Dims for 90% of sensitivity": n90})
    ax.axhline(0.9, color="gray", ls=":", lw=1)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("latent dims kept (sorted)")
    ax.set_ylabel("cumulative sensitivity")
    ax.set_title("(c) Latent usage of G", loc="left")

    handles = [Line2D([], [], color=c["color"], marker=c["marker"], ms=4, label=c["label"]) for c in CONDS]
    handles += [Line2D([], [], color="#7f7f7f", label="VAE decoder (before GAN)"),
                Line2D([], [], color="k", ls="-", label="D(x)"), Line2D([], [], color="k", ls="--", label="D(G(z))")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig2_mechanism")
    if rows:
        write_table(pd.DataFrame(rows), "table3_latent_usage")


def fig3():
    df = load_hist("base_e300_s0")
    if df is None:
        print("[warn] base_e300_s0 missing")
        return
    d = df[df.epoch > 0].sort_values("epoch")
    best = d.loc[d.fid.idxmin()]
    fig, axs = plt.subplots(1, 2, figsize=(7.2 * S, 2.8 * S))
    ax = axs[0]
    ax.plot(d.epoch, d.fid, color="#1f77b4", marker="o", ms=3, lw=1.4)
    ax.axvline(best.epoch, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(0.04, 0.06, f"best: epoch {int(best.epoch)}, FID {best.fid:.1f}", transform=ax.transAxes, fontsize=FS - 1)
    ax.set_xlabel("GAN epoch")
    ax.set_title("(a) FID along a long training run", loc="left")
    ax = axs[1]
    ax.plot(d.epoch, d.precision, color="#9467bd", marker="o", ms=3, lw=1.4, label="precision")
    ax.plot(d.epoch, d.recall, color="#17becf", marker="s", ms=3, lw=1.4, label="recall")
    ax.plot(d.epoch, d.class_entropy, color="#7f7f7f", marker="^", ms=3, lw=1.4, ls="--",
            label="class entropy (1 = balanced)")
    ax.axvline(best.epoch, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("GAN epoch")
    ax.set_title("(b) Quality vs diversity", loc="left")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig3_long_training")


def fig4():
    spec = [("VAE decoder (before GAN)", VAE_CKPT),
            ("VAE init, std. D, ep 30", f"{RUNS}/vae_init_s0/G_ep030.pth"),
            ("VAE init, std. D, ep 100", f"{RUNS}/vae_init_s0/G_final.pth"),
            ("VAE init + D-reg, ep 100", f"{RUNS}/vae_drop_b05_s0/G_final.pth"),
            ("Baseline, ep 100", f"{RUNS}/baseline_s0/G_final.pth"),
            ("D-reg, ep 100", f"{RUNS}/drop_b05_s0/G_final.pth"),
            ("Baseline long, ep 150", f"{RUNS}/base_e300_s0/G_ep150.pth"),
            ("Baseline long, ep 200", f"{RUNS}/base_e300_s0/G_ep200.pth"),
            ("Baseline long, ep 250", f"{RUNS}/base_e300_s0/G_ep250.pth"),
            ("Baseline long, ep 300", f"{RUNS}/base_e300_s0/G_ep300.pth")]
    spec = [(n, p) for n, p in spec if os.path.exists(p)]
    real = np.bincount(datasets.MNIST(DATA, train=True, download=True).targets.numpy(), minlength=10) / 60000
    clf = load_clf()
    freqs = [real] + [class_freq(load_G(p), clf) for _, p in spec]
    labels = ["MNIST train (reference)"] + [n for n, _ in spec]
    freqs = np.array(freqs)
    M = np.log2(np.clip(freqs / real, 1 / 16, 16))
    fig, ax = plt.subplots(figsize=(6.4 * S, (0.36 * len(labels) + 1.2) * S))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3), aspect="auto")
    for i in range(freqs.shape[0]):
        for j in range(10):
            ax.text(j, i, f"{100 * freqs[i, j]:.0f}", ha="center", va="center", fontsize=FS - 2,
                    color="white" if abs(M[i, j]) > 2 else "black")
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("digit (predicted by an MNIST classifier)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\log_2$(generated / real frequency)")
    ax.set_title("Cell text: % of generated samples", loc="left", fontsize=FS - 1)
    save(fig, "fig4_class_balance")


def fig5():
    epochs, crop = [0, 10, 50, 100], 2 + 6 * 30  # top-left 6x6 digits of each 8x8 grid

    def find_png(run, ep):
        for base in (RESULTS, RUNS):
            p = os.path.join(base, run, f"samples_ep{ep:03d}.png")
            if os.path.exists(p):
                return p
        return None

    fig, axs = plt.subplots(len(CONDS), len(epochs), figsize=(6.4 * S, 6.6 * S))
    for i, c in enumerate(CONDS):
        for j, e in enumerate(epochs):
            ax = axs[i, j]
            p = find_png(c["runs"][0], e)
            if p:
                ax.imshow(imread(p)[:crop, :crop], cmap="gray")
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"epoch {e}")
            if j == 0:
                ax.set_ylabel(textwrap.fill(c["short"], 16), rotation=0, ha="right", va="center",
                              labelpad=6, fontsize=FS - 2)
    fig.tight_layout()
    save(fig, "fig5_samples")


def tables():
    rows = []
    for c in CONDS:
        df = cond_hist(c)
        if df is None:
            continue
        rows.append({"Condition": c["short"], "FID @10": stat(df, "fid", 10, 1), "FID @100": stat(df, "fid", 100, 1),
                     "Precision @100": stat(df, "precision", 100, 2), "Recall @100": stat(df, "recall", 100, 2),
                     "Class entropy @100": stat(df, "class_entropy", 100, 3), "Seeds": df["run"].nunique()})
    write_table(pd.DataFrame(rows), "table1_main")

    def m(key, metric, ep=100):
        df = cond_hist(CK[key])
        return np.nan if df is None else df.loc[df.epoch == ep, metric].mean()

    pairs = [("VAE init - baseline (standard D)", "vae", "base"), ("VAE init - baseline (with D-reg)", "vae_reg", "reg"),
             ("D-reg - standard D (no VAE)", "reg", "base"), ("D-reg - standard D (with VAE)", "vae_reg", "vae")]
    eff = [{"Contrast at epoch 100": n, "$\\Delta$FID": f"{m(a, 'fid') - m(b, 'fid'):+.1f}",
            "$\\Delta$P": f"{m(a, 'precision') - m(b, 'precision'):+.2f}",
            "$\\Delta$R": f"{m(a, 'recall') - m(b, 'recall'):+.2f}"} for n, a, b in pairs]
    write_table(pd.DataFrame(eff), "table1b_effects")

    vc = vae_cost_epochs()
    rows = []
    for c in CONDS:
        df = cond_hist(c)
        if df is None:
            continue
        g = agg(df, "fid")
        r = {"Condition": c["short"]}
        for thr in (60, 40):
            e = epochs_to_stay_below(g.epoch, g["mean"], thr)
            r[f"GAN epochs, FID $\\leq$ {thr}"] = "n/r" if np.isnan(e) else f"{e:.0f}"
            r[f"Total incl. VAE, FID $\\leq$ {thr}"] = "n/r" if np.isnan(e) else f"{e + (vc if c['vae'] else 0):.0f}"
        rows.append(r)
    write_table(pd.DataFrame(rows), "table2_convergence")
    print(f"(VAE pretraining counted as {vc:.1f} epoch-equivalents)")


JOBS = [("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5), ("tables", tables)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=[n for n, _ in JOBS])
    ap.add_argument("--slides", action="store_true", help="Larger fonts/figures, output in figures_slides/.")
    a = ap.parse_args()
    if a.slides:
        S, FS, OUT = 1.5, 13, "figures_slides"
    os.makedirs(OUT, exist_ok=True)
    setup_style(FS)
    for name, fn in JOBS:
        if a.only and name not in a.only:
            continue
        try:
            fn()
        except Exception:
            print(f"[error] {name} failed:")
            traceback.print_exc()
