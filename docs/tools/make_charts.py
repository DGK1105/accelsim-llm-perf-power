"""Generate the README charts (light + dark PNGs) from results/*.csv.
Palette: dataviz reference instance (blue / orange / aqua), status colors for the kernel strip."""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "docs", "img"); os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e6e5e1",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", good="#0ca30c", crit="#d03b3b", neutral="#c9c8c2"),
    "dark":  dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", grid="#33332f",
                  s1="#3987e5", s2="#d95926", s3="#199e70", good="#0ca30c", crit="#d03b3b", neutral="#4a4a46"),
}
BAR_PX = 22  # thickness cap

def load(run):
    return {r["app"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", run, "summary.csv")))}

def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"]); ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=t["text2"], labelsize=9, length=0)
    ax.xaxis.grid(True, color=t["grid"], linewidth=1); ax.set_axisbelow(True)
    ax.yaxis.grid(False)

def rounded_hbar(ax, y, w, h, color, x0=0.0):
    """Horizontal bar: square at baseline, 4px-ish rounded data end."""
    ax.add_patch(Rectangle((x0, y - h/2), w, h, color=color, linewidth=0))
    r = min(h * 0.35, w * 0.02) if w > 0 else 0
    ax.add_patch(FancyBboxPatch((x0 + w - 2*r, y - h/2), 2*r, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                color=color, linewidth=0, mutation_aspect=1))

def fig_ax(w, h, t, n=1, wr=None):
    fig, axes = plt.subplots(1, n, figsize=(w, h), dpi=160, facecolor=t["surface"], gridspec_kw=dict(width_ratios=wr) if wr else None)
    return fig, (axes if n > 1 else [axes])

def short(app): return app.split("-")[0]

# ---------------------------------------------------------------- chart 1: speedup + power, V100 vs H100
def chart_v100_h100(t, name):
    v, h = load("smoke-power"), load("h100-power")
    apps = sorted(v, key=lambda a: float(v[a]["cycles"]) / float(h[a]["cycles"]))
    sp = [float(v[a]["cycles"]) / float(h[a]["cycles"]) for a in apps]
    pv = [float(v[a]["avg_power_W"]) for a in apps]; ph = [float(h[a]["avg_power_W"]) for a in apps]
    fig, (a1, a2) = fig_ax(11, 4.6, t, 2, [1, 1.15])
    ys = list(range(len(apps))); hgt = 0.5
    # panel 1: H100 speedup (single series, no legend)
    style(a1, t)
    for y, s in zip(ys, sp): rounded_hbar(a1, y, s, hgt, t["s1"])
    a1.set_yticks(ys); a1.set_yticklabels([short(a) for a in apps], color=t["text"])
    a1.set_xlim(0, 1.6); a1.axvline(1, color=t["text2"], linewidth=1)
    a1.set_title("Cycles on V100 model ÷ cycles on H100 model", loc="left", color=t["text"], fontsize=11, pad=10)
    a1.text(1.6, ys[-1], f"{sp[-1]:.2f}×", va="center", ha="right", color=t["text"], fontsize=9)
    a1.text(sp[0] + 0.03, ys[0], f"{sp[0]:.2f}×", va="center", color=t["text2"], fontsize=9)
    a1.set_xlabel("speedup (same traces)", color=t["text2"], fontsize=9)
    # panel 2: average power, two series
    style(a2, t)
    for y, (p1, p2) in enumerate(zip(pv, ph)):
        rounded_hbar(a2, y + 0.17, p1, 0.3, t["s1"]); rounded_hbar(a2, y - 0.17, p2, 0.3, t["s2"])
    a2.set_yticks(ys); a2.set_yticklabels([""] * len(apps)); a2.set_xlim(0, 110)
    a2.set_title("AccelWattch average power (W)", loc="left", color=t["text"], fontsize=11, pad=10)
    a2.set_xlabel("watts (relative model, V100 calibration)", color=t["text2"], fontsize=9)
    imax = max(range(len(ph)), key=lambda i: ph[i])
    a2.text(ph[imax] + 1.5, imax - 0.17, f"{ph[imax]:.0f} W", va="center", color=t["text2"], fontsize=9)
    a2.legend(handles=[Rectangle((0, 0), 1, 1, color=t["s1"]), Rectangle((0, 0), 1, 1, color=t["s2"])],
              labels=["QV100-SASS (V100)", "H100-SASS"], loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False, fontsize=9, labelcolor=t["text"])
    for a in (a1, a2): a.set_ylim(-0.7, len(apps) - 0.3)
    fig.suptitle("rodinia_2.0-ft V100 traces replayed on both GPU models", x=0.01, ha="left", color=t["text2"], fontsize=9, y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}-{tname}.png"), facecolor=t["surface"]); plt.close(fig)

# ---------------------------------------------------------------- chart 2: LLM layer, prefill vs decode per kernel role
ROLES = ["RMSNorm (input)", "QKV GEMM", "rotary embedding", "KV-cache write", "FlashAttention-3", "fill", "o-proj GEMM",
         "RMSNorm (post-attn)", "gate/up GEMM", "SiLU × up", "down GEMM (split-K)", "split-K reduce"]
FIXED = "qwen25-0.5b__layer12-fixed"

def llm_data():
    import gzip
    d = os.path.join(ROOT, "results", FIXED); cyc = []
    for l in gzip.open(os.path.join(d, "simulator_stdout.txt.gz"), "rt", errors="replace"):
        if l.startswith("gpu_sim_cycle ="): cyc.append(int(l.split("=")[1]))
    pw = [float(l.split("=")[1]) for l in open(os.path.join(d, "accelwattch_power_report.log")) if l.strip().startswith("kernel_avg_power")]
    assert len(cyc) == len(pw) == 96
    n = len(ROLES); dec = range(1, 8)
    return ([cyc[r] for r in range(n)], [sum(cyc[p*n + r] for p in dec) / 7 for r in range(n)],
            [pw[r] for r in range(n)], [sum(pw[p*n + r] for p in dec) / 7 for r in range(n)])

def chart_llm(t, name):
    pc, dc, pp, dp = llm_data()
    fig, (a1, a2) = fig_ax(11, 5.6, t, 2, [1.25, 1])
    n = len(ROLES); ys = list(range(n))[::-1]
    for ax, v1, v2, title, xl, xmax, unit in ((a1, pc, dc, "Simulated cycles per kernel", "cycles (H100-SASS)", 17500, ""),
                                              (a2, pp, dp, "AccelWattch average power per kernel", "watts (relative model)", 125, " W")):
        style(ax, t)
        for y, p1, p2 in zip(ys, v1, v2):
            rounded_hbar(ax, y + 0.17, p1, 0.3, t["s1"]); rounded_hbar(ax, y - 0.17, p2, 0.3, t["s2"])
        ax.set_yticks(ys); ax.set_xlim(0, xmax); ax.set_ylim(-0.7, n - 0.3)
        ax.set_title(title, loc="left", color=t["text"], fontsize=11, pad=10); ax.set_xlabel(xl, color=t["text2"], fontsize=9)
        i1 = max(range(n), key=lambda i: v1[i]); i2 = max(range(n), key=lambda i: v2[i])
        if not unit or v1[i1] >= v2[i2]:   # power panel: label only the overall peak
            ax.text(v1[i1] + xmax * 0.01, ys[i1] + 0.17, f"{v1[i1]:,.0f}{unit}", va="center", color=t["text2"], fontsize=9)
        ax.text(v2[i2] + xmax * 0.01, ys[i2] - 0.17, f"{v2[i2]:,.0f}{unit}", va="center", color=t["text2"], fontsize=9)
    a1.set_yticklabels(ROLES, color=t["text"]); a2.set_yticklabels([""] * n)
    a1.legend(handles=[Rectangle((0, 0), 1, 1, color=t["s1"]), Rectangle((0, 0), 1, 1, color=t["s2"])],
              labels=["prefill pass (11 tokens)", "decode pass (1 token, mean of 7)"], loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2,
              frameon=False, fontsize=9, labelcolor=t["text"])
    fig.suptitle("Qwen2.5-0.5B decoder layer 12 on the H100 model: all 12 kernels of the block, prefill vs decode",
                 x=0.01, ha="left", color=t["text2"], fontsize=9, y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}-{tname}.png"), facecolor=t["surface"]); plt.close(fig)

# ---------------------------------------------------------------- chart 3: kernel strip, before / after the post-processing fix
def chart_strip(t, name):
    n = 96
    fig, (ax,) = fig_ax(11, 2.7, t)
    ax.set_facecolor(t["surface"]); ax.axis("off")
    rows = ((1.55, "Before: v2.0.0 post-processor drops TRYWAITs of the split-K GEMM", lambda i: t["good"] if i <= 10 else (t["crit"] if i == 11 else t["neutral"])),
            (0.0, "After: upstream fix from issue #561, re-traced", lambda i: t["good"]))
    for y0, label, col in rows:
        ax.text(0, y0 + 0.72, label, color=t["text"], fontsize=10, va="bottom")
        for i in range(1, n + 1):
            ax.add_patch(Rectangle((i - 1 + 0.08, y0), 0.84, 0.6, color=col(i), linewidth=0))
    ax.text(10.5, 1.47, "▲ kernel 11 deadlocks", color=t["text2"], fontsize=9, ha="left", va="top")
    for p in range(1, 8):  # pass boundaries on the "after" row
        ax.plot([p * 12, p * 12], [-0.12, 0.0], color=t["text2"], linewidth=1)
    ax.text(6, -0.2, "prefill", color=t["text2"], fontsize=9, ha="center", va="top")
    ax.text(54, -0.2, "7 decode passes, 12 kernels each", color=t["text2"], fontsize=9, ha="center", va="top")
    ax.set_xlim(0, n); ax.set_ylim(-1.35, 2.75)
    ly = -1.15
    ax.add_patch(Rectangle((0.2, ly), 0.9, 0.4, color=t["good"], linewidth=0)); ax.text(1.5, ly + 0.2, "✓ simulated", color=t["text2"], fontsize=9, va="center")
    ax.add_patch(Rectangle((14, ly), 0.9, 0.4, color=t["crit"], linewidth=0)); ax.text(15.3, ly + 0.2, "✕ deadlocked", color=t["text2"], fontsize=9, va="center")
    ax.add_patch(Rectangle((29, ly), 0.9, 0.4, color=t["neutral"], linewidth=0)); ax.text(30.3, ly + 0.2, "not reached", color=t["text2"], fontsize=9, va="center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}-{tname}.png"), facecolor=t["surface"]); plt.close(fig)

for tname, t in THEMES.items():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["text"], "axes.labelcolor": t["text2"],
                         "xtick.color": t["text2"], "ytick.color": t["text2"]})
    chart_v100_h100(t, "v100-vs-h100"); chart_llm(t, "llm-layer-kernels"); chart_strip(t, "llm-layer-kernel-strip")
print(sorted(os.listdir(OUT)))
