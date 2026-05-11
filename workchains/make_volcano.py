import numpy as np
import matplotlib.pyplot as plt
from uvsib.db.tables import *
from uvsib.db.utils import *

def main():
    formulas = []
    etas = []
    x_desc = []
    pds = []

    rows = query_by_columns(
        DBSurfaceMLAdsorbate,
        {"composition": "WO2", "reaction": "OER"}
    )

    for r in rows:
        formulas.append(r.composition)
        etas.append(r.eta)
        x_desc.append(r.dG_steps[1])
        pds.append(int(np.argmax(r.dG_steps) + 1))

    etas = np.array(etas)
    x_desc = np.array(x_desc)
    pds = np.array(pds)

    # Lowest-η site per oxide
    best = {}
    for f, e, x, p in zip(formulas, etas, x_desc, pds):
        if f not in best or e < best[f][0]:
            best[f] = (e, x, p)
    materials = sorted(best.items(), key=lambda kv: kv[1][0])

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6.8))

    ax.scatter(x_desc, -etas, s=4, c="#dddddd", alpha=0.5, zorder=1,
               label=f"all sites ({len(rows):,})")

    # universal-scaling volcano: y = min(x - 1.97, 1.23 - x), apex at (1.6, -0.37)
    xline = np.linspace(0.0, 3.2, 400)
    ax.plot(xline[xline <= 1.6], (xline - 1.97)[xline <= 1.6], color="black", lw=1.8, zorder=3)
    ax.plot(xline[xline >= 1.6], (1.23 - xline)[xline >= 1.6], color="black", lw=1.8, zorder=3)
    ax.scatter([1.6], [-0.37], marker="*", s=130, color="black", zorder=5)

    ax.axhline(0, color="#779977", lw=0.6)
    ax.axhline(-0.37, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(1.6, color="#bbbbbb", lw=0.5, ls=":")
    ax.text(0.05, 0.02, r"ideal: $-\eta = 0$", color="#557755", fontsize=9)
    ax.annotate(
        "universal-scaling peak\n$\\eta^*=0.37$ V at $x=1.6$ eV",
        xy=(1.6, -0.37), xytext=(2.2, -0.05), fontsize=9,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
    )

    pds_colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
    pds_labels = {
        1: r"PDS 1 (*$\to$*OH)",
        2: r"PDS 2 (*OH$\to$*O)",
        3: r"PDS 3 (*O$\to$*OOH)",
        4: r"PDS 4 (*OOH$\to$O$_2$)",
    }
    seen = set()
    for f, (e, x, p) in materials:
        ax.scatter(x, -e, s=85, c=pds_colors[p], edgecolor="black", lw=0.7, zorder=6,
                   label=pds_labels[p] if p not in seen else None)
        seen.add(p)
        ax.annotate(f, (x, -e), xytext=(5, 5), textcoords="offset points",
                    fontsize=8.5, zorder=7)

    ax.set_xlabel(r"$\Delta G_{*\!O} - \Delta G_{*\!O\!H}$  (eV)")
    ax.set_ylabel(r"$-\eta$  (V)")
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(-1.4, 0.1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
#    fig.savefig("volcano.pdf", bbox_inches="tight")
    fig.savefig("volcano.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
