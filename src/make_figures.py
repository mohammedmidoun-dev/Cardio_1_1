"""
make_figures.py — les deux figures du manuscrit que les scripts d'analyse
ne produisent pas.

Les autres figures sortent déjà de weighting_experiments.py et de
revision_analyses.py. Celles-ci manquaient au dépôt :

  Figure 1  les cinq courbes de calibration assemblées en un seul panneau,
            ordonnées par prévalence, avec les étiquettes (a) à (e)
  Figure 4  ce que laisse la correction analytique, et l'atténuation
            en fonction de la flexibilité du modèle

Tout est lu depuis results/tables et results/figures. Aucune valeur n'est
saisie à la main.

    python src/make_figures.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLOUR = {"logreg": "#1F4E79", "hgb": "#C00000", "rf": "#548235"}
LABEL = {"logreg": "penalised logistic regression",
         "hgb": "gradient boosting",
         "rf": "random forest"}
# décalage des étiquettes de points, pour éviter les chevauchements
OFFSET = {"logreg": (0, 8), "hgb": (0, -14), "rf": (0, 9)}


# ------------------------------------------------------------------ figure 1
def figure_calibration_panel():
    """Assemble les cinq courbes de calibration en une figure unique.

    Les panneaux sont ordonnés par prévalence croissante : lus dans cet
    ordre, ils montrent le biais traverser la diagonale.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  Pillow absent : figure 1 ignorée (pip install pillow)")
        return

    order = [("a", "framingham"), ("b", "hungarian"), ("c", "cleveland"),
             ("d", "kaggle_cvd70k"), ("e", "va_longbeach")]
    paths = [FIGURES / f"figV2_L1_calibration_{name}.png" for _, name in order]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"  panneaux absents, figure 1 ignorée : {', '.join(missing)}")
        return

    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    gap = 40
    canvas = Image.new("RGB", (2 * w + gap, 3 * h + 2 * gap), "white")
    # deux colonnes, le cinquième panneau centré sous les quatre autres
    positions = [(0, 0), (w + gap, 0), (0, h + gap), (w + gap, h + gap),
                 ((2 * w + gap - w) // 2, 2 * (h + gap))]
    for im, pos in zip(images, positions):
        canvas.paste(im, pos)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 78)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for (letter, _), (x, y) in zip(order, positions):
        draw.text((x + 26, y + 16), f"({letter})", fill="black", font=font)

    out = FIGURES / "figV2_L1_calibration_panel.png"
    canvas.save(out, dpi=(300, 300))
    print(f"  écrit : {out.name}  ({canvas.size[0]}x{canvas.size[1]} px)")


# ------------------------------------------------------------------ figure 4
def figure_prior_correction():
    """Ce que laisse la correction analytique, et d'où vient l'atténuation."""
    f_corr = TABLES / "wexp_D_prior_correction.csv"
    f_mech = TABLES / "wexp_D3_mechanism.csv"
    for f in (f_corr, f_mech):
        if not f.exists():
            print(f"  {f.name} absent : lancez prior_correction.py d'abord")
            return

    corr = pd.read_csv(f_corr)
    mech = pd.read_csv(f_mech)
    # une profondeur illimitée est enregistrée comme valeur manquante
    mech["setting"] = mech.setting.fillna(np.inf)

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))

    # (a) résidu après correction complète, courbes pointillées avant correction
    after = corr[corr.arm == "weighted + prior_full"].pivot_table(
        index="target_prevalence", columns="algo", values="calib_intercept")
    before = corr[corr.arm == "weighted"].pivot_table(
        index="target_prevalence", columns="algo", values="calib_intercept")
    for kind in ["logreg", "hgb", "rf"]:
        ax[0].plot(after.index, after[kind], marker="o", ms=5, lw=1.8,
                   color=COLOUR[kind], label=LABEL[kind])
        ax[0].plot(before.index, before[kind], ls=":", lw=1.2,
                   color=COLOUR[kind], alpha=0.5)
    ax[0].axhline(0, color="k", lw=1.2)
    ax[0].set_xlabel("Outcome prevalence")
    ax[0].set_ylabel("Calibration intercept after correction")
    ax[0].set_title("(a) What the analytical correction leaves", fontsize=11)
    ax[0].text(0.255, 0.33, "over-correction", fontsize=9.5, color=COLOUR["rf"])
    ax[0].text(0.255, -0.62, "under-correction", fontsize=9.5, color=COLOUR["hgb"])
    ax[0].text(0.052, -2.62, "dotted: before correction", fontsize=8.5, color="#666666")
    ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8.5, loc="lower right")

    # (b) fraction exprimée contre flexibilité ; chaque famille sur son axe
    varied = {"logreg": "penalty C: 0.001 to 100",
              "rf": "max depth: 3 to unlimited",
              "hgb": "learning rate: 0.02 to 0.2"}
    for kind in ["logreg", "hgb", "rf"]:
        g = mech[mech.algo == kind].sort_values("setting")
        x = np.linspace(0, 1, len(g))
        ax[1].plot(x, g.alpha.values, marker="s", ms=6, lw=1.8,
                   color=COLOUR[kind], label=f"{LABEL[kind]} \u2014 {varied[kind]}")
        for xi, yi, setting in zip(x, g.alpha.values, g.setting.values):
            text = "unlim." if np.isinf(setting) else f"{setting:g}"
            ax[1].annotate(text, (xi, yi), textcoords="offset points",
                           xytext=OFFSET[kind], ha="center", fontsize=7,
                           color=COLOUR[kind])
    ax[1].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(1.0, 1.015, "full theoretical shift", ha="right", fontsize=8.5)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["least flexible", "most flexible"], fontsize=9)
    ax[1].set_xlim(-0.09, 1.09)
    ax[1].set_ylim(0.58, 1.07)
    ax[1].set_ylabel("Fraction of the theoretical shift expressed")
    ax[1].set_title("(b) Attenuation tracks flexibility, not penalty strength",
                    fontsize=11)
    ax[1].grid(alpha=0.25)
    ax[1].legend(fontsize=7.8, loc="lower left")

    plt.tight_layout()
    out = FIGURES / "figW_prior_correction.png"
    plt.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  écrit : {out.name}")

    # les valeurs citées dans le texte, pour vérification
    print("\n  Valeurs reprises dans le manuscrit :")
    low = after.index.min()
    for kind in ["logreg", "hgb", "rf"]:
        print(f"    {kind:7s} résidu à {low:.2f} : {after.loc[low, kind]:+.3f}"
              f"   (avant correction {before.loc[low, kind]:+.3f})")
    for kind, g in mech.groupby("algo"):
        print(f"    {kind:7s} fraction de {g.alpha.min():.3f} à {g.alpha.max():.3f}")


def main():
    print(f"Tableaux : {TABLES}")
    print(f"Figures  : {FIGURES}\n")
    if not TABLES.exists():
        sys.exit(f"Dossier absent : {TABLES}")
    print("Figure 1")
    figure_calibration_panel()
    print("\nFigure 4")
    figure_prior_correction()
    print("\nTerminé.")


if __name__ == "__main__":
    main()
