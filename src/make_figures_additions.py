# À ajouter à src/make_figures.py — trois figures que weighting_experiments.py ne produit pas.
# Figure 3 du manuscrit, Figures S2 et S3 du supplément.

def figure_logit_shift():
    """Figure 3 — ordonnée de calibration observée contre logit(prévalence)."""
    f = TABLES / "wexp_A_prevalence_gradient.csv"
    if not f.exists():
        print(f"  {f.name} absent"); return
    w = pd.read_csv(f)
    w = w[w.arm == "weighted"]
    logit = lambda p: np.log(p / (1 - p))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    x = np.linspace(-3.2, 0.15, 50)
    ax.plot(x, x, "--", color="k", lw=1.4, label="theoretical shift, slope 1")
    for kind in ["logreg", "hgb", "rf"]:
        g = w[w.algo == kind].groupby("target_prevalence").calib_intercept.mean()
        xs = logit(np.asarray(g.index, float))
        ax.plot(xs, g.values, "o", ms=7, color=COLOUR[kind], alpha=0.9)
        slope, icpt = np.polyfit(xs, g.values, 1)
        ax.plot(x, slope * x + icpt, lw=1.7, color=COLOUR[kind],
                label=f"{LABEL[kind]}, slope {slope:.3f}")
    g = w.groupby("target_prevalence").calib_intercept.mean()
    xs = logit(np.asarray(g.index, float))
    slope, _ = np.polyfit(xs, g.values, 1)
    r = np.corrcoef(xs, g.values)[0, 1]
    ax.text(-0.10, -2.95, f"all families pooled:\nslope {slope:.3f},  r = {r:.4f}",
            fontsize=9.5, style="italic", ha="right")
    ax.set_xlabel("logit(outcome prevalence)")
    ax.set_ylabel("Observed calibration intercept")
    ax.set_xlim(-3.2, 0.15); ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "figW_logit_shift.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  écrit : figW_logit_shift.png   (pente {slope:.3f}, r {r:.4f})")


def figure_corrections():
    """Figure S2 — erreur de calibration par jeu et par correction."""
    f = TABLES / "wexp_B_corrections.csv"
    if not f.exists():
        print(f"  {f.name} absent"); return
    b = pd.read_csv(f)
    names = {"none": "none", "class_weight": "class weighting", "oversample": "over-sampling",
             "smote": "SMOTE", "undersample": "under-sampling"}
    order = ["none", "class_weight", "oversample", "smote", "undersample"]
    colours = ["#7F7F7F", "#1F4E79", "#2E75B6", "#C00000", "#843C0C"]
    cohorts = b.groupby("cohort").prevalence.first().sort_values().index.tolist()
    piv = b.pivot_table(index="cohort", columns="correction", values="ece10").loc[cohorts, order]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    width = 0.16
    for j, corr in enumerate(order):
        ax.bar(np.arange(len(cohorts)) + (j - 2) * width, piv[corr].values,
               width=width, color=colours[j], label=names[corr])
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([f"{c}\n(prev. {b[b.cohort == c].prevalence.iloc[0]:.3f})" for c in cohorts],
                       fontsize=9)
    ax.set_ylabel("Expected calibration error")
    ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8.5, ncol=5, loc="upper center")
    plt.tight_layout()
    plt.savefig(FIGURES / "figW_corrections.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  écrit : figW_corrections.png")


def figure_recalibration_method():
    """Figure S3 — méthode de recalibrage contre taille du jeu."""
    f = TABLES / "wexp_C_recalibration_method.csv"
    if not f.exists():
        print(f"  {f.name} absent"); return
    c = pd.read_csv(f)
    size = c.groupby("cohort").n.first().sort_values()
    style = {"weighted": ("#C00000", "o", "weighted, no remedy"),
             "weighted + Platt": ("#1F4E79", "s", "Platt scaling"),
             "weighted + isotonic": ("#548235", "^", "isotonic regression")}
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for arm, (col, mk, lab) in style.items():
        g = c[c.arm == arm].groupby("cohort").ece10.mean().reindex(size.index)
        ax[0].plot(size.values, g.values, marker=mk, ms=7, lw=1.8, color=col, label=lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Dataset size (n, log scale)")
    ax[0].set_ylabel("Expected calibration error"); ax[0].grid(alpha=0.25)
    ax[0].legend(fontsize=8.5); ax[0].set_title("Calibration after each remedy", fontsize=11)
    base = c[c.arm == "weighted"].groupby("cohort").auroc.mean()
    for arm, (col, mk, lab) in style.items():
        if arm == "weighted":
            continue
        g = (base - c[c.arm == arm].groupby("cohort").auroc.mean()).reindex(size.index)
        ax[1].plot(size.values, g.values, marker=mk, ms=7, lw=1.8, color=col, label=lab)
    ax[1].axhline(0, color="k", lw=1.1)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Dataset size (n, log scale)")
    ax[1].set_ylabel("AUROC lost relative to the weighted model")
    ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8.5)
    ax[1].set_title("Cost in discrimination", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES / "figW_recalibration_method.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  écrit : figW_recalibration_method.png")
