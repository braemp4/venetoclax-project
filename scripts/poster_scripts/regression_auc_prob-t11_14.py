from figs_output import fig_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.Harmonized_Datasets import harmonize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.formula.api as smf

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Setup
# ============================================================

pipeline_results_dir = fig_output + "pipeline_output/"

model_name = "lr_reg"

RED = "#C62828"
RED_LIGHT = "#E57373"
BLUE = "#1565C0"
GREY = "#666666"

sns.set_style("white")


# ============================================================
# Load data
# ============================================================

datasets = harmonize()

mf_ad = datasets["Moffitt"]
dc_ad = datasets["Decampos"]

cv_results = (
    pd.read_csv(
        f"{pipeline_results_dir}{model_name}_cv_results.csv"
    )
    .set_index("samples")
)

ev_results = (
    pd.read_csv(
        f"{pipeline_results_dir}{model_name}_ev_results.csv"
    )
    .set_index("samples")
)

mf_ad = mf_ad[cv_results.index]
dc_ad = dc_ad[ev_results.index]

mf_ad.obs["y_pred"] = cv_results["y_pred"]
mf_ad.obs["y_prob"] = cv_results["y_prob"]

dc_ad.obs["y_pred"] = ev_results["y_pred"]

mf_ad = mf_ad[mf_ad.obs["t11_14"].notna()]

mf_ad.obs["t11_14"] = (
    mf_ad.obs["t11_14"]
    .astype(int)
    .astype("category")
)


# ============================================================
# Regression data
# ============================================================

df = pd.DataFrame(mf_ad.obs)[
    ["y_prob", "t11_14", "ABT-199"]
].copy()

t1114_codes = df["t11_14"].cat.codes.to_numpy(dtype=float)

y = df["ABT-199"].to_numpy()
y_prob = df["y_prob"].to_numpy()


# ============================================================
# OLS models
# ============================================================

# Model 1: model probability only
X1 = add_constant(y_prob)

model_1 = OLS(
    y,
    X1
).fit(cov_type="HC3")


# Model 2: FISH only
X2 = add_constant(t1114_codes)

model_2 = OLS(
    y,
    X2
).fit(cov_type="HC3")


# Model 3: FISH + model probability
X3 = np.column_stack([
    t1114_codes,
    y_prob
])

X3 = add_constant(X3)

model_3 = OLS(
    y,
    X3
).fit(cov_type="HC3")


print("### Model probability only ###")
print(model_1.summary())

print("\n### t(11;14) FISH only ###")
print(model_2.summary())

print("\n### Model probability + t(11;14) FISH ###")
print(model_3.summary())


# ============================================================
# Formula version of Model 3
# ============================================================

d = df.rename(
    columns={
        "ABT-199": "ven_auc",
        "t11_14": "t1114",
        "y_prob": "post_prob"
    }
).copy()

m3r = smf.ols(
    "ven_auc ~ C(t1114) + post_prob",
    data=d
).fit(cov_type="HC3")

print("\n### Formula Model 3 ###")
print(m3r.summary())


# ============================================================
# Bootstrap for model probability
# ============================================================

rng = np.random.default_rng(0)

obs = m3r.params["post_prob"]

boot = np.array([
    smf.ols(
        "ven_auc ~ C(t1114) + post_prob",
        data=d.iloc[
            rng.choice(
                len(d),
                len(d),
                replace=True
            )
        ]
    ).fit().params["post_prob"]
    for _ in range(5000)
])

print(f"\nbeta        = {obs:.3f}")
print(
    f"95% CI      = "
    f"[{np.percentile(boot, 2.5):.3f}, "
    f"{np.percentile(boot, 97.5):.3f}]"
)
print(
    f"bootstrap p = "
    f"{2 * min((boot >= 0).mean(), (boot <= 0).mean()):.4f}"
)


# ============================================================
# Regression line plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 5),
    facecolor="white"
)

ax.set_facecolor("white")

sns.regplot(
    x=y_prob,
    y=y,
    ax=ax,
    scatter_kws={
        "s": 42,
        "alpha": 0.75,
        "color": RED
    },
    line_kws={
        "color": RED,
        "linewidth": 2
    },
    color=RED,
    ci=95
)

# Make confidence interval lighter
if len(ax.collections) > 1:
    ax.collections[1].set_facecolor(RED_LIGHT)
    ax.collections[1].set_alpha(0.25)

ax.set_xlabel("Model probability")
ax.set_ylabel("Venetoclax response (AUC)")

ax.set_title(
    "Model Probability vs. Venetoclax Response",
    fontsize=14,
    pad=12
)


# ------------------------------------------------------------
# Plain Matplotlib legend
# ------------------------------------------------------------

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=RED,
        markeredgecolor=RED,
        markersize=6,
        label="Observed"
    ),

    Line2D(
        [0],
        [0],
        color=RED,
        linewidth=2,
        label="Regression line"
    ),

    Patch(
        facecolor=RED_LIGHT,
        edgecolor="none",
        alpha=0.25,
        label="95% CI"
    )
]

ax.legend(
    handles=legend_handles,
    loc="best",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    fancybox=False,
    framealpha=1,
    fontsize=9
)

ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    f"{fig_output}model_probability_regression.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# ============================================================
# Forest plot
# ============================================================

forest_rows = [
    (
        "Model probability",
        model_1.params[1],
        model_1.conf_int()[1][0],
        model_1.conf_int()[1][1],
        model_1.pvalues[1],
        RED
    ),

    (
        "t11;14 FISH",
        model_2.params[1],
        model_2.conf_int()[1][0],
        model_2.conf_int()[1][1],
        model_2.pvalues[1],
        BLUE
    ),

    (
        "t11;14 FISH | model probability",
        model_3.params[1],
        model_3.conf_int()[1][0],
        model_3.conf_int()[1][1],
        model_3.pvalues[1],
        BLUE
    ),

    (
        "Model probability | t11;14 FISH",
        model_3.params[2],
        model_3.conf_int()[2][0],
        model_3.conf_int()[2][1],
        model_3.pvalues[2],
        RED
    )
]


fig, ax = plt.subplots(
    figsize=(7, 4.5),
    facecolor="white"
)

ax.set_facecolor("white")

y_pos = np.arange(len(forest_rows))


# ------------------------------------------------------------
# Coefficients and confidence intervals
# ------------------------------------------------------------

for i, (
    label,
    beta,
    ci_low,
    ci_high,
    p,
    color
) in enumerate(forest_rows):

    ax.errorbar(
        beta,
        i,
        xerr=[
            [beta - ci_low],
            [ci_high - beta]
        ],
        fmt="o",
        color=color,
        ecolor=color,
        markersize=7,
        capsize=4,
        linewidth=2
    )


# ------------------------------------------------------------
# Reference line at zero
# ------------------------------------------------------------

ax.axvline(
    0,
    color=GREY,
    linewidth=1.2,
    linestyle="--"
)


ax.set_yticks(y_pos)

ax.set_yticklabels([
    row[0]
    for row in forest_rows
])

ax.set_xlabel("Regression coefficient")

ax.set_title(
    "OLS Regression Results",
    fontsize=14,
    pad=12
)


# ============================================================
# Center x-axis around zero
# ============================================================

all_low = min(
    row[2]
    for row in forest_rows
)

all_high = max(
    row[3]
    for row in forest_rows
)

max_abs = max(
    abs(all_low),
    abs(all_high)
)

padding = max_abs * 0.20

x_min = -max_abs - padding
x_max = max_abs + padding

ax.set_xlim(
    x_min,
    x_max
)


# ============================================================
# P-values
# ============================================================

p_x = x_max - (x_max - x_min) * 0.02

for i, row in enumerate(forest_rows):

    p = row[4]

    if p < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p:.3f}"

    ax.text(
        p_x,
        i,
        p_text,
        va="center",
        ha="right",
        fontsize=9,
        color="black",
        fontweight="normal"
    )


# ============================================================
# Clean poster style
# ============================================================

ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    f"{fig_output}ols_forest_plot.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
