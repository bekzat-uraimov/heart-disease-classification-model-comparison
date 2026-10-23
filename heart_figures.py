"""Rerun the heart_classification.ipynb pipeline and save the README figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/PedroBSB/CS310/refs/heads/main/Data/Heart.csv"
OUT = Path("figures")
C_VALUES = [0.01, 0.02, 0.1, 0.15, 0.3, 1, 4]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926",
    },
}

plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def style(ax, t, grid_axis):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def header(ax, t, title, subtitle):
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 36), textcoords="offset points",
                fontsize=13, color=t["ink"], weight="bold")
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                fontsize=9.5, color=t["ink2"])


def legend(ax, t):
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False,
              fontsize=9, labelcolor=t["ink2"], handletextpad=0.3, columnspacing=1.2)


def save(name, draw, figsize):
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=figsize, facecolor=t["surface"])
        draw(ax, t)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}-{mode}.png", dpi=200, facecolor=t["surface"])
        plt.close(fig)


def draw_accuracy_gap(results):
    names = list(results)

    def draw(ax, t):
        style(ax, t, "x")
        for i, name in enumerate(names):
            train, test = results[name]
            ax.plot([train, test], [i, i], color=t["axis"], linewidth=2, zorder=1)
            ax.scatter(train, i, s=80, color=t["s1"], edgecolor=t["surface"], linewidth=2,
                       zorder=2, label="Train" if i == 0 else None)
            ax.scatter(test, i, s=80, color=t["s2"], edgecolor=t["surface"], linewidth=2,
                       zorder=3, label="Test" if i == 0 else None)
            lo, hi = sorted([train, test])
            ax.text(lo - 0.006, i, f"{lo:.1%}", ha="right", va="center", fontsize=9, color=t["ink2"])
            ax.text(hi + 0.006, i, f"{hi:.1%}", ha="left", va="center", fontsize=9, color=t["ink2"])
        ax.set_yticks(range(len(names)), names, fontsize=10, color=t["ink"])
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlim(0.80, 1.04)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        header(ax, t, "Train vs test accuracy",
               "Wide gap = the model memorized training data. Test set: 60 patients.")
        legend(ax, t)

    save("heart_accuracy_gap", draw, (8, 3.6))


def draw_c_sweep(sweep):
    cs = [c for c, _, _ in sweep]
    train = [tr for _, tr, _ in sweep]
    test = [te for _, _, te in sweep]
    best = max(test)
    best_cs = [c for c, _, te in sweep if te == best]
    best_x = (best_cs[0] * best_cs[-1]) ** 0.5

    def draw(ax, t):
        style(ax, t, "y")
        ax.plot(cs, train, color=t["s1"], linewidth=2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label="Train")
        ax.plot(cs, test, color=t["s2"], linewidth=2, marker="o", markersize=7,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label="Test")
        ax.annotate(f"{best:.1%} test at C={' and '.join(map(str, best_cs))}", (best_x, best),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=9, color=t["ink"])
        ax.set_xscale("log")
        ax.set_xticks(cs, [str(c) for c in cs])
        ax.minorticks_off()
        ax.set_xlabel("C (smaller = stronger L2 regularization)", fontsize=9, color=t["ink2"])
        ax.set_ylim(0.78, 0.95)
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        header(ax, t, "Logistic Regression accuracy vs C",
               "Picked on test accuracy, so the best point is optimistic. See Limitations.")
        legend(ax, t)

    save("heart_c_sweep", draw, (8, 4))


def draw_coefficients(coefs):
    coefs = coefs.reindex(coefs["C=1"].abs().sort_values(ascending=False).index)
    features = list(coefs.index)

    def draw(ax, t):
        style(ax, t, "x")
        ax.axvline(0, color=t["axis"], linewidth=1)
        for i, f in enumerate(features):
            a, b = coefs.loc[f, "C=1"], coefs.loc[f, "C=0.1"]
            ax.plot([a, b], [i, i], color=t["axis"], linewidth=2, zorder=1)
        ax.scatter(coefs["C=1"], range(len(features)), s=60, color=t["s1"],
                   edgecolor=t["surface"], linewidth=1.5, zorder=2, label="C=1 (default)")
        ax.scatter(coefs["C=0.1"], range(len(features)), s=60, color=t["s2"],
                   edgecolor=t["surface"], linewidth=1.5, zorder=3, label="C=0.1")
        ax.set_yticks(range(len(features)), features, fontsize=9, color=t["ink"])
        ax.set_ylim(len(features) - 0.5, -0.5)
        ax.set_xlabel("Coefficient (standardized features; positive = higher odds of heart disease)",
                      fontsize=9, color=t["ink2"])
        header(ax, t, "L2 regularization shrinks coefficients toward zero",
               "15 of 16 shrink as C drops from 1 to 0.1.")
        legend(ax, t)

    save("heart_coefficients", draw, (8, 6.2))


def main():
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_URL).drop("Unnamed: 0", axis=1)
    n_raw = len(df)
    df = df.dropna()
    y = df["AHD"]
    X = pd.get_dummies(df.drop("AHD", axis=1), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    def fit_lr(c):
        return LogisticRegression(C=c, random_state=42).fit(X_train_s, y_train)

    lr = fit_lr(1)
    lr_small_c = fit_lr(0.1)
    y_train_01 = y_train.map({"No": 0, "Yes": 1})
    y_test_01 = y_test.map({"No": 0, "Yes": 1})
    xgb = XGBClassifier(random_state=42).fit(X_train_s, y_train_01)

    results = {
        "Logistic Regression (C=1)": (lr.score(X_train_s, y_train), lr.score(X_test_s, y_test)),
        "Logistic Regression (C=0.1)": (lr_small_c.score(X_train_s, y_train), lr_small_c.score(X_test_s, y_test)),
        "XGBoost": (xgb.score(X_train_s, y_train_01), xgb.score(X_test_s, y_test_01)),
    }
    sweep = []
    for c in C_VALUES:
        m = fit_lr(c)
        sweep.append((c, m.score(X_train_s, y_train), m.score(X_test_s, y_test)))
    coefs = pd.DataFrame({"C=1": lr.coef_[0], "C=0.1": lr_small_c.coef_[0]}, index=X.columns)

    print(f"rows: {n_raw} raw, {len(df)} after dropna, {X.shape[1]} features after get_dummies")
    print(f"split: {len(X_train)} train / {len(X_test)} test")
    print(f"AHD all: {y.value_counts().to_dict()}  test: {y_test.value_counts().to_dict()}")
    for name, (train, test) in results.items():
        print(f"{name}: train {train:.4f}  test {test:.4f}  ({round(test * len(y_test))}/{len(y_test)} correct)")
    for c, train, test in sweep:
        print(f"C={c}: train {train:.4f}  test {test:.4f}")
    print(coefs.round(3).to_string())

    draw_accuracy_gap(results)
    draw_c_sweep(sweep)
    draw_coefficients(coefs)


if __name__ == "__main__":
    main()
