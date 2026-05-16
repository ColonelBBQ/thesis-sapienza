import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import pathlib
ARTIFACT_DIR = str(pathlib.Path(__file__).parent.resolve())

BASE_COLS = [
    "threshold","exact_match_accuracy","hamming_loss","jaccard_score",
    "binary_precision","binary_recall","binary_f1","binary_support",
    "micro_precision","micro_recall","micro_f1",
    "macro_precision","macro_recall","macro_f1",
    "weighted_precision","weighted_recall","weighted_f1",
    "roc_auc_macro","roc_auc_micro",
    "label_accuracy__compute","label_precision__compute","label_recall__compute","label_f1__compute","label_support__compute","label_auc__compute",
    "label_accuracy__data_handling","label_precision__data_handling","label_recall__data_handling","label_f1__data_handling","label_support__data_handling","label_auc__data_handling",
    "label_accuracy__network","label_precision__network","label_recall__network","label_f1__network","label_support__network","label_auc__network",
    "label_accuracy__security_compliance","label_precision__security_compliance","label_recall__security_compliance","label_f1__security_compliance","label_support__security_compliance","label_auc__security_compliance",
    "label_accuracy__management_monitoring","label_precision__management_monitoring","label_recall__management_monitoring","label_f1__management_monitoring","label_support__management_monitoring","label_auc__management_monitoring",
    "label_accuracy__cloud_service_essentials","label_precision__cloud_service_essentials","label_recall__cloud_service_essentials","label_f1__cloud_service_essentials","label_support__cloud_service_essentials","label_auc__cloud_service_essentials",
    "epoch","train_loss","validation_loss",
]
EXTRA_COLS = ["run_id","condition","timestamp"]
ALL_COLS = BASE_COLS + EXTRA_COLS

FILES = {
    "Baseline": f"{ARTIFACT_DIR}/baseline_epoch_metrics.csv",
    "Deambiguified": f"{ARTIFACT_DIR}/deambiguified_epoch_metrics.csv",
    "No Ambiguity": f"{ARTIFACT_DIR}/no_ambiguity_epoch_metrics.csv",
}

COLORS = {"Baseline": "#4C72B0", "Deambiguified": "#DD8452", "No Ambiguity": "#55A868"}
LABELS = ["compute", "data_handling", "network", "security_compliance", "management_monitoring", "cloud_service_essentials"]

def load(path, condition_name):
    df = pd.read_csv(path, names=ALL_COLS, skiprows=1, engine="python", on_bad_lines="skip")
    df = df[BASE_COLS + ["run_id"]].copy()
    for col in BASE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["condition"] = condition_name
    df = df.dropna(subset=["epoch","train_loss","validation_loss"])
    df["epoch"] = df["epoch"].astype(int)
    return df

frames = [load(path, name) for name, path in FILES.items()]
data = pd.concat(frames, ignore_index=True)

# ── per-epoch mean ± std across runs ──────────────────────────────────────────
KEY_METRICS = ["exact_match_accuracy", "micro_f1", "macro_f1", "roc_auc_macro",
               "train_loss", "validation_loss"]

agg = (data.groupby(["condition","epoch"])[KEY_METRICS + ["hamming_loss","jaccard_score"]]
           .agg(["mean","std"])
           .reset_index())
agg.columns = ["_".join(c).strip("_") for c in agg.columns]

epochs = sorted(data["epoch"].unique())

# ── Figure 1: loss + main metrics ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Training Dynamics — All Conditions (mean ± std across runs)", fontsize=14, fontweight="bold")

metrics_plot = [
    ("train_loss",            "Train Loss"),
    ("validation_loss",       "Validation Loss"),
    ("exact_match_accuracy",  "Exact Match Accuracy"),
    ("micro_f1",              "Micro F1"),
    ("macro_f1",              "Macro F1"),
    ("roc_auc_macro",         "ROC-AUC (macro)"),
]

for ax, (metric, title) in zip(axes.flat, metrics_plot):
    for cname, color in COLORS.items():
        sub = agg[agg["condition"] == cname].sort_values("epoch")
        mu = sub[f"{metric}_mean"].values
        sd = sub[f"{metric}_std"].fillna(0).values
        xs = sub["epoch"].values
        ax.plot(xs, mu, marker="o", label=cname, color=color, linewidth=2)
        ax.fill_between(xs, mu - sd, mu + sd, alpha=0.15, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    if ax == axes.flat[0]:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/fig1_training_dynamics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_training_dynamics.png")

# ── Figure 2: per-label F1 at best macro-F1 epoch ────────────────────────────
best_rows = []
for cname in COLORS:
    sub = data[data["condition"] == cname]
    best_epoch = sub.groupby("epoch")["macro_f1"].mean().idxmax()
    best_rows.append(sub[sub["epoch"] == best_epoch].mean(numeric_only=True).to_frame().T.assign(condition=cname, best_epoch=best_epoch))
best_df = pd.concat(best_rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(LABELS))
width = 0.25
for i, (cname, color) in enumerate(COLORS.items()):
    row = best_df[best_df["condition"] == cname].iloc[0]
    f1_vals = [row[f"label_f1__{lbl}"] for lbl in LABELS]
    bars = ax.bar(x + i * width, f1_vals, width, label=cname, color=color, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels([l.replace("_", "\n") for l in LABELS], fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Label F1 at Best Macro-F1 Epoch (mean across runs)", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/fig2_per_label_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_per_label_f1.png")

# ── Figure 3: summary bar chart at last epoch ─────────────────────────────────
last_epoch = data["epoch"].max()
last = data[data["epoch"] == last_epoch]
summary_metrics = ["exact_match_accuracy","micro_f1","macro_f1","roc_auc_macro","hamming_loss","jaccard_score"]
summary_labels  = ["Exact\nMatch","Micro\nF1","Macro\nF1","ROC-AUC\nMacro","Hamming\nLoss","Jaccard"]

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(summary_metrics))
width = 0.25
for i, (cname, color) in enumerate(COLORS.items()):
    sub = last[last["condition"] == cname]
    vals = [sub[m].mean() for m in summary_metrics]
    errs = [sub[m].std() for m in summary_metrics]
    ax.bar(x + i * width, vals, width, yerr=errs, label=cname, color=color, alpha=0.85,
           capsize=3, error_kw=dict(elinewidth=1.2))

ax.set_xticks(x + width)
ax.set_xticklabels(summary_labels, fontsize=10)
ax.set_ylabel("Score")
ax.set_title(f"Summary Metrics at Final Epoch (epoch {last_epoch}) — mean ± std across runs", fontsize=12, fontweight="bold")
ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{ARTIFACT_DIR}/fig3_final_epoch_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_final_epoch_summary.png")

# ── Print numeric summary ─────────────────────────────────────────────────────
print("\n=== BEST-EPOCH SUMMARY (highest mean macro_f1 across runs) ===")
for cname in COLORS:
    sub = data[data["condition"] == cname]
    by_ep = sub.groupby("epoch")[["macro_f1","micro_f1","exact_match_accuracy","roc_auc_macro","hamming_loss","jaccard_score","validation_loss"]].mean()
    best_ep = by_ep["macro_f1"].idxmax()
    row = by_ep.loc[best_ep]
    n_runs = sub["run_id"].nunique()
    print(f"\n{cname} — best epoch {best_ep} (across {n_runs} runs):")
    for k, v in row.items():
        print(f"  {k:30s}: {v:.4f}")

print("\n=== FINAL EPOCH SUMMARY ===")
last_ep = data["epoch"].max()
for cname in COLORS:
    sub = data[(data["condition"] == cname) & (data["epoch"] == last_ep)]
    print(f"\n{cname} (epoch {last_ep}, n={len(sub)}):")
    for m in ["exact_match_accuracy","micro_f1","macro_f1","roc_auc_macro","hamming_loss","jaccard_score","validation_loss"]:
        print(f"  {m:30s}: {sub[m].mean():.4f} ± {sub[m].std():.4f}")
