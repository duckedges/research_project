#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import torch


ROOT = Path("/projects/bentosprg6/linwx/research_project")

TRAINVAL_FEATURES = ROOT / "data/task6/task6_head1_readout_features.npz"
TEST_FEATURES = ROOT / "data/task6/task6_head1_test_readout_features.npz"

CHECKPOINT = ROOT / "data/task6/task6_head1_lastlayer_finetuned.pt"

BASELINE = ROOT / "data/task6/task6_pilot_calibrated_predictions.csv"

PRED_OUT = ROOT / "data/task6/task6_head1_heldout_test_predictions.csv"
METRIC_OUT = ROOT / "data/task6/task6_head1_heldout_test_metrics.csv"


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))

    if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0:
        pearson = float(np.corrcoef(y, p)[0, 1])
    else:
        pearson = np.nan

    yr = pd.Series(y).rank(method="average").to_numpy()
    pr = pd.Series(p).rank(method="average").to_numpy()

    if len(y) > 1 and np.std(yr) > 0 and np.std(pr) > 0:
        spearman = float(np.corrcoef(yr, pr)[0, 1])
    else:
        spearman = np.nan

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    r2 = (
        float(1.0 - ss_res / ss_tot)
        if ss_tot > 0
        else np.nan
    )

    return {
        "n": len(y),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2": r2,
    }


print("===== TASK 6 HELD-OUT TEST EVALUATION =====")

# ------------------------------------------------------------
# Load checkpoint.
# ------------------------------------------------------------

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

state = ckpt["state_dict"]

print()
print("Checkpoint:", CHECKPOINT)
print("Best epoch:", ckpt.get("best_epoch"))
print("Stored best validation RMSE:", ckpt.get("best_validation_rmse"))
print("Training rows:", ckpt.get("training_rows"))
print("Validation rows:", ckpt.get("validation_rows"))
print("Test rows used during training:", ckpt.get("test_rows_used"))


# ------------------------------------------------------------
# Locate the fine-tuned final regression layer automatically.
# ------------------------------------------------------------

candidates = []

for k, v in state.items():

    if (
        torch.is_tensor(v)
        and k.endswith(".weight")
        and tuple(v.shape) == (1, 384)
        and "to_affinity_pred_value" in k
    ):
        candidates.append(k)

print()
print("Final-layer weight candidates:", candidates)

if len(candidates) != 1:
    raise RuntimeError(
        "Expected exactly one 384->1 affinity prediction "
        f"weight tensor, found: {candidates}"
    )

weight_key = candidates[0]
bias_key = weight_key.rsplit(".", 1)[0] + ".bias"

if bias_key not in state:
    raise RuntimeError(
        f"Bias corresponding to {weight_key} was not found."
    )

w = (
    state[weight_key]
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float64)
)

b = float(
    state[bias_key]
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)[0]
)

print("Weight key:", weight_key)
print("Bias key:", bias_key)
print("Weight shape:", w.shape)
print("Bias:", b)


# ------------------------------------------------------------
# Recover train-only standardization statistics.
# ------------------------------------------------------------

tv = np.load(TRAINVAL_FEATURES, allow_pickle=True)

X_tv = tv["X"].astype(np.float64)
y_tv = tv["y"].astype(np.float64)
splits = tv["split"].astype(str)

train_mask = splits == "train"
val_mask = splits == "validation"

if train_mask.sum() != 48:
    raise RuntimeError(
        f"Expected 48 training rows, found {train_mask.sum()}"
    )

if val_mask.sum() != 8:
    raise RuntimeError(
        f"Expected 8 validation rows, found {val_mask.sum()}"
    )

mu = X_tv[train_mask].mean(axis=0)
sigma = X_tv[train_mask].std(axis=0)
sigma[sigma < 1e-6] = 1.0


# ------------------------------------------------------------
# Determine how the saved layer is represented.
#
# Candidate A:
#   checkpoint weights already transformed back to raw feature
#   coordinates.
#
# Candidate B:
#   checkpoint weights remain in standardized coordinates.
#
# Validation is used ONLY to confirm checkpoint reconstruction,
# not to retrain or select a new model.
# ------------------------------------------------------------

pred_raw = X_tv @ w + b

X_tv_z = (X_tv - mu) / sigma
pred_standardized = X_tv_z @ w + b

stored_val_rmse = float(
    ckpt["best_validation_rmse"]
)

raw_val_rmse = metrics(
    y_tv[val_mask],
    pred_raw[val_mask],
)["rmse"]

standardized_val_rmse = metrics(
    y_tv[val_mask],
    pred_standardized[val_mask],
)["rmse"]

print()
print("===== CHECKPOINT RECONSTRUCTION QC =====")
print("Stored validation RMSE:      ", stored_val_rmse)
print("Raw-feature validation RMSE: ", raw_val_rmse)
print(
    "Standardized validation RMSE:",
    standardized_val_rmse,
)

raw_error = abs(raw_val_rmse - stored_val_rmse)
z_error = abs(
    standardized_val_rmse - stored_val_rmse
)

if raw_error <= z_error:
    coordinate_mode = "raw"
    tv_pred = pred_raw
else:
    coordinate_mode = "standardized"
    tv_pred = pred_standardized

reconstructed_val_rmse = metrics(
    y_tv[val_mask],
    tv_pred[val_mask],
)["rmse"]

print("Selected coordinate mode:", coordinate_mode)
print(
    "Reconstructed validation RMSE:",
    reconstructed_val_rmse,
)

if abs(reconstructed_val_rmse - stored_val_rmse) > 1e-3:
    raise RuntimeError(
        "Saved checkpoint could not reproduce the stored "
        "validation RMSE. Stop before evaluating test cases."
    )

print("Checkpoint reconstruction: PASS")


# ------------------------------------------------------------
# Show train/validation metrics as an additional sanity check.
# ------------------------------------------------------------

print()
print("===== RECONSTRUCTED TRAIN / VALIDATION =====")

for split_name, mask in [
    ("train", train_mask),
    ("validation", val_mask),
]:
    m = metrics(
        y_tv[mask],
        tv_pred[mask],
    )

    print(split_name, m)


# ------------------------------------------------------------
# Load the untouched test feature vectors.
# ------------------------------------------------------------

test = np.load(TEST_FEATURES, allow_pickle=True)

X_test = test["X"].astype(np.float64)
y_test = test["y"].astype(np.float64)
test_indices = test["task6_index"].astype(int)
test_names = test["input_name"].astype(str)

if X_test.shape != (8, 384):
    raise RuntimeError(
        f"Expected test feature matrix (8, 384), "
        f"found {X_test.shape}"
    )

if len(y_test) != 8:
    raise RuntimeError(
        f"Expected 8 test targets, found {len(y_test)}"
    )


# ------------------------------------------------------------
# Apply the saved fine-tuned layer.
# ------------------------------------------------------------

if coordinate_mode == "raw":
    finetuned_pred = X_test @ w + b
else:
    X_test_z = (X_test - mu) / sigma
    finetuned_pred = X_test_z @ w + b

if not np.isfinite(finetuned_pred).all():
    raise RuntimeError(
        "Non-finite fine-tuned predictions detected."
    )


# ------------------------------------------------------------
# Load pretrained/calibrated baselines and align by task6_index.
# ------------------------------------------------------------

base = pd.read_csv(BASELINE)

base = (
    base.loc[base["split"] == "test"]
    .copy()
    .sort_values("task6_index")
)

if len(base) != 8:
    raise RuntimeError(
        f"Expected 8 baseline test rows, found {len(base)}"
    )

base = base.set_index("task6_index")

rows = []

for i, input_name, target, pred in zip(
    test_indices,
    test_names,
    y_test,
    finetuned_pred,
):
    if i not in base.index:
        raise RuntimeError(
            f"Test index {i} missing from baseline CSV."
        )

    r = base.loc[i]

    rows.append(
        {
            "task6_index": int(i),
            "input_name": input_name,
            "model_id": r["model_id"],
            "nucleic_acid_type": r["nucleic_acid_type"],
            "total_polymer_length": r["total_polymer_length"],
            "kd_nm": r["kd_nm"],
            "boltz_target": float(target),

            "pretrained_head1_pred": float(
                r["head1_pred"]
            ),

            "calibrated_head1_pred": float(
                r["head1_calibrated"]
            ),

            "finetuned_head1_pred": float(pred),

            "ensemble_calibrated_pred": float(
                r["ensemble_calibrated"]
            ),
        }
    )

pred_df = (
    pd.DataFrame(rows)
    .sort_values("task6_index")
    .reset_index(drop=True)
)

# Verify labels align with baseline file.
if not np.allclose(
    pred_df["boltz_target"].to_numpy(),
    base.loc[
        pred_df["task6_index"],
        "boltz_target",
    ].to_numpy(),
    atol=1e-6,
):
    raise RuntimeError(
        "Test targets do not match baseline manifest."
    )


# ------------------------------------------------------------
# Final untouched-test metrics.
# ------------------------------------------------------------

metric_rows = []

comparisons = [
    ("pretrained_head1", "pretrained_head1_pred"),
    ("calibrated_head1", "calibrated_head1_pred"),
    ("finetuned_head1", "finetuned_head1_pred"),
    ("calibrated_ensemble", "ensemble_calibrated_pred"),
]

target = pred_df["boltz_target"].to_numpy()

for name, col in comparisons:
    m = metrics(
        target,
        pred_df[col].to_numpy(),
    )

    metric_rows.append(
        {
            "model": name,
            **m,
        }
    )

metric_df = pd.DataFrame(metric_rows)

pred_df.to_csv(PRED_OUT, index=False)
metric_df.to_csv(METRIC_OUT, index=False)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

print()
print("===== 8 HELD-OUT TEST PREDICTIONS =====")

print(
    pred_df[
        [
            "task6_index",
            "model_id",
            "boltz_target",
            "calibrated_head1_pred",
            "finetuned_head1_pred",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("===== FINAL HELD-OUT TEST METRICS =====")

print(
    metric_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

baseline_rmse = float(
    metric_df.loc[
        metric_df["model"] == "calibrated_head1",
        "rmse",
    ].iloc[0]
)

finetuned_rmse = float(
    metric_df.loc[
        metric_df["model"] == "finetuned_head1",
        "rmse",
    ].iloc[0]
)

baseline_mae = float(
    metric_df.loc[
        metric_df["model"] == "calibrated_head1",
        "mae",
    ].iloc[0]
)

finetuned_mae = float(
    metric_df.loc[
        metric_df["model"] == "finetuned_head1",
        "mae",
    ].iloc[0]
)

print()
print("===== FINE-TUNING CHANGE VS CALIBRATED HEAD 1 =====")

print(
    "RMSE:",
    baseline_rmse,
    "->",
    finetuned_rmse,
    "change=",
    finetuned_rmse - baseline_rmse,
)

print(
    "MAE:",
    baseline_mae,
    "->",
    finetuned_mae,
    "change=",
    finetuned_mae - baseline_mae,
)

print()
print("Saved:", PRED_OUT)
print("Saved:", METRIC_OUT)

print()
print("===== PASS =====")
print(
    "Held-out test evaluation completed. "
    "The 8 test labels were not used for training, "
    "early stopping, or checkpoint selection."
)
