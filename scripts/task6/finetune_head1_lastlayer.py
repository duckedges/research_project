#!/usr/bin/env python3

from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from boltz.model.modules.affinity import AffinityModule


ROOT = Path("/projects/bentosprg6/linwx")
PROJECT = ROOT / "research_project"

MANIFEST = PROJECT / "data/task6/task6_af3_input_manifest.csv"
CALIBRATION = PROJECT / "data/task6/task6_calibration_coefficients.csv"
CALIBRATED_PREDS = PROJECT / "data/task6/task6_pilot_calibrated_predictions.csv"

AF3_ROOT = ROOT / "task6_runtime/pilot_af3"
BRIDGE_ROOT = ROOT / "task6_runtime/bridge/pilot"

BOLTZ_CKPT = ROOT / "cache/boltz/boltz2_aff.ckpt"

FEATURE_OUT = PROJECT / "data/task6/task6_head1_readout_features.npz"
PRED_OUT = PROJECT / "data/task6/task6_head1_finetune_trainval_predictions.csv"
METRIC_OUT = PROJECT / "data/task6/task6_head1_finetune_metrics.csv"
MODEL_OUT = PROJECT / "data/task6/task6_head1_lastlayer_finetuned.pt"


def extract_module_state(state, prefix):
    prefix_dot = prefix + "."
    out = {}

    for key, value in state.items():
        if key.startswith(prefix_dot):
            out[key[len(prefix_dot):]] = value

    return out


def find_embeddings(input_name):
    case_root = AF3_ROOT / input_name

    candidates = sorted(
        case_root.rglob(f"{input_name}_seed-42_embeddings.npz")
    )

    if not candidates:
        candidates = sorted(case_root.rglob("*embeddings.npz"))

    if not candidates:
        raise FileNotFoundError(
            f"No AF3 embeddings found for {input_name}"
        )

    return candidates[0]


def load_case(row, device):
    input_name = str(row["input_name"])

    emb_path = find_embeddings(input_name)

    bridge_path = (
        BRIDGE_ROOT /
        f"{input_name}_bridge_meta.npz"
    )

    if not bridge_path.exists():
        raise FileNotFoundError(
            f"Missing bridge file: {bridge_path}"
        )

    emb = np.load(emb_path)
    bridge = np.load(bridge_path)

    single = np.asarray(
        emb["single_embeddings"],
        dtype=np.float32,
    )

    pair = np.asarray(
        emb["pair_embeddings"],
        dtype=np.float32,
    )

    if single.ndim == 2:
        single = single[None, ...]

    if pair.ndim == 3:
        pair = pair[None, ...]

    s_inputs = torch.from_numpy(single).to(
        device=device,
        dtype=torch.float32,
    )

    z = torch.from_numpy(pair).to(
        device=device,
        dtype=torch.float32,
    )

    x_pred = torch.from_numpy(
        np.asarray(bridge["x_pred"], dtype=np.float32)
    ).to(
        device=device,
        dtype=torch.float32,
    )

    feats = {
        "token_to_rep_atom": torch.from_numpy(
            np.asarray(
                bridge["token_to_rep_atom"],
                dtype=np.float32,
            )
        ).to(device),

        "token_pad_mask": torch.from_numpy(
            np.asarray(
                bridge["token_pad_mask"],
                dtype=np.float32,
            )
        ).to(device),

        "mol_type": torch.from_numpy(
            np.asarray(
                bridge["mol_type"],
                dtype=np.int64,
            )
        ).to(device),

        "affinity_token_mask": torch.from_numpy(
            np.asarray(
                bridge["affinity_token_mask"],
                dtype=bool,
            )
        ).to(device),
    }

    return s_inputs, z, x_pred, feats


def metric_dict(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    mae = np.mean(np.abs(p - y))
    rmse = np.sqrt(np.mean((p - y) ** 2))

    pearson = (
        np.corrcoef(y, p)[0, 1]
        if len(y) > 1
        and np.std(y) > 0
        and np.std(p) > 0
        else np.nan
    )

    yr = pd.Series(y).rank(method="average").to_numpy()
    pr = pd.Series(p).rank(method="average").to_numpy()

    spearman = (
        np.corrcoef(yr, pr)[0, 1]
        if len(y) > 1
        and np.std(yr) > 0
        and np.std(pr) > 0
        else np.nan
    )

    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "r2": float(r2),
    }


print("===== TASK 6 BOLTZ HEAD-1 LAST-LAYER FINE-TUNING =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Load manifest. TEST IS EXCLUDED HERE.
# ---------------------------------------------------------

manifest = pd.read_csv(MANIFEST)

work = manifest[
    manifest["split"].isin(["train", "validation"])
].copy()

print()
print("===== DATA SPLITS =====")
print(work["split"].value_counts())

assert len(work) == 56
assert (work["split"] == "train").sum() == 48
assert (work["split"] == "validation").sum() == 8
assert not (work["split"] == "test").any()

print("Test rows loaded into fine-tuning:", 0)


# ---------------------------------------------------------
# Load pretrained Boltz affinity head 1.
# ---------------------------------------------------------

print()
print("===== LOADING PRETRAINED BOLTZ HEAD 1 =====")

try:
    ckpt = torch.load(
        BOLTZ_CKPT,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    ckpt = torch.load(
        BOLTZ_CKPT,
        map_location="cpu",
    )

state = ckpt["state_dict"]
hp = ckpt["hyper_parameters"]

token_s = int(hp["token_s"])
token_z = int(hp["token_z"])

args1 = hp["affinity_model_args1"]

head1 = AffinityModule(
    token_s,
    token_z,
    **args1,
)

state1 = extract_module_state(
    state,
    "affinity_module1",
)

result = head1.load_state_dict(
    state1,
    strict=True,
)

print("Head 1 tensors:", len(state1))
print(
    "Missing/unexpected:",
    result.missing_keys,
    result.unexpected_keys,
)


# ---------------------------------------------------------
# Locate final Boltz affinity regression layer.
# ---------------------------------------------------------

regressor = head1.affinity_heads.to_affinity_pred_value

linear_layers = [
    m
    for m in regressor.modules()
    if isinstance(m, nn.Linear)
]

if not linear_layers:
    raise RuntimeError(
        "No Linear layer found in affinity prediction head."
    )

final_linear = linear_layers[-1]

print()
print("===== FINAL REGRESSION LAYER =====")
print(final_linear)
print(
    "Parameters:",
    final_linear.weight.numel()
    + (
        final_linear.bias.numel()
        if final_linear.bias is not None
        else 0
    ),
)


# ---------------------------------------------------------
# Fold TRAIN-ONLY affine calibration into actual Boltz
# final regression layer as stable initialization.
# ---------------------------------------------------------

coef = pd.read_csv(CALIBRATION)

coef = coef[
    coef["predictor"] == "head1_pred"
].iloc[0]

slope = float(coef["slope"])
intercept = float(coef["intercept"])

print()
print("===== TRAIN-ONLY CALIBRATED INITIALIZATION =====")
print("Slope:", slope)
print("Intercept:", intercept)

with torch.no_grad():
    final_linear.weight.mul_(slope)

    if final_linear.bias is not None:
        final_linear.bias.mul_(slope)
        final_linear.bias.add_(intercept)


head1 = head1.to(device)
head1.eval()

# Re-acquire reference after .to(device)
regressor = head1.affinity_heads.to_affinity_pred_value
linear_layers = [
    m
    for m in regressor.modules()
    if isinstance(m, nn.Linear)
]
final_linear = linear_layers[-1]


# ---------------------------------------------------------
# Extract the frozen representation entering the actual
# Boltz final regression layer.
# ---------------------------------------------------------

capture = {}


def capture_input(module, inputs):
    capture["x"] = (
        inputs[0]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .clone()
    )


hook = final_linear.register_forward_pre_hook(
    capture_input
)

features = []
labels = []
splits = []
input_names = []
task6_indices = []
initial_predictions = []


print()
print("===== EXTRACTING FROZEN BOLTZ READOUT FEATURES =====")

with torch.no_grad():

    for n, (_, row) in enumerate(work.iterrows(), start=1):

        input_name = str(row["input_name"])

        s_inputs, z, x_pred, feats = load_case(
            row,
            device,
        )

        capture.clear()

        out = head1(
            s_inputs=s_inputs,
            z=z,
            x_pred=x_pred,
            feats=feats,
            multiplicity=1,
            use_kernels=False,
        )

        if "x" not in capture:
            raise RuntimeError(
                f"Final-layer hook failed for {input_name}"
            )

        features.append(
            capture["x"].numpy()
        )

        labels.append(
            float(row["boltz_target"])
        )

        splits.append(
            str(row["split"])
        )

        input_names.append(input_name)

        task6_indices.append(
            int(row["task6_index"])
        )

        initial_predictions.append(
            float(
                out["affinity_pred_value"]
                .detach()
                .cpu()
                .reshape(-1)[0]
            )
        )

        print(
            f"[{n:02d}/56]",
            row["split"],
            input_name,
            "target=",
            round(float(row["boltz_target"]), 4),
            "initial=",
            round(initial_predictions[-1], 4),
        )

        del s_inputs, z, x_pred, feats, out

        if device.type == "cuda":
            torch.cuda.empty_cache()


hook.remove()


X = np.stack(features).astype(np.float32)
y = np.asarray(labels, dtype=np.float32)

splits = np.asarray(splits)
input_names = np.asarray(input_names)
task6_indices = np.asarray(task6_indices)

print()
print("Feature matrix:", X.shape)
print("Targets:", y.shape)


# ---------------------------------------------------------
# Save extracted features for reproducibility.
# ---------------------------------------------------------

np.savez_compressed(
    FEATURE_OUT,
    X=X,
    y=y,
    split=splits,
    input_name=input_names,
    task6_index=task6_indices,
)

print("Saved features:", FEATURE_OUT)


# ---------------------------------------------------------
# TRAIN / VALIDATION ONLY.
# ---------------------------------------------------------

train_mask = splits == "train"
val_mask = splits == "validation"

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]


# Standardize using TRAIN ONLY.
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)

sigma[sigma < 1e-6] = 1.0

X_train_z = (
    (X_train - mu) / sigma
).astype(np.float32)

X_val_z = (
    (X_val - mu) / sigma
).astype(np.float32)


# ---------------------------------------------------------
# Convert calibrated Boltz final-layer initialization to
# standardized-feature coordinates.
# ---------------------------------------------------------

w_orig = (
    final_linear.weight
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
    .astype(np.float32)
)

if final_linear.bias is not None:
    b_orig = float(
        final_linear.bias
        .detach()
        .cpu()
        .reshape(-1)[0]
    )
else:
    b_orig = 0.0

w_z_init = w_orig * sigma

b_z_init = (
    b_orig
    + float(np.dot(w_orig, mu))
)


readout = nn.Linear(
    X.shape[1],
    1,
    bias=True,
)

with torch.no_grad():
    readout.weight.copy_(
        torch.from_numpy(
            w_z_init.reshape(1, -1)
        )
    )
    readout.bias.copy_(
        torch.tensor([b_z_init])
    )


Xt = torch.from_numpy(X_train_z)
yt = torch.from_numpy(y_train).reshape(-1, 1)

Xv = torch.from_numpy(X_val_z)
yv = torch.from_numpy(y_val).reshape(-1, 1)


criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    readout.parameters(),
    lr=1e-2,
    weight_decay=1e-3,
)


# ---------------------------------------------------------
# Initial metrics before gradient updates.
# ---------------------------------------------------------

readout.eval()

with torch.no_grad():
    initial_train = (
        readout(Xt).reshape(-1).numpy()
    )
    initial_val = (
        readout(Xv).reshape(-1).numpy()
    )

initial_train_metrics = metric_dict(
    y_train,
    initial_train,
)

initial_val_metrics = metric_dict(
    y_val,
    initial_val,
)

print()
print("===== INITIAL CALIBRATED METRICS =====")
print("Train:", initial_train_metrics)
print("Validation:", initial_val_metrics)


# ---------------------------------------------------------
# Fine-tune ONLY the actual final Boltz regression layer.
# Early stopping uses VALIDATION only.
# ---------------------------------------------------------

torch.manual_seed(42)

best_state = copy.deepcopy(
    readout.state_dict()
)

best_epoch = 0
best_val_rmse = initial_val_metrics["rmse"]

patience = 100
bad_epochs = 0
max_epochs = 1000


print()
print("===== FINE-TUNING =====")

for epoch in range(1, max_epochs + 1):

    readout.train()

    optimizer.zero_grad()

    pred = readout(Xt)

    loss = criterion(
        pred,
        yt,
    )

    loss.backward()
    optimizer.step()

    readout.eval()

    with torch.no_grad():

        train_pred = (
            readout(Xt)
            .reshape(-1)
            .numpy()
        )

        val_pred = (
            readout(Xv)
            .reshape(-1)
            .numpy()
        )

    val_rmse = float(
        np.sqrt(
            np.mean(
                (val_pred - y_val) ** 2
            )
        )
    )

    if val_rmse < best_val_rmse - 1e-8:

        best_val_rmse = val_rmse
        best_epoch = epoch
        best_state = copy.deepcopy(
            readout.state_dict()
        )

        bad_epochs = 0

    else:
        bad_epochs += 1

    if epoch == 1 or epoch % 50 == 0:
        print(
            "epoch=",
            epoch,
            "train_loss=",
            round(float(loss.item()), 6),
            "val_rmse=",
            round(val_rmse, 6),
            "best=",
            round(best_val_rmse, 6),
        )

    if bad_epochs >= patience:
        print(
            "Early stopping at epoch",
            epoch,
        )
        break


readout.load_state_dict(best_state)
readout.eval()


with torch.no_grad():

    train_pred = (
        readout(Xt)
        .reshape(-1)
        .numpy()
    )

    val_pred = (
        readout(Xv)
        .reshape(-1)
        .numpy()
    )


final_train_metrics = metric_dict(
    y_train,
    train_pred,
)

final_val_metrics = metric_dict(
    y_val,
    val_pred,
)


print()
print("===== BEST FINE-TUNED MODEL =====")
print("Best epoch:", best_epoch)
print("Train:", final_train_metrics)
print("Validation:", final_val_metrics)


# ---------------------------------------------------------
# Convert best standardized linear weights back to original
# Boltz feature coordinates.
# ---------------------------------------------------------

w_z = (
    readout.weight
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)

b_z = float(
    readout.bias
    .detach()
    .cpu()
    .reshape(-1)[0]
)

w_final = w_z / sigma

b_final = (
    b_z
    - float(np.dot(w_final, mu))
)


with torch.no_grad():

    final_linear.weight.copy_(
        torch.from_numpy(
            w_final.reshape(1, -1)
        ).to(
            device=final_linear.weight.device,
            dtype=final_linear.weight.dtype,
        )
    )

    if final_linear.bias is not None:
        final_linear.bias.copy_(
            torch.tensor(
                [b_final],
                device=final_linear.bias.device,
                dtype=final_linear.bias.dtype,
            )
        )


# ---------------------------------------------------------
# Save train/validation predictions.
# TEST IS STILL ABSENT.
# ---------------------------------------------------------

all_pred = np.empty(len(work), dtype=float)

all_pred[train_mask] = train_pred
all_pred[val_mask] = val_pred

pred_df = work[
    [
        "task6_index",
        "input_name",
        "model_id",
        "split",
        "boltz_target",
    ]
].copy()

pred_df["initial_calibrated_pred"] = (
    initial_predictions
)

pred_df["finetuned_head1_pred"] = (
    all_pred
)

pred_df.to_csv(
    PRED_OUT,
    index=False,
)


metric_rows = []

for stage, tr, va in [
    (
        "calibrated_initial",
        initial_train_metrics,
        initial_val_metrics,
    ),
    (
        "finetuned",
        final_train_metrics,
        final_val_metrics,
    ),
]:

    metric_rows.append({
        "stage": stage,
        "split": "train",
        **tr,
    })

    metric_rows.append({
        "stage": stage,
        "split": "validation",
        **va,
    })


metric_df = pd.DataFrame(metric_rows)

metric_df.to_csv(
    METRIC_OUT,
    index=False,
)


# ---------------------------------------------------------
# Save actual modified Boltz affinity-head weights.
# ---------------------------------------------------------

state_cpu = {
    k: v.detach().cpu()
    for k, v in head1.state_dict().items()
}

torch.save(
    {
        "task": "Task 6 AF3-to-Boltz affinity adaptation",
        "method": "Head-1 final regression layer fine-tuning",
        "base_checkpoint": str(BOLTZ_CKPT),
        "token_s": token_s,
        "token_z": token_z,
        "affinity_model_args1": args1,
        "state_dict": state_cpu,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "training_rows": 48,
        "validation_rows": 8,
        "test_rows_used": 0,
    },
    MODEL_OUT,
)


print()
print("===== SAVED =====")
print("Predictions:", PRED_OUT)
print("Metrics:", METRIC_OUT)
print("Fine-tuned model:", MODEL_OUT)

print()
print("===== SUCCESS =====")
print(
    "Boltz-2 Head 1 final regression layer was "
    "fine-tuned using 48 train cases and selected "
    "using 8 validation cases."
)
print("Test cases used during fine-tuning: 0")
