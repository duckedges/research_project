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

# ============================================================
# HELD-OUT TEST OVERRIDE
# Test targets were never used for training/model selection.
# ============================================================

work = pd.read_csv(
    "data/task6/task6_af3_input_manifest.csv"
)

work = (
    work.loc[work["split"] == "test"]
    .copy()
    .sort_values("task6_index")
)

if len(work) != 8:
    raise RuntimeError(
        f"Expected 8 held-out test cases, found {len(work)}"
    )

FEATURE_OUT = Path(
    "data/task6/task6_head1_test_readout_features.npz"
)

print()
print("===== HELD-OUT TEST FEATURE EXTRACTION =====")
print("Test rows:", len(work))
print("Test indices:", work["task6_index"].tolist())
print("Output:", FEATURE_OUT)
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
