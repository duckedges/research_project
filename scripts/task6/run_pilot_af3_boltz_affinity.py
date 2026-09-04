#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch

from boltz.model.modules.affinity import AffinityModule


ROOT = Path("/projects/bentosprg6/linwx")
PROJECT = ROOT / "research_project"

parser = argparse.ArgumentParser()
parser.add_argument("--task6-index", type=int, required=True)
args = parser.parse_args()

MANIFEST = PROJECT / "data/task6/task6_af3_input_manifest.csv"

manifest = pd.read_csv(MANIFEST)

rows = manifest.loc[
    manifest["task6_index"].astype(int) == args.task6_index
]

if len(rows) != 1:
    raise RuntimeError(
        f"Expected exactly one manifest row for task6_index "
        f"{args.task6_index}; found {len(rows)}."
    )

row = rows.iloc[0]

TASK6_INDEX = int(row["task6_index"])
INPUT_NAME = str(row["input_name"])

BOLTZ_CKPT = ROOT / "cache/boltz/boltz2_aff.ckpt"

AF3_ROOT = (
    ROOT
    / "task6_runtime/pilot_af3"
    / INPUT_NAME
)

embedding_candidates = sorted(
    AF3_ROOT.rglob("*embeddings.npz")
)

if not embedding_candidates:
    raise FileNotFoundError(
        f"Could not locate AF3 embeddings under {AF3_ROOT}"
    )

EMBEDDINGS = embedding_candidates[0]

BRIDGE_META = (
    ROOT
    / "task6_runtime/bridge/pilot"
    / f"{INPUT_NAME}_bridge_meta.npz"
)

if not BRIDGE_META.exists():
    raise FileNotFoundError(
        f"Bridge metadata not found: {BRIDGE_META}"
    )

OUTDIR = ROOT / "task6_runtime/affinity/pilot"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTDIR / f"{INPUT_NAME}_affinity.json"


def extract_module_state(state, prefix):
    prefix_dot = prefix + "."
    out = {}

    for key, value in state.items():
        if key.startswith(prefix_dot):
            out[key[len(prefix_dot):]] = value

    return out


print("===== TASK 6: AF3 -> STANDALONE BOLTZ AFFINITY =====")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------------
# Load AF3 representations
# ---------------------------------------------------------------

emb = np.load(EMBEDDINGS)

single_np = emb["single_embeddings"].astype(np.float32)
pair_np = emb["pair_embeddings"].astype(np.float32)

n_tokens = single_np.shape[0]

print()
print("===== AF3 REPRESENTATIONS =====")
print("single:", single_np.shape)
print("pair:  ", pair_np.shape)

assert single_np.shape == (n_tokens, 384)
assert pair_np.shape == (n_tokens, n_tokens, 128)

# ---------------------------------------------------------------
# Load bridge metadata
# ---------------------------------------------------------------

meta = np.load(BRIDGE_META)

x_pred_np = meta["x_pred"].astype(np.float32)
token_to_rep_atom_np = meta["token_to_rep_atom"].astype(np.float32)
token_pad_mask_np = meta["token_pad_mask"].astype(np.float32)
mol_type_np = meta["mol_type"].astype(np.int64)
affinity_token_mask_np = meta["affinity_token_mask"].astype(np.bool_)
cross_pair_mask_np = meta["cross_pair_mask"].astype(np.float32)

print()
print("===== BRIDGE FEATURES =====")
print("x_pred:              ", x_pred_np.shape)
print("token_to_rep_atom:   ", token_to_rep_atom_np.shape)
print("token_pad_mask:      ", token_pad_mask_np.shape)
print("mol_type:            ", mol_type_np.shape)
print("affinity_token_mask: ", affinity_token_mask_np.shape)
print("cross_pair_mask:     ", cross_pair_mask_np.shape)

# ---------------------------------------------------------------
# Convert AF3 data into the tensor shapes expected by AffinityModule
# ---------------------------------------------------------------

s_inputs = torch.from_numpy(single_np)[None, :, :].to(device)

z = torch.from_numpy(pair_np)[None, :, :, :].to(device)

cross_pair_mask = (
    torch.from_numpy(cross_pair_mask_np)
    .to(device)
)

# Native Boltz masks z before sending it into the affinity head.
z_affinity = z * cross_pair_mask[None, :, :, None]

x_pred = torch.from_numpy(x_pred_np).to(device)

feats = {
    "token_to_rep_atom": torch.from_numpy(
        token_to_rep_atom_np
    ).to(device),

    "token_pad_mask": torch.from_numpy(
        token_pad_mask_np
    ).to(device),

    "mol_type": torch.from_numpy(
        mol_type_np
    ).to(device),

    "affinity_token_mask": torch.from_numpy(
        affinity_token_mask_np
    ).to(device),
}

print()
print("===== BOLTZ INPUT SHAPES =====")
print("s_inputs:  ", tuple(s_inputs.shape))
print("z_affinity:", tuple(z_affinity.shape))
print("x_pred:    ", tuple(x_pred.shape))

# ---------------------------------------------------------------
# Load Boltz checkpoint configuration
# ---------------------------------------------------------------

print()
print("===== LOADING BOLTZ CHECKPOINT =====")

ckpt = torch.load(
    BOLTZ_CKPT,
    map_location="cpu",
    mmap=True,
    weights_only=False,
)

hp = ckpt["hyper_parameters"]
state = ckpt["state_dict"]

token_s = hp["token_s"]
token_z = hp["token_z"]

args1 = hp["affinity_model_args1"]
args2 = hp["affinity_model_args2"]

print("token_s:", token_s)
print("token_z:", token_z)
print("affinity ensemble:", hp["affinity_ensemble"])

assert token_s == 384
assert token_z == 128
assert hp["affinity_ensemble"] is True

# ---------------------------------------------------------------
# Instantiate only the two Boltz affinity heads
# ---------------------------------------------------------------

print()
print("===== INSTANTIATING AFFINITY HEAD 1 =====")

head1 = AffinityModule(
    token_s,
    token_z,
    **args1,
)

state1 = extract_module_state(
    state,
    "affinity_module1",
)

print("Head 1 tensors:", len(state1))

result1 = head1.load_state_dict(
    state1,
    strict=True,
)

print(
    "Head 1 missing/unexpected:",
    result1.missing_keys,
    result1.unexpected_keys,
)

print()
print("===== INSTANTIATING AFFINITY HEAD 2 =====")

head2 = AffinityModule(
    token_s,
    token_z,
    **args2,
)

state2 = extract_module_state(
    state,
    "affinity_module2",
)

print("Head 2 tensors:", len(state2))

result2 = head2.load_state_dict(
    state2,
    strict=True,
)

print(
    "Head 2 missing/unexpected:",
    result2.missing_keys,
    result2.unexpected_keys,
)

# Checkpoint is no longer needed.
del state
del ckpt

head1 = head1.to(device).eval()
head2 = head2.to(device).eval()

# ---------------------------------------------------------------
# Standalone forward pass
# ---------------------------------------------------------------

print()
print("===== RUNNING STANDALONE AFFINITY HEADS =====")

with torch.inference_mode():

    out1 = head1(
        s_inputs=s_inputs,
        z=z_affinity,
        x_pred=x_pred,
        feats=feats,
        multiplicity=1,
        use_kernels=False,
    )

    out2 = head2(
        s_inputs=s_inputs,
        z=z_affinity,
        x_pred=x_pred,
        feats=feats,
        multiplicity=1,
        use_kernels=False,
    )

head1_value = float(
    out1["affinity_pred_value"]
    .detach()
    .cpu()
    .reshape(-1)[0]
)

head2_value = float(
    out2["affinity_pred_value"]
    .detach()
    .cpu()
    .reshape(-1)[0]
)

head1_logit = float(
    out1["affinity_logits_binary"]
    .detach()
    .cpu()
    .reshape(-1)[0]
)

head2_logit = float(
    out2["affinity_logits_binary"]
    .detach()
    .cpu()
    .reshape(-1)[0]
)

head1_prob = float(
    torch.sigmoid(
        out1["affinity_logits_binary"]
    )
    .detach()
    .cpu()
    .reshape(-1)[0]
)

head2_prob = float(
    torch.sigmoid(
        out2["affinity_logits_binary"]
    )
    .detach()
    .cpu()
    .reshape(-1)[0]
)

# This exactly matches boltz2.py ensemble logic.
ensemble_value = (
    head1_value + head2_value
) / 2.0

ensemble_prob = (
    head1_prob + head2_prob
) / 2.0

print()
print("===== TASK 6 RESULT =====")
print("Head 1 affinity_pred_value:", head1_value)
print("Head 2 affinity_pred_value:", head2_value)
print("Ensemble affinity_pred_value:", ensemble_value)

print()
print("Head 1 binary probability:", head1_prob)
print("Head 2 binary probability:", head2_prob)
print("Ensemble binary probability:", ensemble_prob)

all_finite = all(
    np.isfinite(x)
    for x in [
        head1_value,
        head2_value,
        ensemble_value,
        head1_prob,
        head2_prob,
        ensemble_prob,
    ]
)

print()
print("All outputs finite:", all_finite)

result = {
    "complex": INPUT_NAME,
    "af3_tokens": int(n_tokens),
    "protein_tokens": int((mol_type_np[0] == 0).sum()),
    "aptamer_tokens": int(affinity_token_mask_np[0].sum()),

    "bridge_method": (
        "AF3 single/pair embeddings substituted for "
        "Boltz token representations; AF3 representative "
        "coordinates supplied directly to isolated "
        "Boltz-2 affinity ensemble."
    ),

    "head1_affinity_pred_value": head1_value,
    "head2_affinity_pred_value": head2_value,
    "ensemble_affinity_pred_value": ensemble_value,

    "head1_binary_probability": head1_prob,
    "head2_binary_probability": head2_prob,
    "ensemble_binary_probability": ensemble_prob,

    "all_outputs_finite": bool(all_finite),
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
    )

print()
print("Saved:", OUTPUT_JSON)

print()
print("===== SUCCESS =====")
print(
    "AF3 embeddings were passed through the isolated "
    "pretrained Boltz-2 affinity ensemble."
)
