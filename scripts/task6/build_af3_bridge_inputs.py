#!/usr/bin/env python3

import json
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path("/projects/bentosprg6/linwx")

INPUT_JSON = (
    ROOT
    / "research_project/af3_inputs/utexas_eif4e_apt1_no_msa.json"
)

AF3_ROOT = (
    ROOT
    / "task6_runtime/af3_output/eif4e_apt1_embeddings"
    / "utexas_eif4e_apt1_no_msa"
)

EMBEDDINGS = (
    AF3_ROOT
    / "seed-42_embeddings"
    / "utexas_eif4e_apt1_no_msa_seed-42_embeddings.npz"
)

OUTDIR = ROOT / "task6_runtime/bridge"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "eif4e_af3_bridge_meta.npz"


def find_model_cif():
    candidates = sorted(AF3_ROOT.rglob("*_model.cif"))

    # Prefer the top-level AF3 selected model rather than individual samples.
    preferred = [
        p for p in candidates
        if "seed-" not in str(p)
    ]

    if preferred:
        return preferred[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError("Could not locate AF3 model CIF.")


def normalize_ids(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    raise TypeError(f"Unexpected chain id field: {value!r}")


def get_rep_atom(residue, mol_kind):
    """
    Return a representative coordinate for one AF3 polymer token.

    Protein: CA
    RNA/DNA: C4' preferred, then C1', then P.
    """

    if mol_kind == "protein":
        preferred = ["CA"]
    elif mol_kind in ("rna", "dna"):
        preferred = ["C4'", "C1'", "P"]
    else:
        raise ValueError(f"Unsupported polymer type: {mol_kind}")

    for atom_name in preferred:
        atom = residue.find_atom(atom_name, "\0")
        if atom is not None:
            pos = atom.pos
            return np.array(
                [pos.x, pos.y, pos.z],
                dtype=np.float32,
            ), atom_name

    # Last-resort fallback for prototype.
    atoms = list(residue)
    if atoms:
        pos = atoms[0].pos
        return np.array(
            [pos.x, pos.y, pos.z],
            dtype=np.float32,
        ), atoms[0].name

    raise RuntimeError(
        f"No atoms found for residue {residue.name}"
    )


print("===== TASK 6: BUILD AF3 -> BOLTZ BRIDGE INPUTS =====")

print("Input JSON:", INPUT_JSON)
print("Embeddings:", EMBEDDINGS)

model_cif = find_model_cif()
print("Model CIF:", model_cif)

# ------------------------------------------------------------------
# Load AF3 representations
# ------------------------------------------------------------------

emb = np.load(EMBEDDINGS)

single = emb["single_embeddings"].astype(np.float32)
pair = emb["pair_embeddings"].astype(np.float32)

n_tokens = single.shape[0]

assert single.shape == (n_tokens, 384)
assert pair.shape == (n_tokens, n_tokens, 128)

print()
print("AF3 token count:", n_tokens)
print("single:", single.shape)
print("pair:  ", pair.shape)

# ------------------------------------------------------------------
# Read AF3 input schema / polymer ordering
# ------------------------------------------------------------------

with open(INPUT_JSON) as f:
    af3_input = json.load(f)

sequence_entries = af3_input["sequences"]

token_plan = []

for entry in sequence_entries:
    if "protein" in entry:
        kind = "protein"
        obj = entry[kind]
        mol_type_value = 0

    elif "rna" in entry:
        kind = "rna"
        obj = entry[kind]
        mol_type_value = 1

    elif "dna" in entry:
        kind = "dna"
        obj = entry[kind]
        mol_type_value = 2

    else:
        # We deliberately stop here for this first protein-aptamer
        # bridge instead of silently mis-tokenizing ligands.
        key = next(iter(entry))
        raise ValueError(
            f"Prototype bridge encountered unsupported entity: {key}"
        )

    ids = normalize_ids(obj["id"])
    sequence = obj["sequence"]

    for chain_id in ids:
        for residue_index, _ in enumerate(sequence, start=1):
            token_plan.append(
                {
                    "chain": chain_id,
                    "residue_index": residue_index,
                    "kind": kind,
                    "mol_type": mol_type_value,
                }
            )

print()
print("Tokens implied by AF3 JSON:", len(token_plan))

if len(token_plan) != n_tokens:
    raise RuntimeError(
        "AF3 token count does not match polymer token count: "
        f"{len(token_plan)} vs {n_tokens}"
    )

# ------------------------------------------------------------------
# Parse AF3 predicted structure
# ------------------------------------------------------------------

structure = gemmi.read_structure(str(model_cif))

if len(structure) == 0:
    raise RuntimeError("No model found in AF3 CIF.")

model = structure[0]

chains = {chain.name: chain for chain in model}

print()
print("CIF chains:", sorted(chains))

coords = []
mol_types = []
binder_mask = []
token_labels = []
fallback_atoms = []

# Build polymer residue lists by chain.
polymer_residues = {}

for chain_id, chain in chains.items():
    polymer_residues[chain_id] = list(chain.get_polymer())

chain_offsets = {}

for token in token_plan:
    chain_id = token["chain"]

    if chain_id not in polymer_residues:
        raise RuntimeError(
            f"Chain {chain_id!r} from AF3 JSON not found in CIF. "
            f"CIF chains={sorted(polymer_residues)}"
        )

    idx = chain_offsets.get(chain_id, 0)
    residues = polymer_residues[chain_id]

    if idx >= len(residues):
        raise RuntimeError(
            f"Not enough residues in CIF chain {chain_id}: "
            f"needed index {idx}, found {len(residues)} residues"
        )

    residue = residues[idx]
    chain_offsets[chain_id] = idx + 1

    xyz, atom_used = get_rep_atom(
        residue,
        token["kind"],
    )

    coords.append(xyz)
    mol_types.append(token["mol_type"])

    # For this first test:
    # protein = receptor
    # RNA/DNA = aptamer binder
    is_binder = token["kind"] in ("rna", "dna")
    binder_mask.append(1 if is_binder else 0)

    token_labels.append(
        f"{chain_id}:{idx + 1}:{residue.name}:{atom_used}"
    )

    expected = (
        "CA"
        if token["kind"] == "protein"
        else "C4'"
    )

    if atom_used != expected:
        fallback_atoms.append(
            (token_labels[-1], expected, atom_used)
        )

coords = np.stack(coords).astype(np.float32)
mol_types = np.asarray(mol_types, dtype=np.int64)
binder_mask = np.asarray(binder_mask, dtype=np.bool_)

if coords.shape != (n_tokens, 3):
    raise RuntimeError(
        f"Coordinate shape mismatch: {coords.shape}"
    )

# ------------------------------------------------------------------
# Minimal Boltz affinity features
# ------------------------------------------------------------------

token_pad_mask = np.ones(
    (1, n_tokens),
    dtype=np.float32,
)

mol_type = mol_types[None, :]
affinity_token_mask = binder_mask[None, :]

# Since x_pred already contains one representative coordinate
# per token, token_to_rep_atom can be the identity matrix.
token_to_rep_atom = np.eye(
    n_tokens,
    dtype=np.float32,
)[None, :, :]

x_pred = coords[None, :, :]

# Reproduce the cross_pair_mask used in boltz2.py:
# receptor<->binder plus binder<->binder.
rec = (mol_type[0] == 0) & (token_pad_mask[0] > 0)
lig = affinity_token_mask[0] & (token_pad_mask[0] > 0)

cross_pair_mask = (
    lig[:, None] * rec[None, :]
    + rec[:, None] * lig[None, :]
    + lig[:, None] * lig[None, :]
).astype(np.float32)

# Prevent diagonal just as the AffinityModule does internally.
cross_pair_mask *= (
    1.0 - np.eye(n_tokens, dtype=np.float32)
)

# Save only bridge metadata; AF3 embeddings remain in their
# original NPZ.
np.savez_compressed(
    OUTPUT,
    x_pred=x_pred,
    token_to_rep_atom=token_to_rep_atom,
    token_pad_mask=token_pad_mask,
    mol_type=mol_type,
    affinity_token_mask=affinity_token_mask,
    cross_pair_mask=cross_pair_mask,
    token_labels=np.asarray(token_labels),
)

print()
print("===== BRIDGE QC =====")
print("x_pred:             ", x_pred.shape)
print("token_to_rep_atom:  ", token_to_rep_atom.shape)
print("token_pad_mask:     ", token_pad_mask.shape)
print("mol_type:           ", mol_type.shape)
print("affinity_token_mask:", affinity_token_mask.shape)
print("cross_pair_mask:    ", cross_pair_mask.shape)

print()
print("Protein/receptor tokens:", int(rec.sum()))
print("Aptamer/binder tokens:  ", int(lig.sum()))
print("Total tokens:            ", n_tokens)

print()
print("Representative-atom fallbacks:", len(fallback_atoms))

for item in fallback_atoms[:10]:
    print("  ", item)

print()
print("All coordinates finite:", bool(np.isfinite(x_pred).all()))
print("All single finite:     ", bool(np.isfinite(single).all()))
print("All pair finite:       ", bool(np.isfinite(pair).all()))

print()
print("Saved:", OUTPUT)

print()
print("===== SUCCESS =====")
print("AF3 bridge metadata constructed successfully.")
