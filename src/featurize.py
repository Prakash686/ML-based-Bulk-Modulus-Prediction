"""Turn Matbench log_kvrh crystal structures into numerical descriptors.

Two groups of features:
  * Composition: fraction-weighted mean / std / min / max / range of elemental
    properties (Magpie-style).
  * Structure: lattice parameters, volume per atom, density, packing fraction
    and nearest-neighbour distance statistics.
"""
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Element, Structure

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "log_kvrh.json.gz"
OUT = ROOT / "data" / "features.csv"

ELEM_PROPS = {
    "Z": lambda e: e.Z,
    "atomic_mass": lambda e: float(e.atomic_mass),
    "electronegativity": lambda e: e.X,
    "atomic_radius": lambda e: e.atomic_radius,
    "melting_point": lambda e: e.melting_point,
    "row": lambda e: e.row,
    "group": lambda e: e.group,
    "molar_volume": lambda e: e.molar_volume,
    "ionization_energy": lambda e: e.ionization_energies[0] if e.ionization_energies else None,
    "valence_electrons": lambda e: sum(n for (_, _, n) in e.full_electronic_structure[-3:]),
}


def _elem_table():
    table = {}
    for e in Element:
        row = {}
        for name, fn in ELEM_PROPS.items():
            try:
                v = fn(e)
                row[name] = float(v) if v is not None else np.nan
            except Exception:
                row[name] = np.nan
        table[e.symbol] = row
    df = pd.DataFrame(table).T
    return df.fillna(df.mean())  # fill rare missing values with column mean


ELEM_TABLE = _elem_table()


def featurize(struct_dict):
    s = Structure.from_dict(struct_dict)
    comp = s.composition.fractional_composition
    els = [el.symbol for el in comp.elements]
    w = np.array([comp[el] for el in els])
    props = ELEM_TABLE.loc[els]

    f = {}
    for p in ELEM_TABLE.columns:
        x = props[p].to_numpy()
        mean = np.sum(w * x)
        f[f"{p}_mean"] = mean
        f[f"{p}_std"] = np.sqrt(np.sum(w * (x - mean) ** 2))
        f[f"{p}_min"] = x.min()
        f[f"{p}_max"] = x.max()
        f[f"{p}_range"] = x.max() - x.min()

    n = len(s)
    lat = s.lattice
    f["n_elements"] = len(els)
    f["n_sites"] = n
    f["volume_per_atom"] = s.volume / n
    f["density"] = float(s.density)
    f["a"], f["b"], f["c"] = lat.abc
    f["alpha"], f["beta"], f["gamma"] = lat.angles
    atom_vol = sum(4 / 3 * np.pi * ELEM_TABLE.at[site.specie.symbol, "atomic_radius"] ** 3 for site in s)
    f["packing_fraction"] = atom_vol / s.volume

    # nearest-neighbour distances (periodic images included)
    nn = []
    for i in range(n):
        neigh = s.get_neighbors(s[i], 5.0)
        nn.append(min(x.nn_distance for x in neigh) if neigh else 5.0)
    nn = np.array(nn)
    f["nn_dist_mean"] = nn.mean()
    f["nn_dist_min"] = nn.min()
    f["nn_dist_max"] = nn.max()
    f["nn_dist_std"] = nn.std()
    return f


def main():
    raw = json.load(gzip.open(DATA))
    structures = [row[0] for row in raw["data"]]
    y = [row[1] for row in raw["data"]]
    print(f"Featurizing {len(structures)} structures...")
    feats = Parallel(n_jobs=-1, batch_size=64, verbose=5)(delayed(featurize)(s) for s in structures)
    df = pd.DataFrame(feats)
    df["log10_kvrh"] = y
    df.to_csv(OUT, index=False)
    print(f"Saved {df.shape} -> {OUT}")


if __name__ == "__main__":
    main()
