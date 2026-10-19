# 01 – Project overview

## 1. Problem

The **bulk modulus (K)** measures how strongly a material resists uniform compression: a higher K means a stiffer material. It matters for structural materials, energy systems, electronics and engineering design.

Bulk modulus is normally obtained from **first-principles (DFT) calculations**, which are accurate but computationally expensive. This project builds a **machine-learning framework** that predicts bulk modulus directly from a material's crystal structure in milliseconds, so large numbers of candidate materials can be screened quickly.

**Goal:** given a crystal structure, predict `log10(K_VRH)`, and compare several ML models to find the most effective one.

## 2. Dataset – Matbench `log_kvrh`

| Property | Value |
|---|---|
| Source | Matbench benchmark suite (Dunn et al., 2020), derived from the Materials Project |
| Samples | 10,987 crystalline materials |
| Input | Crystal structure (pymatgen `Structure`: lattice, atomic species, atomic positions) |
| Target | `log10(K_VRH)`, with K in GPa |
| Target range | K = 1 to 575 GPa (median 86 GPa) |

**What is K_VRH?** For a single crystal, the bulk modulus can be averaged in different ways:
- **Voigt**: assumes uniform strain, which gives an upper bound.
- **Reuss**: assumes uniform stress, which gives a lower bound.
- **Hill**: the average of the Voigt and Reuss values. K_VRH is this Voigt-Reuss-Hill average.

**Why log10?** Bulk modulus spans more than two orders of magnitude. Predicting the logarithm makes errors relative (a 10 % error counts the same for soft and stiff materials) and stops very stiff materials from dominating training.

## 3. From crystal structure to numbers (descriptors)

ML algorithms such as Random Forest cannot read a crystal structure directly, so every structure is converted into **65 numerical descriptors** (`src/featurize.py`).

### 3.1 Composition descriptors (50)
For each of 10 elemental properties, five statistics are computed across the elements in the material, weighted by each element's atomic fraction:

| Elemental property | Why it may matter |
|---|---|
| Atomic number Z, atomic mass | Element identity and size |
| Electronegativity | Ionic vs. covalent bonding character |
| Atomic radius, molar volume | Atom size, which affects how densely atoms pack |
| Melting point | A proxy for bond strength |
| Periodic row, group | Position in the periodic table |
| First ionization energy | How tightly electrons are held |
| Valence electrons | Bonding electrons available |

Statistics: **mean, standard deviation, min, max, range**. This is similar in spirit to the widely used Magpie descriptors.

### 3.2 Structure descriptors (15)
| Descriptor | Meaning |
|---|---|
| `n_elements`, `n_sites` | Number of distinct elements and of atoms in the unit cell |
| `volume_per_atom` | Unit cell volume ÷ number of atoms |
| `density` | Mass density (g/cm³) |
| `a, b, c, alpha, beta, gamma` | Lattice parameters and angles |
| `packing_fraction` | Total volume of atomic spheres ÷ cell volume |
| `nn_dist_mean/min/max/std` | Statistics of each atom's nearest-neighbour distance (periodic images included, 5 Å cutoff) |

## 4. Models

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Linear baseline | Features standardised |
| Random Forest | Tree ensemble (bagging) | 500 trees, each split considers 33 % of features |
| MLP | Neural network | 3 hidden layers (256-128-64), early stopping, features standardised |
| Gradient Boosting | Tree ensemble (boosting) | scikit-learn `HistGradientBoostingRegressor`, 1,500 iterations, learning rate 0.03. Added beyond the original proposal |

## 5. Evaluation protocol

1. **Train/test split:** 80 % train (8,789 materials) / 20 % test (2,198 materials), `random_state=42`. The test set is never used during training.
2. **5-fold cross-validation** on the training set to check that results are stable.
3. **Metrics** (all on the test set):
   - **MAE** – mean absolute error in log10 units (the Matbench standard metric)
   - **RMSE** – root mean squared error, which penalises large errors more
   - **R²** – fraction of variance explained (1.0 = perfect)
   - **MAE in GPa** – after converting predictions back with 10^x, for an intuitive reading

## 6. Two models in the app

The interactive app (`app.py`) uses two Gradient Boosting models:
- **Structure-aware model** (all 65 descriptors) for test materials and uploaded CIF files.
- **Composition-only model** (51 descriptors: the 50 composition statistics plus `n_elements`) for the "type a formula" feature. A formula alone does not describe the atomic arrangement, so this model is somewhat less accurate (R² ≈ 0.85 vs. 0.91).
