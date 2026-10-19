# Machine Learning-Based Prediction of Bulk Modulus of Materials

Predicting the Voigt-Reuss-Hill bulk modulus, `log10(K_VRH / GPa)`, of crystalline materials directly from their crystal structure, using the **Matbench `log_kvrh`** benchmark dataset (10,987 materials).

## Pipeline

1. **Load** – Matbench `log_kvrh` JSON (pymatgen `Structure` + target).
2. **Featurize** (`src/featurize.py`) – 65 descriptors per material:
   - **Composition (50):** fraction-weighted mean, std, min, max and range of 10 elemental properties (Z, atomic mass, electronegativity, atomic radius, melting point, row, group, molar volume, first ionization energy, valence electrons).
   - **Structure (15):** number of elements and sites, volume per atom, density, lattice parameters a/b/c and α/β/γ, atomic packing fraction, nearest-neighbour distance statistics.
3. **Train and compare** (`src/train.py`): Linear Regression, Random Forest, MLP, plus Gradient Boosting as an additional model.
4. **Evaluate:** 80/20 hold-out test split (seed 42), plus 5-fold cross-validation on the training set. Metrics are MAE, RMSE and R² (in log10 units), and MAE in GPa.

## Results (hold-out test set, 2,198 materials)

| Model | CV MAE (log10) | Test MAE (log10) | Test RMSE (log10) | Test R² | Test MAE (GPa) |
|---|---|---|---|---|---|
| Linear Regression | 0.1085 ± 0.0016 | 0.1090 | 0.1665 | 0.807 | 19.59 |
| Random Forest | 0.0713 ± 0.0015 | 0.0696 | 0.1220 | 0.896 | 11.89 |
| MLP | 0.0910 ± 0.0036 | 0.0991 | 0.1493 | 0.845 | 18.33 |
| **Gradient Boosting** | **0.0674 ± 0.0010** | **0.0668** | **0.1162** | **0.906** | **11.21** |

**Findings**
- The tree ensembles (Gradient Boosting, Random Forest) clearly outperform Linear Regression and the MLP. The relationship between the descriptors and bulk modulus is strongly non-linear.
- The most important descriptors are **volume per atom**, **mean melting point**, **mean molar volume**, **density** and **packing fraction**. Physically, more tightly packed, strongly bonded (high-melting) materials resist compression more.
- The largest errors occur for very soft materials (`log10 K < 1`), which are rare in the dataset (see the target distribution plot).

| Parity plots | Model comparison |
|---|---|
| ![](results/parity_plots.png) | ![](results/model_comparison.png) |

| Feature importance | Target distribution |
|---|---|
| ![](results/feature_importance.png) | ![](results/target_distribution.png) |

## How to run

```bash
pip install -r requirements.txt
python src/featurize.py   # ~20 s, writes data/features.csv
python src/train.py       # ~few min, writes results/
```

## Repository layout

```
data/log_kvrh.json.gz   Matbench log_kvrh dataset
data/features.csv       generated descriptors + target
src/featurize.py        structure -> descriptor extraction
src/train.py            model training, evaluation, plots
results/                metrics (csv/md) and figures
```

## Dataset

Dunn et al., *Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm*, npj Computational Materials 6, 138 (2020).
