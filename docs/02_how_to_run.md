# 02 – How to run

## 1. Requirements

- **Python 3.10 or newer** (developed and tested with Python 3.14)
- About 200 MB of disk space for packages
- No GPU needed. Everything runs on a normal laptop CPU.

## 2. Get the code

```bash
git clone https://github.com/Prakash686/ML-based-Bulk-Modulus-Prediction.git
cd ML-based-Bulk-Modulus-Prediction
```

## 3. Install dependencies

Optional but recommended: create a virtual environment first.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pymatgen` (reads crystal structures and provides element data), `joblib`, `tabulate`, `streamlit`.

## 4. Run the pipeline

The dataset (`data/log_kvrh.json.gz`) is already included in the repository. You do not need to download anything.

### Step 1 – Generate descriptors

```bash
python src/featurize.py
```

- Reads all 10,987 crystal structures
- Computes 65 descriptors per material, in parallel on all CPU cores
- Writes `data/features.csv`
- Takes about 20 seconds

> `data/features.csv` is already committed, so you can skip this step if you only want to train or run the app.

### Step 2 – Train and evaluate the models

```bash
python src/train.py
```

- Splits the data 80/20, runs 5-fold cross-validation, and trains Linear Regression, Random Forest, MLP and Gradient Boosting
- Prints a metrics table to the console
- Writes everything to `results/`:

| File | Content |
|---|---|
| `metrics.csv`, `metrics.md` | MAE / RMSE / R² for every model |
| `parity_plots.png` | Predicted vs. true values for each model |
| `model_comparison.png` | Bar chart of test MAE |
| `feature_importance.png`, `.csv` | Most important descriptors (Random Forest) |
| `target_distribution.png` | Histogram of log10(K_VRH) |

- Takes a few minutes, mostly for cross-validation of the Random Forest and MLP

## 5. Launch the interactive app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. The first load trains two models, which takes about 30 seconds. After that, results are cached.

| Tab | How to use it |
|---|---|
| **Predict from formula** | Type a chemical formula (`Si`, `NaCl`, `Fe2O3`, `WC`, `MgAl2O4`...) to get the predicted bulk modulus in GPa, a stiffness label and where the material sits among all known materials. If the formula exists in the dataset, the DFT values are shown for comparison. |
| **Browse test materials** | Pick any of the 2,198 held-out materials to compare predicted and actual values. |
| **Upload your own CIF** | Upload a `.cif` crystal structure file, for example from [Materials Project](https://next-gen.materialsproject.org/), to get a prediction from the structure-aware model. |
| **Model performance** | Metrics table and all result plots. |

Stop the app with `Ctrl + C` in the terminal.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pymatgen` | Run `pip install -r requirements.txt` inside the same environment you run Python from |
| `pymatgen` fails to install | Upgrade pip first (`python -m pip install --upgrade pip`), or use Python 3.11/3.12 |
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `FileNotFoundError: data/features.csv` | Run `python src/featurize.py` first |
| "Could not parse that formula" | Use correct element capitalisation, e.g. `NaCl` not `nacl` |
| Many `UserWarning`s from pymatgen | Harmless; they are suppressed in the scripts |
