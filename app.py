"""Streamlit UI for bulk modulus prediction.

Run:  streamlit run app.py
"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymatgen.core import Composition, Structure
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from featurize import composition_features, featurize  # noqa: E402

SEED = 42
st.set_page_config(page_title="Bulk Modulus Predictor", layout="wide")


def make_gb():
    return HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.03, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=SEED)


@st.cache_resource(show_spinner="Training models (one-time, ~30 s)...")
def load():
    df = pd.read_csv(ROOT / "data" / "features.csv")
    raw = json.load(gzip.open(ROOT / "data" / "log_kvrh.json.gz"))
    structs = [row[0] for row in raw["data"]]
    formulas = [Structure.from_dict(d).composition.reduced_formula for d in structs]
    X, y = df.drop(columns="log10_kvrh"), df["log10_kvrh"]
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)

    # Full model: composition + structure descriptors
    model = make_gb().fit(X.iloc[idx_tr], y.iloc[idx_tr])
    # Formula-only model: composition descriptors only
    comp_cols = list(composition_features(Composition("NaCl")).keys())
    comp_model = make_gb().fit(X.iloc[idx_tr][comp_cols], y.iloc[idx_tr])
    p = comp_model.predict(X.iloc[idx_te][comp_cols])
    comp_scores = (r2_score(y.iloc[idx_te], p), mean_absolute_error(y.iloc[idx_te], p))

    test = pd.DataFrame({
        "idx": idx_te,
        "formula": [formulas[i] for i in idx_te],
        "true": y.iloc[idx_te].to_numpy(),
        "pred": model.predict(X.iloc[idx_te]),
    })
    lookup = pd.DataFrame({"formula": formulas, "k_gpa": 10 ** y.to_numpy(),
                           "in_train": np.isin(np.arange(len(df)), idx_tr)})
    return model, X.columns, structs, test, comp_model, comp_cols, comp_scores, lookup


from featurize import ELEM_TABLE  # noqa: E402
comp_model_elements = set(ELEM_TABLE.index)


def predict(model, cols, s: Structure):
    f = pd.DataFrame([featurize(s.as_dict())])[cols]
    return float(model.predict(f)[0]), f


def structure_card(s: Structure):
    lat = s.lattice
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Formula", s.composition.reduced_formula)
    c2.metric("Atoms in cell", len(s))
    c3.metric("Density", f"{s.density:.2f} g/cm³")
    c4.metric("Volume / atom", f"{s.volume / len(s):.2f} Å³")
    st.caption(f"Lattice a, b, c = {lat.a:.3f}, {lat.b:.3f}, {lat.c:.3f} Å  ·  "
               f"α, β, γ = {lat.alpha:.1f}°, {lat.beta:.1f}°, {lat.gamma:.1f}°")


def stiffness_label(k):
    if k < 20:
        return "Very soft"
    if k < 80:
        return "Soft"
    if k < 180:
        return "Moderately stiff"
    return "Very stiff"


model, cols, structs, test, comp_model, comp_cols, comp_scores, lookup = load()

st.title("Bulk Modulus Predictor")
st.write("Predicts the Voigt-Reuss-Hill bulk modulus **K_VRH** of a crystalline material from its "
         "crystal structure, using a Gradient Boosting model trained on Matbench `log_kvrh` "
         "(test R² ≈ 0.91).")

tab0, tab1, tab2, tab3 = st.tabs(["Predict from formula", "Browse test materials",
                                  "Upload your own CIF", "Model performance"])

with tab0:
    st.write("Type a chemical formula, e.g. `Si`, `NaCl`, `Fe2O3`, `TiC`, `MgAl2O4`. "
             "Only the composition is used here (no crystal structure), so this model is a bit less "
             f"accurate (test R² = {comp_scores[0]:.2f}, MAE = {comp_scores[1]:.3f} log10 units).")
    text = st.text_input("Chemical formula", value="Fe2O3")
    if text.strip():
        try:
            comp = Composition(text.strip())
            if not comp.elements or any(str(el) not in comp_model_elements for el in comp.elements):
                raise ValueError("unknown element")
        except Exception:
            st.error("Could not parse that formula. Use element symbols like Fe2O3 or NaCl.")
        else:
            feats = pd.DataFrame([composition_features(comp)])[comp_cols]
            logk = float(comp_model.predict(feats)[0])
            k = 10 ** logk
            formula = comp.reduced_formula
            c1, c2 = st.columns(2)
            c1.metric(f"Predicted K_VRH of {formula}", f"{k:.1f} GPa", help=f"log10 = {logk:.3f}")
            c2.metric("Stiffness", stiffness_label(k))
            known = lookup[lookup.formula == formula]
            if len(known):
                st.success(f"{formula} is in the dataset ({len(known)} polymorph(s)). "
                           "DFT values for comparison:")
                st.dataframe(known.assign(**{"K_VRH (GPa)": known.k_gpa.round(1),
                                             "Seen in training": known.in_train})
                             [["formula", "K_VRH (GPa)", "Seen in training"]], hide_index=True)
            else:
                st.caption("This formula is not in the dataset, so this is a genuine prediction "
                           "for a new material.")

            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.hist(lookup.k_gpa, bins=np.logspace(0, 2.8, 60), color="#B0B7C3")
            ax.axvline(k, color="red", lw=2, label=f"{formula}: {k:.0f} GPa")
            ax.set(xscale="log", xlabel="Bulk modulus (GPa)", ylabel="Materials",
                   title="Where it sits among 10,987 known materials")
            ax.legend()
            st.pyplot(fig, use_container_width=False)

with tab1:
    st.write("These 2,198 materials were **never seen during training**. Pick one to compare the "
             "prediction with the DFT-computed value.")
    labels = [f"{r.formula}  (#{r.idx})" for r in test.itertuples()]
    choice = st.selectbox("Material", range(len(test)), format_func=lambda i: labels[i])
    row = test.iloc[choice]
    s = Structure.from_dict(structs[row.idx])
    structure_card(s)

    k_true, k_pred = 10 ** row.true, 10 ** row.pred
    err = abs(k_pred - k_true) / k_true * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted K_VRH", f"{k_pred:.1f} GPa")
    c2.metric("Actual K_VRH (DFT)", f"{k_true:.1f} GPa")
    c3.metric("Relative error", f"{err:.1f} %")
    st.info(stiffness_label(k_pred))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(test.true, test.pred, s=4, alpha=0.25, color="grey", label="All test materials")
    ax.scatter([row.true], [row.pred], s=120, color="red", edgecolor="black", zorder=3,
               label=row.formula)
    ax.plot([0, 2.8], [0, 2.8], "k--", lw=1)
    ax.set(xlabel="Actual log10(K_VRH / GPa)", ylabel="Predicted", xlim=(0, 2.8), ylim=(0, 2.8))
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=False)

with tab2:
    st.write("Upload a crystal structure in **CIF** format (e.g. downloaded from the "
             "Materials Project) to get a bulk modulus prediction.")
    up = st.file_uploader("CIF file", type=["cif"])
    if up is not None:
        try:
            s = Structure.from_str(up.getvalue().decode("utf-8", errors="ignore"), fmt="cif")
        except Exception as e:
            st.error(f"Could not read this CIF file: {e}")
        else:
            structure_card(s)
            logk, feats = predict(model, cols, s)
            st.metric("Predicted K_VRH", f"{10 ** logk:.1f} GPa", help=f"log10 = {logk:.3f}")
            st.info(stiffness_label(10 ** logk))
            with st.expander("Computed descriptors"):
                st.dataframe(feats.T.rename(columns={0: "value"}))

with tab3:
    st.subheader("Model comparison (hold-out test set)")
    st.dataframe(pd.read_csv(ROOT / "results" / "metrics.csv"), hide_index=True)
    st.image(str(ROOT / "results" / "parity_plots.png"))
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "results" / "feature_importance.png"))
    c2.image(str(ROOT / "results" / "target_distribution.png"))
