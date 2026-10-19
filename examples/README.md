# Example CIF files

Ready-to-upload crystal structures for the **"Upload your own CIF"** tab of the app. All of them come from the **test set**, so the model never saw them during training.

| File | Material | Atoms in cell | DFT K_VRH (GPa) | App prediction (GPa) |
|---|---|---|---|---|
| `WC.cif` | Tungsten carbide | 2 | 385 | 325 |
| `SiC.cif` | Silicon carbide | 18 | 213 | 215 |
| `Al2O3.cif` | Alumina | 20 | 189 | 226 |
| `Fe2O3.cif` | Iron(III) oxide | 40 | 178 | 179 |
| `MgO.cif` | Magnesium oxide | 2 | 151 | 158 |
| `Cu.cif` | Copper | 1 | 145 | 161 |
| `C.cif` | Carbon (a non-diamond polymorph) | 4 | 122 | 226 |
| `Si.cif` | Silicon | 4 | 89 | 102 |

Most predictions are within about 15 %. `C.cif` is a deliberately hard case: carbon polymorphs range from soft graphite-like phases to diamond (about 440 GPa), and the model lands in between.

**More structures:** download any material as CIF from [Materials Project](https://next-gen.materialsproject.org/) (search a formula, open the material, then *Export as → CIF*).
