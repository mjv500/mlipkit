# MlipKit
<code>**This is a beta version of MlipKit. Documentation and tutorials will be available soon.**</code>

**MlipKit** is a lightweight Python toolkit for managing, training, and evaluating Machine Learning Interatomic Potentials (MLIPs) such as MTP (Moment Tensor Potentials). It provides a unified and extensible interface for interacting with MLIP engines in an ASE-compatible environment.

---

## 🔧 Features

- Simple wrapper classes for different MLIP backends (e.g. MTP)
- Model training, saving, and reloading
- Pre-trained potential loading and validation
- Compute properties (energy, forces) via trained models
- Metadata and hash tracking for reproducibility
- Automatic generation of [ASE](https://wiki.fysik.dtu.dk/ase/) calculators

---

## 🚀 Installation

```bash
pip install git+https://github.com/chewingram/mlipkit.git
```

Or, if cloning locally:

```bash
git clone https://github.com/chewingram/mlipkit.git
cd mlipkit
pip install .
```

---

## 🧩 Requirements

- Python ≥ 3.8
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- NumPy

Install them with:

```bash
pip install ase numpy
```

---

## 📜 License

MIT License © 2025 Samuel Longo

---

## 🙋‍♂️ Author

**Samuel Longo**  
PhD Candidate – University of Liège  
Contact: [longo.samuel@outlook.it](mailto:longo.samuel@outlook.it)
