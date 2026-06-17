<p align="center">
  <img src="docs/logo.svg" alt="MlipKit logo" width="400"/>
</p>

# MlipKit

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

## 🧪 Example Usage

```python
from ase.io import read
from mlipkit.models import MTP_model

# Load training dataset
dataset = read('Training_set.traj', index=':')

# Define the training hyperparameters for MTP (only the mandatory ones are shown here)
# Please, see MlipKit and MTP documentation (https://gitlab.com/ashapeev/mlip-2) for more details.
hyperparameters = dict(mlip_bin='path/to/the/mlp/binary',
                       untrained_pot_file_dir='path/to/the/directory/containing/the/empty/mtp/potentials',
                       mtp_level=12,
                       min_dist=2,
                       max_dist=8,
                       radial_basis_type='RBChebychev',
                       radial_basis_size='8')

# Define model
model = MTP_model(
    root_dir='custom/directory/model_dir',
    name='example_model',
    train_set=dataset,
    hyperparameters=hyperparameters,
    pre_trained=False
)

# Train model
model.train_model()

# If the training was successful, then it was also saved as `example_model.json` inside root_dir. 

# Load model later
from mlipkit import MlipModel
model = MlipModel.load_model('model_dir/example_model.json')

# Use model to compute properties
atoms = dataset[0]
result = model.compute_properties(atoms=atoms, wdir='custom/directory/tmp_compute')
```

---

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
