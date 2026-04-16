# 🌫️ COAST-AR  
**Coagulation-Aware Sequential Transformer for Aerosol Moment Dynamics**

COAST-AR is a physics-informed autoregressive Transformer framework for predicting the temporal evolution of aerosol populations under coagulation-driven dynamics.

---

## 📌 Overview

COAST-AR provides a data-driven surrogate model that:

- Learns autoregressive temporal evolution of aerosol moments  
- Captures coagulation-driven nonlinear dynamics  
- Enables fast multi-step prediction  
- Reproduces emergent attractor behavior in geometric standard deviation  

---

## 🔗 Dataset (Zenodo)

The dataset used in this work is publicly available on Zenodo:

👉 **[ADD ZENODO LINK HERE]**

### 📥 Steps to Use the Dataset

1. Download the dataset from the Zenodo link above  
2. Extract the contents  
3. Place the dataset inside the repository as:

```
COAST-AR/DATASET/
```

4. Ensure the following structure:

```
DATASET/
└── GROUPKFOLD_STRATIFIED/
    ├── saved_splits/
    └── shared_in_zenodo/
```

---

## ⚙️ Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate coast-ar
```

---

## 🚀 Training

```bash
python train.py --config config.yaml
```

---

## 🧪 Inference

```python
from model_moment import ModelClass
import torch

model = ModelClass(...)
model.load_state_dict(torch.load("path_to_checkpoint.pt"))
model.eval()
```

---

## 💻 Streamlit App

```bash
streamlit run app_streamlit.py
```

---

## 📂 Repository Structure

```
COAST-AR/
├── train.py
├── model_moment.py
├── preprocess_moment.py
├── utils.py
├── CollisionFrequency.py
├── fsp_c.py
├── config.yaml
├── environment.yml
├── app_streamlit.py
├── DATASET/
├── cpt_with_split_data/
└── assets/
```

---

## 📄 Citation

```
@article{coast_ar,
  title={COAST-AR: An Autoregressive Transformer Framework for Sequential Prediction of Aerosol Moment Dynamics},
  author={...},
  year={2026}
}
```

---

## 📬 Notes

- Focused on coagulation-only aerosol dynamics  
- Extendable to broader aerosol microphysics  

---

