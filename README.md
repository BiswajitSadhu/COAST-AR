

# 🌫️ COAST-AR  
**Coagulation-Aware Sequential Transformer for Aerosol Moment Dynamics**

COAST-AR is a physics-constrained causal autoregressive Transformer framework for predicting the temporal evolution of aerosol populations under coagulation-driven dynamics.

<img width="1280" height="720" alt="coast-ar" src="https://github.com/user-attachments/assets/2aa4afd1-819e-4527-b4e4-8211a69d2e65" />

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

👉 **10.5281/zenodo.18656856**

### 📥 Steps to Use the Dataset

1. Download the dataset from the Zenodo link above  
2. Extract the contents  
3. Place the dataset inside the repository inside:

```
COAST-AR/DATASET/GROUPKFOLD_STRATIFIED/
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
or
pip install -r requirements.txt
conda activate coast-ar
```

---

## 🚀 Training


```bash
python train.py --config config.yaml
```

---


## 💻 Streamlit App

```bash
streamlit run app_streamlit.py
```
Presently deployed online at Streamlit Community Portal:

**link: https://coast-ar-zpykrjjgdpa2dyeftjqux2.streamlit.app/**


<img width="953" height="1075" alt="coast-ar-streamlit (1)" src="https://github.com/user-attachments/assets/1b17787f-dc20-4512-bbad-2d4d8914c9ad" />

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
├── requirements.txt
├── app_streamlit.py
├── DATASET/
├── cpt_with_split_data/
└── assets/
```

---

## 📄 Citation (TODO)

```
@article{coast_ar,
  title={COAST-AR: A Physics-Constrained Autoregressive Transformer for Coagulation-Driven Aerosol Evolution},
  author={...},
  year={2026}
}
```

---

## 📬 Notes

- Focused on coagulation-only aerosol dynamics  
- Extendable to broader aerosol microphysics  

---

