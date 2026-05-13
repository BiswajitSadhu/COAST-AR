

<img width="1536" height="1024" alt="COAST_AR_architecture (1)" src="https://github.com/user-attachments/assets/e27b684c-f6c7-4781-bedb-6e52bafac8ef" /># 🌫️ COAST-AR  
**Coagulation-Aware Sequential Transformer for Aerosol Moment Dynamics**

COAST-AR is a physics-constrained causal autoregressiv[coast-ar (1).pdf](https://github.com/user-attachments/files/27705401/coast-ar.1.pdf)
e Transformer framework for predicting the temporal evolution of aerosol populations under coagulation-driven dynamics.

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




<img width="1915" height="955" alt="Screenshot from 2026-04-16 12-56-22" src="https://github.com/user-attachments/assets/4f28e897-33dd-4ba5-9a0c-5ec9f7bd3d5b" />

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

