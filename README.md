# Real-Time Proactive Anomaly Detection via Forward and Backward Forecast Modeling

Real-time anomaly detection is critical for preventing failures in high-stakes domains like industrial automation, financial systems, and satellite telemetry. Traditional anomaly detectors are reactive, they detect anomalies only after they’ve occurred. However, in many real-world applications, what’s needed is a proactive solution: one that can anticipate anomalies before they manifest.

This repository introduces two novel, real-time-capable frameworks for proactive anomaly detection:
- **FFM (Forward Forecasting Model):** predicts future values using historical context
- **BRM (Backward Reconstruction Model):** reconstructs past behavior using future context

## Quick Start

### Requirements

This project requires **Python 3.8+** and **pip**.

Follow the steps below to set up and run the code:

---

### 1. Clone the repository

```
git clone https://github.com/anonymous1643/Proactive-Anomaly-Forecasters.git
cd Proactive-Anomaly-Forecasters
```

---

### 2. Create and activate a virtual environment

<details>
<summary>Windows (Command Prompt)</summary>

```
python -m venv venv
venv\Scripts\activate
```
</details>

<details>
<summary>macOS / Linux</summary>

```
python3 -m venv venv
source venv/bin/activate
```
</details>

---

### 3. Install dependencies

<details>
<summary>If using CPU only (Recommend GPU)</summary>

```
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn pyarrow pyod
```
</details>

<details>
<summary>If using GPU (with CUDA)</summary>

Check your CUDA version using:

```
nvcc --version
```

Then install the appropriate version. Example for CUDA 11.8:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn pyarrow
```
</details>
- Python 3.8+
    
### To run on MSL dataset:

```bash
bash demo.sh 
```

This repository only tests on the MSL dataset due to file size constraints. All other datasets are public on Kaggle.

## Datasets 

This repository uses four multivariate time series datasets, each curated for benchmarking anomaly detection models. The datasets span real-world spacecraft telemetry and server infrastructure data.

---

### SMAP (Soil Moisture Active Passive Satellite)

The SMAP dataset consists of telemetry data from NASA’s SMAP satellite. It includes labeled point and contextual anomalies, annotated using NASA's Incident Surprise Anomaly (ISA) reports.

#### Format
- `.csv` and `.npy` files: `train`, `test`, `labeled_anomalies.csv`

#### Source 
- Original Authors: *Kyle Hundman et al., NASA JPL*
- GitHub: [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)
- Kaggle: [SMAP Dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

---

### MSL (Mars Science Laboratory – Curiosity Rover)

The MSL dataset includes telemetry from NASA’s Mars rover. Anomalies are hand-labeled using domain knowledge from NASA engineers and documents.

#### Format
- `.csv` and `.npy` files: `train`, `test`, `labeled_anomalies.csv`

#### Source 
- Original Authors: *Kyle Hundman et al., NASA JPL*
- GitHub: [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)
- Kaggle: [MSL Dataset on Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

---

### SMD (Server Machine Dataset – OmniAnomaly Version)

The SMD dataset consists of server infrastructure metrics collected from 28 machines over a 5-week period. It includes train/test splits and anomaly labels for each machine entity.

#### Format
- `.txt` files: `train`, `test`, `test_label`, `interpretation_label`

#### Source 
- Original Authors: *NetMan AIOps Team – OmniAnomaly project*
- GitHub: [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
- Kaggle: [SMD Dataset on Kaggle](https://www.kaggle.com/datasets/mgusat/smd-onmiad)

---

### PSM (Pooled Server Metrics – from eBay Inc.)

The PSM dataset was publicly released by eBay Inc. as part of the RANSynCoders project. It contains real server telemetry data for asynchronous anomaly detection.

#### Format
- `.csv` files: `train`, `test`, `test_label`

#### Source 
- Authors: *Abdulaal et al., eBay Inc.*
- GitHub: [https://github.com/eBay/RANSynCoders](https://github.com/eBay/RANSynCoders)

---

## Environment

This implementation has been tested on:

- **GPU:** NVIDIA T4 GPU
- **RAM:** 16 GB  
- **Operating System:** Windows 11 (64-bit)

