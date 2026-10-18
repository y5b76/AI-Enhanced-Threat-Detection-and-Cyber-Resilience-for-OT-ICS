# 🛡️ AI-Enhanced Threat Detection and Cyber Resilience for OT/ICS

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN%20%2B%20LSTM--AE-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RF%20%2B%20IF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6F00?style=for-the-badge)](https://github.com/shap/shap)
[![Datasets](https://img.shields.io/badge/Datasets-SWaT%20%7C%20WaDi%20%7C%20BATADAL-005571?style=for-the-badge)](#-datasets)
[![MITRE ATT&CK for ICS](https://img.shields.io/badge/MITRE%20ATT%26CK-for%20ICS-red?style=for-the-badge)](https://attack.mitre.org/matrices/ics/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](#)

---

## 📌 Project Overview

This repository accompanies my MSc Cyber Security dissertation, **"AI-Enhanced Network Traffic and Process Analytics for Threat Detection and Cyber Resilience Metric Optimisation (OT/ICS Focus)"**.

It evaluates four anomaly detection models across three industrial control system datasets, and asks a question that accuracy metrics alone cannot answer: *which model would you actually want running in a water treatment plant?* The work addresses six research gaps identified in OT/ICS anomaly detection and follows a Design Science Research methodology (Peffers et al., 2007; vom Brocke, Hevner and Maedche, 2020).

Three contributions sit on top of the model comparison:

- ✅ **Resilience-Accuracy Score (RAS)** - a composite metric that ranks models on detection delay and recovery behaviour as well as F1, with weights derived through AHP-Entropy
- ✅ **Quantitative SHAP evaluation** - explanations scored for faithfulness and stability, not just plotted
- ✅ **MITRE ATT&CK for ICS / STIX pipeline** - detections mapped into structured cyber threat intelligence indicators
- ✅ **Four models, three datasets** - Random Forest, CNN, Isolation Forest and LSTM Autoencoder on SWaT, WaDi and BATADAL
- ✅ **Live proof-of-concept testbed** - OpenPLC v3 in Docker with a simulated tank and pump process and an attacker container

> 📄 **[Dissertation PDF →](#)** *(add link once the final version is cleared for release)*

---

## 🧱 Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    THREE ICS DATASETS                                │
│   SWaT (network + process)  |  WaDi (process)  |  BATADAL (process)  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                          │
│  SWaT only: CIP / EtherNet-IP network traffic features           │
│  All datasets: physical / process-layer features                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │
        ┌───────────────┬─────┴─────────┬───────────────┐
        ▼               ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Random Forest│ │     CNN      │ │  Isolation   │ │ LSTM         │
│ (supervised) │ │ (supervised) │ │  Forest      │ │ Autoencoder  │
│              │ │              │ │(unsupervised)│ │(unsupervised)│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       └────────────────┴────────┬───────┴────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                RESILIENCE METRICS  (MTTD, MTTR, miss rate)       │
│   Resilience-Accuracy Score: AHP-Entropy weights, computed       │
│   independently per dataset                                      │
└───────────────────────┬──────────────────────────────────────────┘
                        │
              ┌─────────┴───────────┐
              ▼                     ▼
┌─────────────────────────┐ ┌─────────────────────────────────────┐
│   SHAP EVALUATION       │ │  MITRE ATT&CK for ICS / STIX CTI    │
│   Faithfulness +        │ │  Detections mapped to structured    │
│   stability per model   │ │  threat intelligence indicators     │
└─────────────────────────┘ └─────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
ot-ics-threat-detection/
│
├── README.md
├── Dissertation.pdf                  ← Full dissertation (add when released)
│
├── **[scripts](#)**
│   ├── LOAD_NETWORK.py               ← SWaT network traffic loading
│   ├── compute_resilience_metrics.py ← MTTD, MTTR, miss rate
│   ├── compute_ras.py                ← Resilience-Accuracy Score
│   ├── export_for_shap.py            ← Model and data export for XAI
│   ├── xai_evaluation_colab.py       ← SHAP faithfulness and stability (Colab GPU)
│   ├── extract_shap_attributions.py  ← Attribution extraction
│   └── mitre_attack_stix_mapping.py  ← ATT&CK for ICS to STIX pipeline
│
├── testbed/                          ← OpenPLC v3 demo (Docker)
│
└── screenshots/                      ← Result figures and demo captures
```

---

## ⚙️ Tech Stack

| Component                     | Role                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| **Python 3.12**               | TensorFlow-dependent scripts (`.venv312`)                         |
| **Python 3.14**               | Everything else (`.venv`)                                         |
| **scikit-learn**              | Random Forest, Isolation Forest                                   |
| **TensorFlow / Keras**        | CNN and LSTM Autoencoder                                          |
| **SHAP**                      | Explanations, with the explainer matched to each architecture     |
| **STIX 2**                    | Threat intelligence indicator generation                          |
| **Google Colab (T4 GPU)**     | All SHAP and XAI computation                                      |
| **OpenPLC v3 + Docker**       | Live testbed (Windows 11, Docker Desktop, WSL2)                   |

> **Why two Python environments?** Python 3.14 has no compatible TensorFlow wheels, so anything touching TensorFlow runs in a separate 3.12 environment.

---

## 📊 Datasets

| Dataset     | Source                         | Collected | Role in this work                                  |
| ----------- | ------------------------------ | --------- | -------------------------------------------------- |
| **SWaT**    | iTrust, SUTD                   | Dec 2015  | Full pipeline, including network traffic analytics |
| **WaDi**    | iTrust, SUTD                   | Oct 2017  | Process-layer generalisation benchmark             |
| **BATADAL** | Taormina et al. (2018)         | n/a       | Process-layer generalisation benchmark             |

**Scope note:** the network traffic analytics pipeline (CIP / EtherNet-IP feature engineering) applies to **SWaT only**. WaDi and BATADAL are used to test whether the process-layer findings generalise. The data itself is not included here. SWaT and WaDi are available from iTrust on request.

---

## 🔍 Part 1 - Model Results

F1 scores per model and dataset. Every number below comes from real pipeline output.

| Model             | BATADAL | WaDi  | SWaT  |
| ----------------- | ------- | ----- | ----- |
| Random Forest     | 0.200   | 0.075 | 0.105 |
| Isolation Forest  | 0.067   | 0.033 | 0.326 |
| CNN               | 0.400   | 0.000 | 0.112 |
| LSTM Autoencoder  | 0.222   | 0.335 | 0.294 |

**What stands out**

- No single model wins everywhere. CNN is best on BATADAL and scores zero on WaDi. LSTM-AE is best on WaDi, and Isolation Forest is best on SWaT.
- The F1 values are low in absolute terms. ICS attacks are rare, long-lived and often subtle, and the dissertation reports that honestly rather than tuning toward a headline number.
- The SWaT CNN shows probability saturation, which the SHAP stability analysis independently corroborates.

---

## ⏱️ Part 2 - Resilience-Accuracy Score (RAS)

F1 says how often a detector is right. It says nothing about how long an attack runs before it is caught, or how long false alarms keep firing afterwards. RAS adds both.

- **Weighting:** AHP-Entropy, with priority ordering **MTTD > F1 > MTTR**
- **Miss rate:** applied as a multiplicative penalty
- **Scope:** computed independently for each dataset
- **MTTR definition:** elapsed time from the end of an attack event to the last trailing false-positive alert. This is deliberately **not** the conventional "time to respond", and it is used consistently throughout.

**Why it matters:** on BATADAL, the LSTM Autoencoder is second by F1 but drops to **last by RAS**, because its mean time to detect is 100,800 seconds. A leaderboard built on F1 alone would have hidden that.

---

## 🔎 Part 3 - Explainability (SHAP)

Explanations were evaluated quantitatively using the framework from Nauta et al. (2023) and Hedström et al. (2023), with the SHAP explainer chosen to suit each architecture.

| Finding | Detail |
| --- | --- |
| **Isolation Forest is negatively faithful** | On all three datasets. This is a structural mismatch between the model and SHAP, reported as a finding rather than a bug. |
| **LSTM Autoencoder is the most stable** | Explanations stay consistent across samples. |
| **SWaT CNN is the least stable** | Matches its probability saturation behaviour. |

---

## 🧬 Part 4 - MITRE ATT&CK for ICS and STIX

`mitre_attack_stix_mapping.py` maps detections to ATT&CK for ICS techniques and exports STIX indicators. Run across all 12 model and dataset combinations, it produced **2,084 STIX indicators**:

| Dataset | Indicators |
| ------- | ---------- |
| BATADAL | 130        |
| WaDi    | 202        |
| SWaT    | 1,752      |

---

## 🧪 Live Demonstration Testbed

A small Docker-based environment shows the detection idea end to end:

- **OpenPLC v3 Runtime** running in Docker
- **Python tank and pump stub** simulating the physical process
- **Attacker container** that manipulates the process

> ⚠️ **This is an illustrative proof of concept.** It uses a separately fitted small model and is not part of the reported dissertation results above.

| Fig 1 - Testbed topology | Fig 2 - Attack and detection |
| --- | --- |
| *(add screenshot)* | *(add screenshot)* |

---

## ⚠️ Limitations

- Absolute F1 scores are low on all three datasets, and no model dominates.
- The network traffic analytics contribution is limited to SWaT.
- Isolation Forest explanations are not reliable under SHAP.
- The live demo uses a simulated process, not real plant hardware.
- Results depend on the specific attack scenarios in each public dataset.

---

## 🧠 Frameworks & References

| Framework / Reference                              | Application                                      |
| -------------------------------------------------- | ------------------------------------------------ |
| **Peffers et al. (2007)**                          | Design Science Research methodology              |
| **vom Brocke, Hevner and Maedche (2020)**          | Design Science Research guidelines               |
| **Nauta et al. (2023)**                            | XAI evaluation framework                         |
| **Hedström et al. (2023)**                         | Quantitative explanation evaluation              |
| **Taormina et al. (2018)**                         | BATADAL dataset and challenge                    |
| **iTrust, SUTD**                                   | SWaT and WaDi testbeds                           |
| **MITRE ATT&CK for ICS**                           | Technique mapping                                |
| **STIX / TAXII**                                   | Threat intelligence exchange format              |

---

## 👤 Author

**Toochukwu Praise Ajoku**
MSc Cyber Security, Keele University (2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/toochukwu-praise-ajoku/)

---

*This project uses publicly available research datasets for educational and portfolio purposes. The testbed runs entirely in local containers. No real industrial infrastructure was targeted.*
