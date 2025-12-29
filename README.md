# ECG Arrhythmia Classification with Explainability Analysis

Multi-class ECG arrhythmia classification using Support Vector Machines and Random Forest with Permutation Feature Importance analysis for clinical interpretability.

**Course Project:** COMPSCI5100 - Machine Learning & Artificial Intelligence for Data Scientists, University of Glasgow

---

## 🎯 Project Overview

This project develops explainable machine learning models for automated ECG arrhythmia classification that achieve near-perfect accuracy while identifying the physiological features driving predictions. The study addresses the critical need for transparent AI in medical diagnosis, where clinicians require interpretable models to validate and trust automated systems.

**Key Achievements:**
- **99.87% accuracy** on patient-level holdout validation (Random Forest)
- **Clinically validated features:** QRS complex identified as most important (64.5-75.8% of total importance)
- **Patient-level generalization:** Models tested on completely unseen patients (no data leakage)
- **Full interpretability:** Permutation Feature Importance reveals which temporal segments drive predictions

---

## 📊 Dataset

**MIT-BIH Arrhythmia Database** (Moody & Mark, 2001)
- **Total samples:** 149,768 heartbeats from 49 patients
- **Classes (8):** Normal (N), Left Bundle Branch Block (L), Right Bundle Branch Block (R), Premature Ventricular Contractions (V), Atrial Premature (A), Fusion Ventricular-Normal (F), Fusion Paced-Normal (f), Paced (/)
- **Features:** 275 temporal features per heartbeat, organized into 11 physiological segments:
  - Segments 1-4: PR interval
  - Segments 5-7: QRS complex
  - Segments 8-11: ST segment
- **Sampling rate:** 360 Hz across two channels (MLII and V5)
- **Class imbalance:** 328:1 ratio (Normal: 50.1%, rarest class: 0.15%)

---

## 🔬 Methodology

### Pipeline Architecture

```
ECG Signals → R-peak Detection → Heartbeat Segmentation → Normalization → 
Feature Extraction → Train/Test Split → Resampling → Classification → 
Explainability Analysis → Clustering Analysis
```

### Validation Protocols

**1. Beat Holdout Validation**
- Random 75-25 split of all heartbeats
- Training: 112,326 beats (balanced to 30,992)
- Testing: 37,442 beats (imbalanced)
- Purpose: Assess algorithmic capability

**2. Patient Holdout Validation** ⭐
- Complete patient separation: 43 training patients, 5 test patients
- Training: 200,352 beats (balanced to 25,044 per class)
- Testing: 14,482 beats (imbalanced)
- **Eliminates data leakage** - realistic clinical generalization

### Models

**Support Vector Machine (SVM)**
- Kernel: RBF (Radial Basis Function)
- Regularization: C=1.0
- Gamma: 'scale'
- Chosen for: High-dimensional optimization, optimal decision boundaries

**Random Forest**
- Trees: 100
- Default parameters (unlimited depth, min 2 samples per split)
- Chosen for: Ensemble robustness, handles class imbalance, feature interactions

**K-Means Clustering**
- K=8 (matching 8 arrhythmia classes)
- Distance metric: Euclidean
- Purpose: Assess natural class separation and dimensionality reduction viability

### Explainability Method

**Permutation Feature Importance (PFI)**
- Applied to all 4 trained models (SVM/RF × Beat/Patient holdout)
- 5-fold stratified cross-validation on training data
- Importance metric: Drop in accuracy when segment features are randomly shuffled
- **Segment-level analysis** (not individual features) for clinical interpretability
- Reveals which cardiac cycle phases (PR interval, QRS complex, ST segment) drive predictions

---

## 📈 Results

### Classification Performance

| Model | Validation | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|----------|-----------|---------|----------|
| Random Forest | **Patient Holdout** | **99.87%** | 99.87% | 99.87% | 99.87% |
| SVM | Patient Holdout | 96.33% | 96.46% | 96.33% | 96.31% |
| Random Forest | Beat Holdout | 96.67% | 96.70% | 96.67% | 96.66% |
| SVM | Beat Holdout | 93.84% | 96.04% | 93.84% | 93.78% |

**Key Findings:**
- Patient holdout achieved higher accuracy than beat holdout (eliminating data leakage)
- Random Forest consistently outperformed SVM by 2.83-3.54 percentage points
- All models exceeded 93% accuracy with >96% precision (clinically acceptable)

### Feature Importance Analysis

**QRS Complex Dominance:**
- **64.5-75.8%** of total importance across all models
- **Segment 6** (middle of QRS, features 125-149) most critical: 9.3-46.1% importance
- Aligns with cardiac physiology: ventricular depolarization shows most distinctive morphology

**Top 3 Segments per Model:**

| Model | Segment | Cardiac Phase | Importance |
|-------|---------|---------------|------------|
| **SVM Patient** | 6 (QRS) | Ventricular depolarization | 46.1% |
| | 7 (QRS) | Late QRS | 19.4% |
| | 5 (QRS) | Early QRS | 10.3% |
| **RF Patient** | 6 (QRS) | Ventricular depolarization | 21.5% |
| | 7 (QRS) | Late QRS | 26.7% |
| | **4 (PR)** | AV node timing | 7.0% |

**Model-Specific Patterns:**
- **SVM:** Concentrates on single most discriminative feature (Segment 6: 41.0-46.1%)
- **Random Forest:** Distributes importance across multiple QRS segments (9.3-26.7%)
- RF's balanced approach may better handle variable ECG quality in clinical settings

### Clustering Analysis

**Cluster-Class Alignment:**
- Overall purity: **44.9%** (poor natural separation)
- Best clustering: LBBB (67%), Paced beats (63%)
- Most classes: 28.5-51.4% purity (overlapping feature distributions)

**Dimensionality Reduction:**
- Classification with 8 cluster-based features (distances to cluster centers):
  - SVM Patient: **48.17%** accuracy (-48.2% drop) — catastrophic failure
  - RF Patient: 99.17% accuracy (-0.7% drop) — remarkably resilient
- **Conclusion:** Full 275-dimensional temporal resolution necessary for accurate classification

---

## 📁 Project Structure

```
ecg-classification-and-explainability-analysis/
├── notebooks/
│   ├── data_preprocess.ipynb          # R-peak detection, segmentation, normalization
│   ├── 1_data_exploration.ipynb       # EDA and waveform visualization
│   ├── data_split_resample.ipynb      # Train/test split and class balancing
│   ├── 2_beat_holdout_classification.ipynb      # Beat-level validation
│   ├── 3_patient_holdout_classification.ipynb   # Patient-level validation
│   ├── 4_explainability_analysis.ipynb          # PFI analysis
│   └── 5_clustering_classification_and_analysis.ipynb  # K-Means clustering
├── docs/
│   └── CS1_Report.pdf                 # Complete methodology and analysis
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Technologies

**Languages & Core Libraries:**
- Python 3.x
- NumPy, pandas, SciPy

**Machine Learning:**
- scikit-learn (SVM, Random Forest, K-Means, PCA, t-SNE)
- Cross-validation, stratified splitting

**Signal Processing:**
- wfdb (MIT-BIH database reading)
- biosppy (biosignal processing, R-peak detection)

**Visualization:**
- matplotlib, seaborn

**Environment:**
- Jupyter Notebook

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/mayayayo/ecg-classification-and-explainability-analysis.git
   cd ecg-classification-and-explainability-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download MIT-BIH Database** (if needed)
   ```python
   # The data_preprocess.ipynb notebook includes code to download from PhysioNet
   ```

4. **Run notebooks sequentially**
   ```bash
   jupyter notebook
   ```
   Execute in order:
   1. `data_preprocess.ipynb`
   2. `1_data_exploration.ipynb`
   3. `data_split_resample.ipynb`
   4. `2_beat_holdout_classification.ipynb`
   5. `3_patient_holdout_classification.ipynb`
   6. `4_explainability_analysis.ipynb`
   7. `5_clustering_classification_and_analysis.ipynb`

---

## 🔍 Key Insights & Clinical Implications

### What Makes This Work Clinically Relevant

1. **Patient-Level Validation:** Unlike most studies that use random beat splitting (data leakage), this work tests on completely unseen patients, providing realistic generalization estimates.

2. **Physiologically Validated Features:** PFI analysis confirms models learned genuine cardiac patterns (QRS complex dominance) rather than artifacts or noise.

3. **High Precision (>96%):** Minimizes false alarms, critical for clinical adoption where alert fatigue is a major concern.

4. **Full Interpretability:** Clinicians can verify which ECG segments drive predictions, building trust for decision support deployment.

### Limitations & Future Directions

**Current Limitations:**
- 1980s dataset (older recording technology)
- Only 8 arrhythmia classes (dozens exist clinically)
- Small patient holdout test set (5 patients)
- Preprocessing dependency (R-peak detection errors propagate)
- Default hyperparameters (not optimized)

**Future Work:**
- Multi-center validation with hundreds of patients
- Deep learning comparison (CNNs, LSTMs with attention mechanisms)
- Multi-modal analysis (demographics, 12-lead ECG)
- Real-time monitoring system implementation
- Prospective validation against cardiologist interpretations

---

## 📖 Documentation

Full methodology, analysis, and discussion available in [`docs/CS1_Report.pdf`](docs/CS1_Report.pdf)

**Report Sections:**
1. Introduction & Research Aims
2. Methodology (Dataset, Validation Protocols, Models, Explainability, Clustering)
3. Results (Classification Performance, Feature Importance, Clustering Analysis)
4. Discussion (Performance Analysis, Feature Insights, Clinical Implications)
5. Conclusion

---

## 👤 Author

**Tanzila Tahsin Mayabee**
- MSc Data Science, University of Glasgow (2024-2026)
- BSc Computer Science Engineering, Independent University Bangladesh (2023)
- First Author: *ECG Signal Classification Using Transfer Learning and CNNs* (Springer, 2023)

**Contact:** mayabee.tahsin@gmail.com

---

## 📚 References

- Moody, G.B., & Mark, R.G. (2001). The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.
- Kraik, K., et al. (2025). The most common errors in automatic ECG interpretation. *Frontiers in Physiology*, 16, 1590170.

---

## 📝 License

This project was completed as academic coursework for COMPSCI5100 at University of Glasgow.

---

*This project demonstrates expertise in machine learning pipeline development, medical signal processing, model explainability, and rigorous validation methodology for clinical applications.*
