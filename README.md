# Adaptive Firewall — NSL-KDD Intrusion Detection Demonstration

An ML-based Intrusion Detection System that combines supervised classification and unsupervised anomaly detection to identify suspicious network activity using the NSL-KDD dataset. The project integrates multiple machine-learning models, a calibrated Isolation Forest detector, and an interactive Flask and Socket.IO dashboard for sequential detection-result visualization.

## Features

1. Binary normal-versus-attack classification using Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN).
2. Hard majority-voting ensemble combining the three supervised classifiers.
3. Isolation Forest anomaly detection with a threshold calibrated using normal validation records.
4. Separate training, validation, and official test data handling.
5. Flask and Socket.IO dashboard for sequential KDDTest+ replay.
6. Record-level predictions, ground-truth comparison, anomaly status, and evaluation metrics.

## Architecture and Workflow

1. **Dataset loading:** `models.py` loads the official KDDTrain+ and KDDTest+ datasets separately.
2. **Training and validation:** KDDTrain+ is divided into stratified fitting and validation subsets. Validation attack-class F1 is used to select the supervised `best_pipeline`.
3. **Final supervised training:** Candidate models are refitted on the complete KDDTrain+ dataset and evaluated on the official KDDTest+ dataset.
4. **Anomaly detection:** Two Isolation Forest variants are compared using training and validation data. The selected model and calibrated threshold are saved.
5. **Artifact generation:** Trained pipelines, model metadata, and evaluation metrics are stored locally under `models/`.
6. **Dashboard replay:** `app.py` loads the saved artifacts and replays KDDTest+ records sequentially, sending predictions and ground truth to the dashboard through Socket.IO.

The official test set is reserved for final evaluation and replay. It is not used for supervised model selection or Isolation Forest threshold calibration.

## Dataset and Preprocessing

The project uses the NSL-KDD dataset:

- [KDDTrain+.txt — Training dataset](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt)
- [KDDTest+.txt — Testing dataset](https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt)

The dataset contains 41 traffic features, a label, and `difficulty_level` metadata.

- The label and difficulty metadata are excluded from model inputs.
- `protocol_type`, `service`, and `flag` are categorical features.
- The remaining 38 features are numerical.
- Supervised numerical features use median imputation and standardization.
- Categorical features use most-frequent-value imputation and one-hot encoding.

The supervised target is binary:

- `normal` → 0
- All other dataset labels → 1 (attack)

Labels such as `neptune`, `satan`, and `buffer_overflow` represent attack families in the dataset. The current supervised models do not predict individual attack families. During offline replay, the original dataset label is displayed as ground truth, but it is not provided to the prediction models.

## Machine Learning Models

### Supervised Classification

The project compares four approaches:

| Model | Description |
|---|---|
| Logistic Regression | Binary classification baseline |
| Random Forest | Ensemble-based binary classifier |
| K-Nearest Neighbors (KNN) | Distance-based binary classifier |
| Majority Voting | Hard-voting ensemble of Logistic Regression, Random Forest, and KNN |

The `best_pipeline` is selected using validation attack-class F1. The final candidate pipelines are then refitted on all KDDTrain+ records.

The Random Forest attack probability is also used for the dashboard probability visualization and informational threat score.

### Isolation Forest

The project compares two Isolation Forest variants:

1. **Mixed variant:** Trained using all records in the training fitting subset.
2. **Normal-only variant:** Trained using only normal records from the training fitting subset.

Each variant has its own numerical preprocessing pipeline.

Normal records in the validation subset are divided into separate calibration and evaluation subsets. The anomaly threshold is calibrated from the lower tail of the calibration-normal `score_samples()` distribution using a predefined alpha of 0.05.

The selected variant is determined using validation attack F1, with evaluation-normal false-positive rate used to break ties.

The saved artifact contains:

- Numerical preprocessing pipeline
- Trained Isolation Forest model
- Calibrated threshold
- Alpha value
- Selected variant

The current saved artifact uses the **normal-only variant**, with a threshold of:

`-0.4979774121836357`

A record is flagged as anomalous when its score falls below this threshold.

An anomaly indicates an outlier according to the model; it is not proof of a confirmed attack. The 5% calibration target is not a guarantee of the false-positive rate on unseen data.

## Evaluation Results

### Supervised Models — Official KDDTest+

The following results are from the current `models/metrics_summary.json`. Precision, recall, and F1 refer to the positive attack class.

Confusion matrices use the format:

`[[True Normal, False Positive], [False Negative, True Attack]]`

| Model | Accuracy | Attack Precision | Attack Recall | Attack F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.43% | 91.76% | 62.43% | 74.31% | 0.7921 |
| Random Forest | 76.45% | 96.69% | 60.72% | 74.59% | 0.9598 |
| KNN | 78.03% | 96.79% | 63.50% | 76.69% | 0.8274 |
| Majority Voting | 77.63% | 96.48% | 63.00% | 76.23% | N/A |

**Confusion matrices:**

- Logistic Regression: `[[8992, 719], [4821, 8012]]`
- Random Forest: `[[9444, 267], [5041, 7792]]`
- KNN: `[[9441, 270], [4684, 8149]]`
- Majority Voting: `[[9416, 295], [4748, 8085]]`

ROC-AUC is unavailable for majority voting because the implementation uses hard voting and does not provide class probabilities.

The official KDDTest+ evaluation contains 9,711 normal records and 12,833 attack records.

### Isolation Forest — Official KDDTest+

The saved Isolation Forest artifact and its unchanged threshold were independently evaluated on all 22,544 KDDTest+ records. No fitting, retraining, or threshold calibration was performed on the test set.

| Variant | Accuracy | Attack Precision | Attack Recall | Attack F1 | Normal False-Positive Rate |
|---|---:|---:|---:|---:|---:|
| Normal-only | 78.34% | 93.38% | 66.68% | 77.80% | 6.25% |

Confusion matrix:

`[[9104, 607], [4276, 8557]]`

The measured test false-positive rate differs from the 5% validation calibration target. This illustrates why validation calibration does not guarantee identical performance on unseen data.

These results describe the current implementation. They are **not the published paper's reported results**. In particular, the paper's 99.58% accuracy figure has not been reproduced by this implementation.

## Dashboard

The Flask and Socket.IO dashboard provides:

- Normal and attack prediction counts.
- Random Forest attack-probability visualization.
- Sequential KDDTest+ replay alerts.
- Ensemble predictions and ground-truth labels.
- Prediction-match indicators.
- Separate Isolation Forest anomaly status.
- Saved supervised evaluation metrics.
- Model comparison charts for accuracy, precision, recall, and F1.
- Individual classifier attack probabilities.
- A `POST /behavior_check` endpoint for Isolation Forest prediction using numerical features.

The dashboard replays one KDDTest+ record at a time, using `REPLAY_INTERVAL_SECONDS` (default: 1 second). Replay starts again from the first record after reaching the end of the dataset.

Ground truth is displayed because the replay uses labelled dataset records. It would not be available in real-time inference on unlabeled network traffic.

Risk levels shown in the dashboard are informational values derived from Random Forest attack probability. They do not trigger firewall rules or packet-blocking actions.

## Installation and Execution

### Prerequisites

- Python 3.11 or newer recommended.
- Git.
- Internet access to download the dataset and browser-side CDN resources.

### 1. Clone the repository

```powershell
git clone YOUR_REPOSITORY_URL
cd C1_4_Adaptive_Firewall
```

Replace `YOUR_REPOSITORY_URL` with your repository's GitHub URL.

### 2. Create and activate a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate model artifacts

Model artifacts are generated locally and are excluded from Git through `.gitignore`.

If the required artifacts are not present, run:

```powershell
python models.py
```

This command downloads the datasets, trains and evaluates the models, and generates the required artifacts under `models/`. It may take time and will overwrite existing generated model artifacts.

**Do not run this command if you intend to preserve existing model artifacts.**

### 5. Start the dashboard

```powershell
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

To change the replay interval before launching:

```powershell
$env:REPLAY_INTERVAL_SECONDS = "1.0"
python app.py
```

The application uses Flask-SocketIO threading mode, with debug mode and the automatic reloader disabled for normal startup.

Flask's built-in development server is intended for local demonstration, not production deployment.

## Project Structure

```text
C1_4_Adaptive_Firewall/
│
├── app.py
├── models.py
├── isolation_artifact.py
├── evaluate_rf.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html
│
└── models/                 # Locally generated; ignored by Git
    ├── best_pipeline.joblib
    ├── isoforest.joblib
    ├── knn_pipeline.joblib
    ├── logistic_regression_pipeline.joblib
    ├── majority_voting_pipeline.joblib
    ├── metrics_summary.json
    ├── model_list.json
    └── random_forest_pipeline.joblib
```

## Scope and Future Enhancements

The current implementation demonstrates ML-based intrusion detection using the NSL-KDD dataset and offline dataset replay. It does not yet include live packet capture, automated traffic blocking, production deployment, or operational adaptive retraining.

Future enhancements include:
- Multi-class attack-family classification.
- Evaluation on additional intrusion-detection datasets.
- Live traffic monitoring and carefully controlled response mechanisms.
- Production deployment and API security.

## Technologies Used

- Python
- Flask
- Flask-SocketIO
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML, CSS, JavaScript
- Bootstrap
- Chart.js
- NSL-KDD dataset

---

**Project scope:** An educational machine-learning intrusion detection demonstration focused on comparing supervised classification and anomaly detection techniques using NSL-KDD.