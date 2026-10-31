# ---------------------------------------------------------
# app.py  (FINAL FIXED + FULLY WORKING)
# ---------------------------------------------------------
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO


import os, json, joblib, math, pickle, pandas as pd
from models import TRAIN_URL, TEST_URL, cols
from isolation_artifact import IsolationForestArtifact
# ---------------------------------------------------------
# FLASK + SOCKET CONFIG
# ---------------------------------------------------------
app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


MODEL_DIR = "models"
REPLAY_INTERVAL_SECONDS = float(os.getenv("REPLAY_INTERVAL_SECONDS", "1.0"))
if not math.isfinite(REPLAY_INTERVAL_SECONDS) or REPLAY_INTERVAL_SECONDS <= 0:
    raise ValueError("REPLAY_INTERVAL_SECONDS must be a finite positive value")

MODEL_GENERATION_COMMAND = "python models.py"
FEATURE_COLUMNS = [
    column for column in cols
    if column not in {"label", "difficulty_level"}
]
NUMERIC_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in {"protocol_type", "service", "flag"}
]
INDIVIDUAL_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "knn"
]


def _artifact_generation_help(path, reason):
    return (
        f"{reason}: '{path}'. To generate compatible artifacts, run "
        f"'{MODEL_GENERATION_COMMAND}' from the project root. This retrains "
        "the models and overwrites files in the models directory; do so only "
        "when intentional."
    )


def _load_required_json(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            _artifact_generation_help(path, f"Required {description} is missing")
        )
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            _artifact_generation_help(
                path, f"Required {description} could not be read"
            )
        ) from exc


def _load_required_joblib(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            _artifact_generation_help(path, f"Required {description} is missing")
        )
    try:
        return joblib.load(path)
    except (
        OSError,
        EOFError,
        ValueError,
        ImportError,
        AttributeError,
        TypeError,
        KeyError,
        pickle.UnpicklingError,
    ) as exc:
        raise RuntimeError(
            _artifact_generation_help(
                path,
                f"Required {description} is incompatible or could not be loaded",
            )
        ) from exc


MODEL_NAMES = _load_required_json(
    os.path.join(MODEL_DIR, "model_list.json"),
    "supervised model configuration",
)
if (
    not isinstance(MODEL_NAMES, list)
    or len(MODEL_NAMES) != len(INDIVIDUAL_MODEL_NAMES)
    or any(not isinstance(name, str) for name in MODEL_NAMES)
    or set(MODEL_NAMES) != set(INDIVIDUAL_MODEL_NAMES)
):
    raise RuntimeError(
        _artifact_generation_help(
            os.path.join(MODEL_DIR, "model_list.json"),
            f"Configuration must list exactly {INDIVIDUAL_MODEL_NAMES}",
        )
    )

METRICS_CACHE = _load_required_json(
    os.path.join(MODEL_DIR, "metrics_summary.json"),
    "supervised metrics configuration",
)
required_metric_models = set(INDIVIDUAL_MODEL_NAMES) | {"majority_voting"}
required_metric_fields = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "roc_auc",
}
if (
    not isinstance(METRICS_CACHE, dict)
    or not required_metric_models.issubset(METRICS_CACHE)
    or any(
        not isinstance(METRICS_CACHE[name], dict)
        or not required_metric_fields.issubset(METRICS_CACHE[name])
        for name in required_metric_models
    )
):
    raise RuntimeError(
        _artifact_generation_help(
            os.path.join(MODEL_DIR, "metrics_summary.json"),
            "Metrics configuration does not contain valid entries for all models",
        )
    )


def _load_isolation_artifact():
    path = os.path.join(MODEL_DIR, "isoforest.joblib")
    artifact = _load_required_joblib(path, "calibrated Isolation Forest artifact")
    try:
        preprocessor, detector = artifact
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            _artifact_generation_help(
                path,
                "Isolation Forest artifact must provide a (preprocessor, model) pair",
            )
        ) from exc

    if isinstance(artifact, IsolationForestArtifact) and not all(
        hasattr(artifact, field)
        for field in ("threshold", "alpha", "variant")
    ):
        raise RuntimeError(
            _artifact_generation_help(
                path, "Isolation Forest artifact is missing calibration metadata"
            )
        )
    if (
        not callable(getattr(preprocessor, "transform", None))
        or not callable(getattr(detector, "predict", None))
    ):
        raise RuntimeError(
            _artifact_generation_help(
                path,
                "Isolation Forest artifact has incompatible preprocessing or model objects",
            )
        )
    try:
        actual_features = list(preprocessor.feature_names_in_)
    except (AttributeError, TypeError):
        actual_features = []
    if actual_features != NUMERIC_FEATURE_COLUMNS:
        raise RuntimeError(
            _artifact_generation_help(
                path,
                "Isolation Forest artifact feature schema does not match the project",
            )
        )
    return artifact


ISOLATION_MODEL = _load_isolation_artifact()
print("Loaded Isolation Forest")


# Load Model-B (multi-class)
ATTACK_TYPE_MODEL = None
if os.path.exists("models/attack_type_classifier.joblib"):
    ATTACK_TYPE_MODEL = joblib.load("models/attack_type_classifier.joblib")
    print("Loaded Model-B: attack_type_classifier.joblib")


# Model cache
loaded_models = {}


def load_model(name):
    """Loads a model pipeline only once."""
    if name not in loaded_models:
        path = os.path.join(MODEL_DIR, f"{name}_pipeline.joblib")
        model = _load_required_joblib(path, f"{name} supervised model pipeline")
        if not callable(getattr(model, "predict", None)):
            raise RuntimeError(
                _artifact_generation_help(
                    path, f"The {name} artifact is not a prediction pipeline"
                )
            )
        try:
            actual_features = list(model.feature_names_in_)
        except (AttributeError, TypeError):
            actual_features = []
        if actual_features != FEATURE_COLUMNS:
            raise RuntimeError(
                _artifact_generation_help(
                    path, f"The {name} pipeline feature schema does not match"
                )
            )
        loaded_models[name] = model
        print(f"Loaded model: {name}")
    return loaded_models[name]


for required_model_name in [*INDIVIDUAL_MODEL_NAMES, "majority_voting"]:
    load_model(required_model_name)




# -----------------------------------------------------------
# NEW: MULTI-CLASS ATTACK TYPE CLASSIFIER (Model-B)
# -----------------------------------------------------------
def train_attack_type_classifier():
    """TODO: Unfinished until attack taxonomy and evaluation are verified."""
    print("Training Multi-Class Attack Type Classifier...")


    df_train = pd.read_csv(TRAIN_URL, names=cols)
    df_test = pd.read_csv(TEST_URL, names=cols)
    df = pd.concat([df_train, df_test], ignore_index=True)


    df.drop("difficulty_level", axis=1, inplace=True)


    df["label"] = df["label"].str.strip().str.lower()


    df["attack_type"] = df["label"].apply(
        lambda x: "normal" if "normal" in x else x
    )


    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["label", "attack_type"]]


    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


    df.dropna(subset=num_cols, how="all", inplace=True)


    X = df.drop(["label", "attack_type"], axis=1)
    y = df["attack_type"]


    pre = build_preprocessor(num_cols, cat_cols)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    clf = RandomForestClassifier(n_estimators=150, random_state=42)


    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)


    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)


    print("Attack-Type Classifier Accuracy:", acc)


    joblib.dump(pipe, f"{MODEL_DIR}/attack_type_classifier.joblib")
    print("Saved → models/attack_type_classifier.joblib")




# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")




@app.route("/metrics")
def metrics():
    return jsonify(METRICS_CACHE)




# ---------------------------------------------------------
# BEHAVIORAL CHECK ROUTE
# ---------------------------------------------------------
@app.route("/behavior_check", methods=["POST"])
def behavior_check():
    if ISOLATION_MODEL is None:
        return jsonify({"error": "Isolation Forest not loaded"}), 503

    num_pipe, iso = ISOLATION_MODEL

    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "Expected a JSON object"}), 400

        numeric_features = list(num_pipe.feature_names_in_)
        missing = [col for col in numeric_features if col not in data]

        if missing:
            return jsonify({
                "error": "Missing numerical features",
                "missing": missing
            }), 400

        df = pd.DataFrame([data])[numeric_features]
        df = df.apply(pd.to_numeric, errors="coerce")

        X_scaled = num_pipe.transform(df)
        pred = int(iso.predict(X_scaled)[0])

        return jsonify({
            "prediction": pred,
            "behavior_status": (
                "anomalous_behavior" if pred == -1
                else "normal_behavior"
            )
        })

    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400


def compute_risk(threat_score, attack_type=None):
    severity_map = {
        "neptune": 25,
        "portsweep": 10,
        "backdoor": 30,
        "data_theft": 40
    }


    sev = severity_map.get(attack_type, 0)
    final_score = threat_score + sev


    if final_score < 20:
        return "Low", final_score
    elif final_score < 50:
        return "Medium", final_score
    elif final_score < 80:
        return "High", final_score
    else:
        return "Critical", final_score




# ---------------------------------------------------------
# NSL-KDD DATASET REPLAY
# ---------------------------------------------------------
def simulator():
    print("NSL-KDD KDDTest+ dataset replay running...")
    replay_records = pd.read_csv(TEST_URL, names=cols)
    if replay_records.empty:
        raise RuntimeError("The KDDTest+ dataset contains no replay records")

    missing_columns = [
        column for column in cols if column not in replay_records.columns
    ]
    if missing_columns:
        raise ValueError(
            f"KDDTest+ is missing expected columns: {missing_columns}"
        )

    if set(MODEL_NAMES) != set(INDIVIDUAL_MODEL_NAMES):
        raise RuntimeError(
            "model_list.json must contain exactly the configured individual "
            f"models: {INDIVIDUAL_MODEL_NAMES}"
        )

    ensemble_model = load_model("majority_voting")
    individual_models = {
        name: load_model(name)
        for name in INDIVIDUAL_MODEL_NAMES
    }

    for model_name, model in {
        "majority_voting": ensemble_model,
        **individual_models
    }.items():
        expected = list(model.feature_names_in_)
        if expected != FEATURE_COLUMNS:
            raise ValueError(
                f"{model_name} feature schema does not match the replay "
                "dataset schema"
            )

    print(
        f"Loaded {len(replay_records)} KDDTest+ records; "
        f"replaying one every {REPLAY_INTERVAL_SECONDS:g} seconds. "
        "This is dataset replay, not live traffic capture."
    )

    t = 0
    row_index = 0

    while True:
        if row_index >= len(replay_records):
            print("Reached end of KDDTest+ replay; restarting from first record.")
            row_index = 0

        record = replay_records.iloc[row_index]
        replay_index = row_index
        row_index += 1
        ground_truth_label = str(record["label"]).strip().lower()
        if not ground_truth_label:
            raise ValueError(
                f"KDDTest+ record {replay_index} has an empty label"
            )
        ground_truth_binary = int(ground_truth_label != "normal")

        # Exclude both the label and difficulty metadata before model inference.
        features = record.loc[FEATURE_COLUMNS].to_frame().T
        if list(features.columns) != FEATURE_COLUMNS:
            raise ValueError(
                f"KDDTest+ record {replay_index} has an unexpected feature schema"
            )

        prediction = int(ensemble_model.predict(features)[0])
        model_predictions = {
            name: int(model.predict(features)[0])
            for name, model in individual_models.items()
        }
        model_probabilities = {}
        for name, model in individual_models.items():
            attack_class_index = list(model.classes_).index(1)
            model_probabilities[name] = float(
                model.predict_proba(features)[0, attack_class_index]
            )

        proba = model_probabilities["random_forest"]
        attack_type = (
            "unclassified_attack" if prediction == 1 else "normal"
        )
        threat_score = round(proba * 100)
        risk_level, risk_value = compute_risk(threat_score, None)

        behavior_status = "not_available"
        if ISOLATION_MODEL is not None:
            num_pipe, iso = ISOLATION_MODEL
            numeric_features = list(num_pipe.feature_names_in_)
            numeric_record = features[numeric_features].apply(
                pd.to_numeric,
                errors="coerce"
            )
            X_num = num_pipe.transform(numeric_record)
            iso_pred = int(iso.predict(X_num)[0])
            behavior_status = (
                "anomalous_behavior"
                if iso_pred == -1
                else "normal_behavior"
            )

        socketio.emit("chart_update", {
            "t": t,
            "model": "majority_voting",
            "prediction": prediction,
            "supervised_prediction": prediction,
            "ground_truth_label": ground_truth_label,
            "ground_truth_binary": ground_truth_binary,
            "prediction_matches_ground_truth": (
                prediction == ground_truth_binary
            ),
            "replay_index": replay_index,
            "model_predictions": model_predictions,
            "proba": proba,
            "probability_source": "random_forest",
            "model_probabilities": model_probabilities,
            "threat_score": threat_score,
            "risk_level": risk_level,
            "risk_value": risk_value,
            "behavior_status": behavior_status,
            "isolation_anomaly_status": behavior_status,
            "attack_type": attack_type,
        })

        t += 1
        socketio.sleep(REPLAY_INTERVAL_SECONDS)




# ---------------------------------------------------------
# SOCKET.IO EVENTS
# ---------------------------------------------------------
@socketio.on("connect")
def connected():
    print("Client connected")




# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    socketio.start_background_task(simulator)
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
