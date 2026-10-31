# -------------------------------------------
# models.py  (100% WORKING)
# -------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from isolation_artifact import IsolationForestArtifact


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os, json

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
]


def load_and_prepare():
    # Load the official training and testing datasets separately
    df_train = pd.read_csv(TRAIN_URL, names=cols)
    df_test = pd.read_csv(TEST_URL, names=cols)

    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [
        c for c in cols
        if c not in cat_cols + ["label", "difficulty_level"]
    ]

    def prepare(df):
        df = df.copy()

        # Remove the dataset's difficulty-level metadata
        df.drop(columns=["difficulty_level"], inplace=True)

        # Convert labels to binary: normal = 0, attack = 1
        df["label"] = (
            df["label"]
            .str.strip()
            .str.lower()
            .apply(lambda x: 0 if x == "normal" else 1)
        )

        # Convert numerical features to numeric values
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remove records where every numerical feature is missing
        df.dropna(subset=num_cols, how="all", inplace=True)

        X = df.drop(columns=["label"])
        y = df["label"].astype(int)

        return X, y

    X_train, y_train = prepare(df_train)
    X_test, y_test = prepare(df_test)

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    return X_train, X_test, y_train, y_test, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('enc', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ]
    )


def train_and_evaluate_isolation_forest(
    X_fit,
    y_fit,
    X_val,
    y_val,
    num_cols,
    alpha=0.05,
    random_state=43
):
    """Compare mixed and normal-only fits, calibrating only on normal validation rows."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    normal_validation_indices = X_val.index[y_val == 0]
    calibration_indices, evaluation_normal_indices = train_test_split(
        normal_validation_indices,
        test_size=0.5,
        random_state=random_state
    )
    attack_validation_indices = X_val.index[y_val == 1]

    calibration_normals = X_val.loc[calibration_indices, num_cols]
    evaluation_normals = X_val.loc[evaluation_normal_indices, num_cols]
    evaluation_attacks = X_val.loc[attack_validation_indices, num_cols]
    evaluation_X = pd.concat(
        [evaluation_normals, evaluation_attacks],
        axis=0
    )
    evaluation_y = np.concatenate([
        np.zeros(len(evaluation_normals), dtype=int),
        np.ones(len(evaluation_attacks), dtype=int)
    ])

    training_sets = {
        "mixed_training": X_fit[num_cols],
        "normal_only": X_fit.loc[y_fit == 0, num_cols]
    }
    results = {}
    artifacts = {}

    for variant, fit_data in training_sets.items():
        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        fit_values = preprocessor.fit_transform(fit_data)

        model = IsolationForest(
            contamination="auto",
            random_state=42
        )
        model.fit(fit_values)

        calibration_values = preprocessor.transform(calibration_normals)
        calibration_scores = np.sort(
            model.score_samples(calibration_values)
        )
        threshold_index = int(np.floor(alpha * (len(calibration_scores) - 1)))
        threshold = float(calibration_scores[threshold_index])

        # IsolationForest.predict flags scores below offset_ as outliers.
        model.offset_ = threshold

        evaluation_values = preprocessor.transform(
            evaluation_X[num_cols]
        )
        evaluation_scores = model.score_samples(evaluation_values)
        # An anomaly is an outlier flag, not a confirmed attack.
        # Labels are used here only to evaluate attack-detection performance.
        predictions = (evaluation_scores < threshold).astype(int)
        matrix = confusion_matrix(
            evaluation_y,
            predictions,
            labels=[0, 1]
        )
        true_normal, false_positive = matrix[0]

        results[variant] = {
            "attack_precision": float(
                precision_score(evaluation_y, predictions, zero_division=0)
            ),
            "attack_recall": float(
                recall_score(evaluation_y, predictions, zero_division=0)
            ),
            "attack_f1": float(
                f1_score(evaluation_y, predictions, zero_division=0)
            ),
            "normal_false_positive_rate": float(
                false_positive / (true_normal + false_positive)
            ),
            "confusion_matrix": matrix.tolist(),
            "threshold": threshold,
            "target_false_positive_rate": alpha
        }
        artifacts[variant] = IsolationForestArtifact(
            preprocessor=preprocessor,
            model=model,
            threshold=threshold,
            alpha=alpha,
            variant=variant
        )

    selected_variant = max(
        results,
        key=lambda name: (
            results[name]["attack_f1"],
            -results[name]["normal_false_positive_rate"]
        )
    )
    selected_artifact = artifacts[selected_variant]
    joblib.dump(selected_artifact, f"{MODEL_DIR}/isoforest.joblib")

    for variant, metrics in results.items():
        print(
            f"\nIsolation Forest {variant} "
            f"(validation confusion matrix labels=[normal, attack]):"
        )
        print(metrics["confusion_matrix"])
        print(
            f"attack_precision={metrics['attack_precision']:.4f}, "
            f"attack_recall={metrics['attack_recall']:.4f}, "
            f"attack_f1={metrics['attack_f1']:.4f}, "
            "normal_false_positive_rate="
            f"{metrics['normal_false_positive_rate']:.4f}, "
            f"threshold={metrics['threshold']:.8f}, "
            f"target_fpr={alpha:.2%}"
        )

    print(
        f"Selected Isolation Forest variant by validation attack F1 "
        f"(lower normal FPR breaks ties): {selected_variant}"
    )
    print(
        "An anomaly is an outlier under this model, not confirmation "
        "that a record is an attack."
    )
    return results, selected_artifact


def train_and_save_all():
    X_train, X_test, y_train, y_test, num_cols, cat_cols = load_and_prepare()

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    def report_class_distribution(name, labels):
        counts = labels.value_counts().reindex([0, 1], fill_value=0)
        percentages = labels.value_counts(
            normalize=True
        ).reindex([0, 1], fill_value=0) * 100
        print(
            f"{name} class distribution "
            f"(normal=0, attack=1): "
            f"normal={counts[0]} ({percentages[0]:.2f}%), "
            f"attack={counts[1]} ({percentages[1]:.2f}%)"
        )

    report_class_distribution("Training", y_fit)
    report_class_distribution("Validation", y_val)

    model_names = [
        "logistic_regression",
        "random_forest",
        "knn",
        "majority_voting"
    ]

    def make_models():
        return {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(
                n_estimators=120, random_state=42
            ),
            "knn": KNeighborsClassifier(n_neighbors=5),
            "majority_voting": VotingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=120,
                            random_state=42
                        )
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=5)),
                    ("lr", LogisticRegression(max_iter=500))
                ],
                voting="hard"
            )
        }

    validation_f1 = {}
    for name, model in make_models().items():
        pipe = Pipeline([
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("clf", model)
        ])
        pipe.fit(X_fit, y_fit)
        validation_pred = pipe.predict(X_val)
        score = f1_score(y_val, validation_pred, zero_division=0)
        validation_f1[name] = score
        print(
            f"Validation {name}: "
            f"accuracy={accuracy_score(y_val, validation_pred):.4f}, "
            f"attack_precision="
            f"{precision_score(y_val, validation_pred, zero_division=0):.4f}, "
            f"attack_recall="
            f"{recall_score(y_val, validation_pred, zero_division=0):.4f}, "
            f"attack_f1={score:.4f}"
        )

    best_model_name = max(model_names, key=validation_f1.get)
    print(
        f"Selected best_pipeline by validation attack F1: "
        f"{best_model_name} ({validation_f1[best_model_name]:.4f})"
    )

    final_pipelines = {}
    for name, model in make_models().items():
        print(f"Fitting {name} on the full official training set...")
        pipe = Pipeline([
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("clf", model)
        ])
        pipe.fit(X_train, y_train)
        final_pipelines[name] = pipe
        joblib.dump(pipe, f"{MODEL_DIR}/{name}_pipeline.joblib")

    joblib.dump(
        final_pipelines[best_model_name],
        f"{MODEL_DIR}/best_pipeline.joblib"
    )

    metrics_summary = {}
    report_class_distribution("Official test", y_test)
    for name, pipe in final_pipelines.items():
        test_pred = pipe.predict(X_test)
        test_metrics = {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "precision": float(
                precision_score(y_test, test_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, test_pred, zero_division=0)
            ),
            "f1": float(f1_score(y_test, test_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_test, test_pred, labels=[0, 1]
            ).tolist()
        }

        try:
            test_prob = pipe.predict_proba(X_test)[:, 1]
            test_metrics["roc_auc"] = float(
                roc_auc_score(y_test, test_prob)
            )
        except (AttributeError, ValueError):
            test_metrics["roc_auc"] = None

        metrics_summary[name] = test_metrics
        print(
            f"\nOfficial test {name} "
            f"(confusion matrix labels=[normal, attack]):"
        )
        print(test_metrics["confusion_matrix"])
        print(
            f"accuracy={test_metrics['accuracy']:.4f}, "
            f"attack_precision={test_metrics['precision']:.4f}, "
            f"attack_recall={test_metrics['recall']:.4f}, "
            f"attack_f1={test_metrics['f1']:.4f}, "
            f"roc_auc={test_metrics['roc_auc']}"
        )

    with open(f"{MODEL_DIR}/model_list.json", "w") as f:
        json.dump(model_names[:3], f)

    with open(f"{MODEL_DIR}/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    train_and_evaluate_isolation_forest(
        X_fit,
        y_fit,
        X_val,
        y_val,
        num_cols
    )

    print("All models saved successfully.")


def generate_geo_sample(n=1, center=(28.61, 77.21), radius_km=50):
    out = []
    lat0, lon0 = center
    for _ in range(n):
        d = np.random.random() * radius_km
        angle = np.random.random() * 360
        dlat = (d / 111) * np.cos(np.deg2rad(angle))
        dlon = (d / 111) * np.sin(np.deg2rad(angle)) / np.cos(np.deg2rad(lat0))
        out.append({"lat": float(lat0 + dlat), "lon": float(lon0 + dlon)})
    return out


if __name__ == "__main__":
    train_and_save_all()
    # TODO: Keep multi-class training separate until its target taxonomy and
    # evaluation methodology are verified; it is intentionally not invoked here.
    
