
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from models import load_and_prepare, build_preprocessor

# Load datasets; official test set is intentionally not used here.
X_train, _, y_train, _, num_cols, cat_cols = load_and_prepare()

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

pipe = Pipeline([
    ("pre", build_preprocessor(num_cols, cat_cols)),
    ("clf", RandomForestClassifier(
        n_estimators=120,
        random_state=42
    ))
])

pipe.fit(X_fit, y_fit)

prob = pipe.predict_proba(X_val)[:, 1]

print("\nRandom Forest validation results:")
print("Threshold | Accuracy | Precision | Recall | F1")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (prob >= threshold).astype(int)

    acc = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred, zero_division=0)
    rec = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)

    print(
        f"{threshold:9.2f} | "
        f"{acc:8.4f} | "
        f"{prec:9.4f} | "
        f"{rec:6.4f} | "
        f"{f1:6.4f}"
    )
