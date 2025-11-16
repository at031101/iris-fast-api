#!/usr/bin/env python3
"""
poison_and_train.py

Runs poisoning experiments at multiple levels (5%, 10%, 50%) for a given poison_type.
Logs metrics and artifacts to MLflow.

Usage:
    python experiments/poison_and_train.py --poison_type label_flip
    python experiments/poison_and_train.py --poison_type feature_noise
    python experiments/poison_and_train.py --poison_type feature_outlier
"""
import argparse
import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

def poison_label_flip(X, y, frac, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    k = int(max(1, round(n * frac)))
    idx = rng.choice(n, size=k, replace=False)
    y_poison = y.copy()
    classes = np.unique(y)
    for i in idx:
        current = y_poison[i]
        other = rng.choice([c for c in classes if c != current])
        y_poison[i] = other
    return X.copy(), y_poison

def poison_feature_noise(X, y, frac, seed=42, extreme=False):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    k = int(max(1, round(n * frac)))
    idx = rng.choice(n, size=k, replace=False)
    Xp = X.copy()
    if extreme:
        for i in idx:
            Xp[i] = rng.normal(loc=100.0, scale=50.0, size=X.shape[1])
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-6
        for i in idx:
            Xp[i] = rng.normal(loc=mu, scale=sigma, size=X.shape[1])
    return Xp, y.copy()

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def run_single_experiment(poison_type, poison_frac, seed=42, artifact_dir="artifacts"):

    iris = load_iris()
    X = iris.data
    y = iris.target
    labels = iris.target_names.tolist()

    if poison_type == "label_flip":
        Xp, yp = poison_label_flip(X, y, frac=poison_frac, seed=seed)
    elif poison_type == "feature_noise":
        Xp, yp = poison_feature_noise(X, y, frac=poison_frac, seed=seed, extreme=False)
    elif poison_type == "feature_outlier":
        Xp, yp = poison_feature_noise(X, y, frac=poison_frac, seed=seed, extreme=True)
    else:
        raise ValueError("Unknown poison type")

    X_train, X_val, y_train, y_val = train_test_split(Xp, yp, test_size=0.2, random_state=seed, stratify=yp)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
    report = classification_report(y_val, y_pred, target_names=labels)

    mlflow.set_experiment("iris_poisoning_experiments")

    run_name = f"{poison_type}_{int(poison_frac*100)}pct"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("poison_type", poison_type)
        mlflow.log_param("poison_frac", poison_frac)

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_precision", prec)
        mlflow.log_metric("val_recall", rec)
        mlflow.log_metric("val_f1", f1)

        mlflow.sklearn.log_model(clf, "model")

        os.makedirs(artifact_dir, exist_ok=True)

        rpt_path = os.path.join(artifact_dir, f"report_{run_name}.txt")
        with open(rpt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(rpt_path)

        cm_path = os.path.join(artifact_dir, f"cm_{run_name}.png")
        plot_confusion(y_val, y_pred, labels, cm_path)
        mlflow.log_artifact(cm_path)

    return {
        "fraction": poison_frac,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_type", type=str, required=True,
                        choices=["label_flip", "feature_noise", "feature_outlier"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    FRACTIONS = [0.05, 0.10, 0.50]

    print(f"\n=== Running {args.poison_type} poisoning experiments ===\n")

    results = []
    for frac in FRACTIONS:
        print(f"→ Running {int(frac*100)}% poisoning...")
        res = run_single_experiment(args.poison_type, frac, seed=args.seed)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        print(f"{int(r['fraction']*100)}% → acc={r['accuracy']:.3f}, f1={r['f1']:.3f}")


if __name__ == "__main__":
    main()
