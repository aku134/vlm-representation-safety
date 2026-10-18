"""Train the harmfulness probe on cached activations (Gray Swan recipe).

Per-token logistic regression / MLP on hidden states; an example's score is
the max probability over its token positions; the layer is chosen by held-out
AUC; the threshold is tuned so benign FPR stays under --fpr-budget.

Output: outputs/probe.joblib  {clf, layer, threshold, metrics}
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from vlm_common import ROOT

from extract_activations import CANDIDATE_LAYERS


def load_split(split):
    data = np.load(ROOT / "outputs" / f"activations_{split}.npz", allow_pickle=True)
    labels = json.loads(str(data["labels"]))
    per_layer = {l: {} for l in CANDIDATE_LAYERS}
    for key in data.files:
        if key == "labels":
            continue
        ex_id, layer_s = key.split("|layer")
        per_layer[int(layer_s)][ex_id] = data[key].astype(np.float32)
    return per_layer, labels


def token_matrix(acts: dict, labels: dict):
    """Flatten to per-token rows for training."""
    X, y = [], []
    for ex_id, arr in acts.items():
        X.append(arr)
        y.extend([1 if labels[ex_id] == "harmful" else 0] * len(arr))
    return np.concatenate(X), np.array(y)


def example_scores(clf, acts: dict):
    """Max-over-tokens probability per example (the repo's scoring rule)."""
    ids, scores = [], []
    for ex_id, arr in acts.items():
        p = clf.predict_proba(arr)[:, 1]
        ids.append(ex_id)
        scores.append(float(p.max()))
    return ids, np.array(scores)


def main(args):
    train_acts, train_labels = load_split("train")
    eval_acts, eval_labels = load_split("eval")

    results = []
    for layer in CANDIDATE_LAYERS:
        X, y = token_matrix(train_acts[layer], train_labels)
        if args.probe == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y)

        ids, scores = example_scores(clf, eval_acts[layer])
        y_eval = np.array([1 if eval_labels[i] == "harmful" else 0 for i in ids])
        auc = roc_auc_score(y_eval, scores)
        results.append((auc, layer, clf))
        print(f"layer {layer:>2}: eval AUC = {auc:.4f}")

    auc, layer, clf = max(results, key=lambda r: r[0])
    print(f"\nselected layer {layer} (AUC {auc:.4f})")

    # threshold: highest benign score quantile under the FPR budget
    ids, scores = example_scores(clf, eval_acts[layer])
    y_eval = np.array([1 if eval_labels[i] == "harmful" else 0 for i in ids])
    benign_scores = scores[y_eval == 0]
    threshold = float(np.quantile(benign_scores, 1 - args.fpr_budget)) if len(benign_scores) else 0.5
    tpr = float(((scores >= threshold) & (y_eval == 1)).sum() / max(1, (y_eval == 1).sum()))
    fpr = float(((scores >= threshold) & (y_eval == 0)).sum() / max(1, (y_eval == 0).sum()))
    print(f"threshold {threshold:.4f}: TPR {tpr:.3f}, FPR {fpr:.3f} (budget {args.fpr_budget})")

    out = ROOT / "outputs" / "probe.joblib"
    joblib.dump(
        dict(clf=clf, layer=layer, threshold=threshold,
             metrics=dict(auc=auc, tpr=tpr, fpr=fpr, probe=args.probe)),
        out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["logistic", "mlp"], default="logistic")
    ap.add_argument("--fpr-budget", type=float, default=0.01)
    args = ap.parse_args()
    main(args)
