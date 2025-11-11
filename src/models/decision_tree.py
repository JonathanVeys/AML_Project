import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.metrics import f1_score


# ---------------------------
# Utilities
# ---------------------------
def train_test_split(X, y, test_ratio=0.2, shuffle=True, seed=42):
    assert len(X) == len(y)
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    test_size = int(n * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def gini_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return 1.0 - np.sum(p * p)

def normalize_latlon_inplace(X: np.ndarray, lat_col: int = 0, lon_col: int = 1) -> None:
    # Normalize longitude to [-180, 180)
    X[:, lon_col] = (X[:, lon_col] + 180.0) % 360.0 - 180.0

    # Clamp latitude to valid physical range [-90, 90]
    X[:, lat_col] = np.clip(X[:, lat_col], -90.0, 90.0)

def accuracy_topk(y_true: np.ndarray, proba: np.ndarray, k: int = 5) -> float:
    """
    Top-k accuracy: fraction of samples whose true class is among
    the k highest-probability predictions.
    """
    k = min(k, proba.shape[1])
    topk = np.argsort(-proba, axis=1)[:, :k]
    return np.mean([yt in row for yt, row in zip(y_true, topk)])


# ---------------------------
# CART Decision Tree
# ---------------------------
@dataclass
class _Node:
    is_leaf: bool
    proba: np.ndarray              
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["__class__"] = None
    right: Optional["__class__"] = None


class SimpleDecisionTree:
    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 40,
        min_samples_leaf: int = 20,
        min_impurity_decrease: float = 1e-7,
        n_classes: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_classes = n_classes
        self.root: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.n_classes is None:
            self.n_classes = int(y.max()) + 1
        self.root = self._build(X, y, depth=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None, "Call fit() first."
        X = np.asarray(X)
        out = np.zeros((X.shape[0], self.n_classes), dtype=float)
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out[i] = node.proba
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _leaf_from_counts(self, counts: np.ndarray) -> _Node:
        total = counts.sum()
        if total == 0:
            proba = np.ones(self.n_classes) / self.n_classes
        else:
            proba = counts / total
        return _Node(is_leaf=True, proba=proba)

    def _best_split_for_feature(self, Xf: np.ndarray, y: np.ndarray, n_classes: int,
                                min_left: int, min_right: int) -> Tuple[float, float]:
        order = np.argsort(Xf, kind="mergesort")
        Xs = Xf[order]
        ys = y[order]

        left_counts = np.zeros(n_classes, dtype=int)
        total_counts = np.bincount(ys, minlength=n_classes)

        best_impurity = np.inf
        best_thr = None

        for i in range(len(ys) - 1):
            cls = ys[i]
            left_counts[cls] += 1

            left_n = i + 1
            right_n = len(ys) - left_n
            if left_n < min_left or right_n < min_right:
                continue

            if Xs[i] == Xs[i + 1]:
                continue

            right_counts = total_counts - left_counts

            g_left = gini_counts(left_counts)
            g_right = gini_counts(right_counts)
            impurity = (left_n * g_left + right_n * g_right) / (left_n + right_n)

            if impurity < best_impurity:
                best_impurity = impurity
                best_thr = (Xs[i] + Xs[i + 1]) / 2.0

        if best_thr is None:
            return np.inf, np.nan
        return best_impurity, best_thr

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        base_counts = np.bincount(y, minlength=self.n_classes)
        base_impurity = gini_counts(base_counts)

        best_feature = None
        best_thr = None
        best_impurity = np.inf

        min_left = self.min_samples_leaf
        min_right = self.min_samples_leaf

        for j in range(n_features):
            imp, thr = self._best_split_for_feature(
                X[:, j], y, self.n_classes, min_left=min_left, min_right=min_right
            )
            if imp < best_impurity:
                best_impurity = imp
                best_feature = j
                best_thr = thr

        impurity_decrease = base_impurity - best_impurity
        if best_feature is None or impurity_decrease < self.min_impurity_decrease:
            return None, None, base_impurity
        return best_feature, best_thr, best_impurity

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        counts = np.bincount(y, minlength=self.n_classes)
        if (
            depth >= self.max_depth
            or X.shape[0] < self.min_samples_split
            or (counts > 0).sum() == 1
        ):
            return self._leaf_from_counts(counts)

        feat, thr, child_impurity = self._find_best_split(X, y)
        if feat is None:
            return self._leaf_from_counts(counts)

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._leaf_from_counts(counts)

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        total = counts.sum()
        proba = counts / total if total > 0 else np.ones(self.n_classes) / self.n_classes
        return _Node(
            is_leaf=False,
            proba=proba,
            feature_index=feat,
            threshold=thr,
            left=left,
            right=right,
        )


# ---------------------------
# Main Loop
# ---------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent.parent
    data = np.load(ROOT / "data/species_train.npz", allow_pickle=True)

    X_data = data["train_locs"].astype(float) 

    normalize_latlon_inplace(X_data)

    taxon_ids = data["taxon_ids"]
    id_to_idx = {tid: i for i, tid in enumerate(taxon_ids)}
    y_data = np.array([id_to_idx[tid] for tid in data["train_ids"]], dtype=int)

    n_classes = len(taxon_ids)
    print(f"Samples: {len(X_data)} | Classes: {n_classes}")

    # Split
    (X_tr, y_tr), (X_te, y_te) = train_test_split(X_data, y_data, test_ratio=0.2, shuffle=True, seed=123)

    # Train
    tree = SimpleDecisionTree(
        max_depth=12,
        min_samples_split=40,
        min_samples_leaf=20,
        min_impurity_decrease=1e-7,
        n_classes=n_classes,
    ).fit(X_tr, y_tr)

    # Evaluate
    probs_te = tree.predict_proba(X_te)
    y_pred = probs_te.argmax(axis=1)

    acc_top1 = (y_pred == y_te).mean()
    acc_top5 = accuracy_topk(y_te, probs_te, k=5)
    f1_macro = f1_score(y_te, y_pred, average="macro")
    f1_weighted = f1_score(y_te, y_pred, average="weighted")

    print(f"Test accuracy (top-1): {acc_top1:.4f}")
    print(f"Test accuracy (top-5): {acc_top5:.4f}")
    print(f"F1 score (macro):      {f1_macro:.4f}")
    print(f"F1 score (weighted):   {f1_weighted:.4f}")

    

    rng = np.random.default_rng()
    i = rng.integers(0, len(X_te))
    probs = tree.predict_proba(X_te[i:i+1])[0]
    top5 = np.argsort(-probs)[:5]
    taxon_names = dict(zip(data["taxon_ids"], data["taxon_names"]))
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    print("\nRandom sample:")
    print(f"  Location (lat, lon): {X_te[i]}")
    print(f"  True class idx: {y_te[i]}  | taxon_id: {idx_to_id[y_te[i]]} | name: {taxon_names[idx_to_id[y_te[i]]]}")
    print("  Top-5 predicted:")
    for rank, cls_idx in enumerate(top5, 1):
        tax_id = idx_to_id[cls_idx]
        name = taxon_names[tax_id]
        print(f"    {rank}. idx={cls_idx:>4}  prob={probs[cls_idx]:.4f}  taxon_id={tax_id}  {name}")

    chosen_idx = int(y_tr[rng.integers(0, len(y_tr))])
    chosen_mask_tr = (y_tr == chosen_idx)
    chosen_mask_te = (y_te == chosen_idx)

    plt.figure(figsize=(7, 5))
    plt.title(f"Example species idx={chosen_idx} (train/test points)")
    plt.scatter(X_tr[~chosen_mask_tr, 1], X_tr[~chosen_mask_tr, 0], s=3, alpha=0.1, label="train others")
    plt.scatter(X_tr[chosen_mask_tr, 1], X_tr[chosen_mask_tr, 0], s=8, alpha=0.7, label="train (chosen)")
    plt.scatter(X_te[chosen_mask_te, 1], X_te[chosen_mask_te, 0], s=15, marker="x", label="test (chosen)")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
