import numpy as np
from nearest_neighbors import KNNClassifier

np.int = int


def kfold(n, n_folds):
    folds = []
    len1 = n // n_folds + 1
    len2 = n // n_folds
    for i in range(n % n_folds):
        folds.append(np.arange(i * len1, (i + 1) * len1))
    shift = (n // n_folds + 1) * (n % n_folds)
    for i in range(n_folds - n % n_folds):
        folds.append(np.arange(shift + i * len2, shift + (i + 1) * len2))
    res = []
    for i in range(n_folds):
        val = folds[i]
        test = np.concatenate(folds[:i] + folds[i + 1:])
        test = test.astype(int)
        res.append((test, val))
    return res


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if cv is None:
        cv = kfold(y.size, 5)
    if score == "accuracy":
        res = dict()
        for k in k_list:
            res[k] = np.array([])
            for (train_ind, val_ind) in cv:
                classifier = KNNClassifier(k=k, **kwargs)
                classifier.fit(X[train_ind], y[train_ind])
                n_correct = np.sum(classifier.predict(X[val_ind]) == y[val_ind])
                n_all = val_ind.size
                res[k] = np.append(res[k], float(n_correct) / n_all)
        return res
