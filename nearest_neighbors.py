import numpy as np
from distances import euclidean_distance, cosine_distance
from sklearn.neighbors import NearestNeighbors

eps = 10 ** (-5)

np.int = int


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.test_block_size = test_block_size
        self.weights = weights
        self.strategy = strategy
        self.k = k
        if strategy == "my_own":
            if (metric == 'euclidean'):
                self.distance = euclidean_distance
            else:
                self.distance = cosine_distance
            self.metric = metric
        elif strategy == "brute":
            self.nn = NearestNeighbors(n_neighbors=k, algorithm=strategy, metric=metric)
        else:
            self.nn = NearestNeighbors(
                n_neighbors=k, algorithm=strategy, metric="euclidean"
            )

    def fit(self, X, y):
        self.y = y
        if self.strategy == "my_own":
            self.X = X
        else:
            self.nn.fit(X)

    def find_kneighbors(self, X, return_distance):
        if self.strategy == "my_own":
            if return_distance:
                resn = np.array([])
                resn.shape = (0, self.k)
                resd = np.array([])
                resd.shape = (0, self.k)
                i = 0
                while i < X.shape[0]:
                    dists = self.distance(X[i: i + self.test_block_size], self.X)
                    n_neighbors = np.argsort(dists, axis=1)[:, : self.k]
                    n_dists = np.take_along_axis(dists, n_neighbors, axis=1)
                    resd = np.append(resd, n_dists, axis=0)
                    resn = np.append(resn, n_neighbors, axis=0)
                    i += self.test_block_size
                resn = resn.astype(int)
                return (resd, resn)
            else:
                res = np.array([])
                res.shape = (0, self.k)
                i = 0
                while i < X.shape[0]:
                    dists = self.distance(X[i: i + self.test_block_size], self.X)
                    n_neighbors = np.argsort(dists, axis=1)[:, : self.k]
                    res = np.append(res, n_neighbors, axis=0)
                    i += self.test_block_size
                res = res.astype(int)
                return res
        else:
            if return_distance:
                resn = np.array([])
                resn.shape = (0, self.k)
                resd = np.array([])
                resd.shape = (0, self.k)
                i = 0
                while i < X.shape[0]:
                    res1 = self.nn.kneighbors(
                        X[i: i + self.test_block_size], return_distance=return_distance
                    )
                    resd = np.append(resd, res1[0], axis=0)
                    resn = np.append(resn, res1[1], axis=0)
                    i += self.test_block_size
                resn = resn.astype(int)
                return (resd, resn)
            else:
                res = np.array([])
                res.shape = (0, self.k)
                i = 0
                while i < X.shape[0]:
                    res1 = self.nn.kneighbors(
                        X[i: i + self.test_block_size], return_distance=return_distance
                    )
                    res = np.append(res, res1, axis=0)
                    i += self.test_block_size
                res = res.astype(int)
                return res

    def pred_(self, i):
        return (self.y)[i]

    def find_max_args(self, x, weights=None):
        if weights is not None:
            values, inv_ind = np.unique(x, return_inverse=True)
            sum_weights = np.bincount(inv_ind, weights=weights)
            max_sum = np.argmax(sum_weights[:len(values)])
            return values[max_sum]
        else:
            values, index, counts = np.unique(x, return_index=True, return_counts=True)
            values = values[np.argsort(index)]
            counts = counts[np.argsort(index)]
            return values[np.argmax(counts)]

    def predict(self, X):
        if self.weights:
            (dists, neighbors) = self.find_kneighbors(X, return_distance=True)
            pred = np.vectorize(self.pred_)
            neighbors = pred(neighbors)
            dists = 1 / (dists + eps)
            res = np.zeros(neighbors.shape[0])
            for i in range(neighbors.shape[0]):
                res[i] = self.find_max_args(neighbors[i], weights=dists[i])
            return res
        else:
            neighbors = self.find_kneighbors(X, return_distance=False)
            pred = np.vectorize(self.pred_)
            neighbors = pred(neighbors)
            res = np.zeros(neighbors.shape[0])
            for i in range(neighbors.shape[0]):
                res[i] = self.find_max_args(neighbors[i])
            return res
