import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


class DBHC:

    column_cluster = "cluster_dbhc"

    def __init__(self,
                 n_clusters: int,
                 min_pts: int = 3):
        self.n_clusters = n_clusters
        self.min_pts = min_pts

    def fit_predict(self, X):
        """Compute dbhc method and predict cluster index for each sample.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        labels : pandas.Series of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.X = X.copy()
        self.n_points = X.shape[0]
        self.features_list = list(X.columns)
        self._step_1()
        self._step_2()
        self._step_3()
        self._create_final_labels()
        return self.labels_

    def _step_1(self):
        k = self.min_pts
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(self.X)
        distances, _ = neighbors_fit.kneighbors(self.X)
        distances = distances[:, k - 1]
        distances = sorted(distances)
        j = int(np.sqrt(self.n_points))
        distances_indexes = list(range(1, j + 1) * np.sqrt(self.n_points))
        distances_indexes.append(self.n_points)
        eps_list = [distances[int(round(i, 0)) - 1] for i in distances_indexes]
        eps_list = sorted(list(set(eps_list)))
        eps_list = [i*1.00000000001 for i in eps_list if i > 0]
        self.eps_list = eps_list

    def _step_2(self):
        initial_cluster = []
        max_cluster = 0
        for i, eps in enumerate(self.eps_list):
            dbscan = DBSCAN(eps=eps, min_samples=self.min_pts)
            dbscan.fit(self.X[self.features_list])
            self.X[self.column_cluster] = dbscan.labels_
            _x_clustered = self.X[self.X[self.column_cluster] != -1]
            _x_clustered[self.column_cluster] = _x_clustered[self.column_cluster] + max_cluster
            if _x_clustered.shape[0] > 0:
                max_cluster = _x_clustered[self.column_cluster].max() + 1
            initial_cluster.append(_x_clustered)
            self.X = self.X[self.X[self.column_cluster] == -1]
            if self.X.shape[0] == 0:
                break
        initial_cluster.append(self.X)
        initial_cluster = pd.concat(initial_cluster, ignore_index=False)
        self.clusters = initial_cluster

    def _step_3(self):
        df_cluster = self.clusters[self.clusters[self.column_cluster] != -1]
        df_cluster["n_rows"] = 1
        df_cluster = df_cluster.groupby(self.column_cluster).sum()
        df_cluster[self.features_list] = df_cluster[self.features_list].div(df_cluster["n_rows"], axis=0)
        for i in range(df_cluster.shape[0] - self.n_clusters):
            _distances = pairwise_distances(df_cluster[self.features_list])
            _max_distance = _distances.max()
            np.fill_diagonal(_distances, _max_distance + 1)
            _nearest_clusters = list(np.unravel_index(_distances.argmin(), _distances.shape))
            _clusters_to_aggregate = df_cluster.iloc[list(_nearest_clusters)]
            _clusters_to_aggregate[self.features_list] = _clusters_to_aggregate[self.features_list].multiply(
                _clusters_to_aggregate["n_rows"], axis=0)
            _clusters_to_aggregate = _clusters_to_aggregate.sum()
            _clusters_to_aggregate[self.features_list] = _clusters_to_aggregate[self.features_list].div(
                _clusters_to_aggregate["n_rows"], axis=0)
            new_cluster = df_cluster.index[_nearest_clusters[0]]
            old_cluster = df_cluster.index[_nearest_clusters[1]]
            df_cluster.drop([new_cluster, old_cluster], inplace=True)
            df_cluster.loc[new_cluster] = _clusters_to_aggregate
            self.clusters[self.column_cluster] = self.clusters[self.column_cluster].replace(old_cluster, new_cluster)

    def _create_final_labels(self):
        check_noise = (self.clusters[self.column_cluster] == -1).any()
        self.clusters[self.column_cluster] = self.clusters[self.column_cluster].rank(method="dense").astype("int64")-1
        if check_noise:
            self.clusters[self.column_cluster] = self.clusters[self.column_cluster]-1
        self.labels_ = self.clusters[self.column_cluster]
