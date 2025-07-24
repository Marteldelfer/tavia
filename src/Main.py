from genetic.Populacao import Populacao
import tqdm
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    X, _ = make_blobs(
        n_samples=100, centers=4, cluster_std=1
    )

    blobs = StandardScaler().fit_transform(X)
    blobs = (list(map(tuple, blobs.tolist())))

    pop = Populacao(
        tamanho_populacao=100, min_k=2, max_k=20, pontos_problema=blobs
    )
    for _ in tqdm.trange(100):
        pop.fitness(blobs)
        pop.selecao()
        pop.crossover()
    pop.fitness(blobs)
    melhor = pop.melhor_cromossomo()
    print(melhor)

    """ kmeans = KMeans(n_clusters=7, random_state=0)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=200, c='black', label='Centroids')
    plt.legend()
    plt.title('KMeans Clustering')
    plt.show() """
