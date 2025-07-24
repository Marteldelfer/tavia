from genetic.Populacao import Populacao
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm


def genetico(n_geracoes: int, pontos_problema: list[tuple[float, ...]], tamanho_populacao: int = 100, min_k: int = 2, max_k: int = 20, *args, **kwargs):
    X = StandardScaler().fit_transform(pontos_problema)
    X = (list(map(tuple, X.tolist())))

    print(X)
    pop = Populacao(
        tamanho_populacao=tamanho_populacao,
        min_k=min_k,
        max_k=max_k,
        pontos_problema=X,
        *args, **kwargs
    )

    for _ in tqdm.trange(n_geracoes):
        pop.fitness(X)
        pop.selecao()
        pop.crossover()

    pop.fitness(X)
    melhor = pop.melhor_cromossomo()
    return melhor


if __name__ == "__main__":

    """ X, _ = make_blobs(
        n_samples=200, centers=4, cluster_std=1
    )
    genetico(75, X, 80, 2, 20).plot_centroides(X) """

    X, _ = load_iris(return_X_y=True)
    melhor = genetico(75, X, 80, 2, 20)

    X = StandardScaler().fit_transform(X)
    X = (list(map(tuple, X.tolist())))

    clustered_points = melhor.clusterizar(X)

    clustered_plotable = []
    for pontos in clustered_points:
        cluster = []
        for ponto in pontos:
            # Considerando apenas as duas primeiras dimensões para plotar
            cluster.append(tuple(ponto[0:4:2]))
        clustered_plotable.append(cluster)

    colors = plt.cm.tab10.colors

    for i, (centroide, pontos) in enumerate(zip(melhor, clustered_plotable)):
        color = colors[i % len(colors)]
        if pontos:
            plt.scatter(
                *zip(*pontos), label=f'Cluster {i}', color=color
            )
    plt.legend()
    plt.show()
