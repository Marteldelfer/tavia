import random
import itertools
import matplotlib.pyplot as plt


class Cromossomo:

    def __init__(self, pontos_problema: list[tuple[float, ...]], min_k: int = 1, max_k: int = 10):

        if max_k > len(pontos_problema):
            raise ValueError(
                "Número máximo de centróides não pode ser maior que o número de pontos no problema.")

        self.min_k = min_k
        self.max_k = max_k
        self.dimensions = len(pontos_problema[0])
        self.centroides = []

        k = random.randint(self.min_k, self.max_k)
        centroides = random.sample(pontos_problema, k)
        for centroid in centroides:
            self.centroides.extend(centroid)

    @classmethod
    def from_centroides(cls, centroides: list[float], dimensions: int, min_k: int = 1, max_k: int = 10):
        cromossomo = cls([[] for _ in range(max_k)], min_k, max_k)
        cromossomo.dimensions = dimensions
        cromossomo.centroides = centroides
        return cromossomo

    def __str__(self):
        return f"Cromossomo(dimenssões={self.dimensions}, k={len(self)}, centroides={self.centroides})"

    def __iter__(self):
        return iter(itertools.batched(self.centroides, self.dimensions))

    def __len__(self):
        return len(self.centroides) // self.dimensions

    def _mesclar_centroide(self):
        if not self.min_k < len(self):
            return

        # Selecionar dois centróides aleatórios
        indices = random.sample(range(len(self)), 2)
        novos_centroides = []
        centroides_mesclados = [0 for _ in range(self.dimensions)]

        for i, centroid in enumerate(self):
            if i not in indices:
                novos_centroides.extend(centroid)
            else:
                centroides_mesclados = [
                    x + y / 2 for x, y in zip(centroides_mesclados, centroid)
                ]
        # Adicionar o novo centróide mesclado
        novos_centroides.extend(centroides_mesclados)
        self.centroides = novos_centroides

    def _gerar_centroide(self):
        if not self.max_k > len(self):
            return

        # Selecionar um centróide aleatório
        indice = random.randrange(0, len(self))
        novo_centroide = self.centroides[
            indice * self.dimensions: indice * self.dimensions + self.dimensions
        ]
        novo_centroide = [
            x + random.normalvariate(0, x / 2) for x in novo_centroide
        ]
        self.centroides.extend(novo_centroide)

    def _mutar_centroide(self):
        indice_centroide = random.randrange(0, len(self))
        centroide_modificado = self.centroides[
            indice_centroide * self.dimensions: indice_centroide * self.dimensions + self.dimensions
        ]
        # Modificar o centróide com uma perturbação aleatória
        centroide_modificado = [
            x + random.normalvariate(0, x / 4) for x in centroide_modificado
        ]
        self.centroides[
            indice_centroide * self.dimensions: indice_centroide * self.dimensions + self.dimensions
        ] = centroide_modificado

    def mutacao(
            self,
            p_mutacao_centroide: float = 0.02,
            p_mesclar_centroide: float = 0.02,
            p_gerar_centroide: float = 0.02
    ):
        if random.random() < p_mutacao_centroide:
            self._mutar_centroide()
        if random.random() < p_mesclar_centroide:
            self._mesclar_centroide()
        if random.random() < p_gerar_centroide:
            self._gerar_centroide()

    @classmethod
    def cruzamento(cls, pai_a: 'Cromossomo', pai_b: 'Cromossomo') -> tuple['Cromossomo', 'Cromossomo']:
        indice_pai_a = random.randrange(0, len(pai_a))
        indice_pai_b = random.randrange(0, len(pai_b))

        filho_a = pai_a.centroides[:indice_pai_a * pai_a.dimensions] + \
            pai_b.centroides[indice_pai_b * pai_b.dimensions:]
        filho_b = pai_b.centroides[:indice_pai_b * pai_b.dimensions] + \
            pai_a.centroides[indice_pai_a * pai_a.dimensions:]

        # Correção de tamanho dos filhos
        while len(filho_a) > pai_a.dimensions * pai_a.max_k:
            filho_a.pop(random.randrange(len(filho_a)))
        while len(filho_b) > pai_b.dimensions * pai_b.max_k:
            filho_b.pop(random.randrange(len(filho_b)))

        while len(filho_a) < pai_a.dimensions * pai_a.min_k:
            filho_a.extend([random.uniform(-1, 1)
                           for _ in range(pai_a.dimensions)])
        while len(filho_b) < pai_b.dimensions * pai_b.min_k:
            filho_b.extend([random.uniform(-1, 1)
                           for _ in range(pai_b.dimensions)])

        return (
            cls.from_centroides(
                filho_a, pai_a.dimensions, pai_a.min_k, pai_a.max_k
            ), cls.from_centroides(
                filho_b, pai_b.dimensions, pai_b.min_k, pai_b.max_k
            )
        )

    @staticmethod
    def distance(centroid: tuple[float, ...], point: tuple[float, ...]) -> float:
        return sum((c - p) ** 2 for c, p in zip(centroid, point)) ** 0.5

    def fitness(self, pontos_problema: list[tuple[float, ...]]):
        total_distance = 0.0
        for point in pontos_problema:
            min_distancia = float('inf')
            for centroid in self:
                distance = Cromossomo.distance(point, centroid)
                if distance < min_distancia:
                    min_distancia = distance
            total_distance += min_distancia
        self.aptidao = 1 / (1 + total_distance) / (len(self) ** .5)

    def plot_centroides(self, pontos_problema: list[tuple[float, ...]]):
        pontos_cluster = [[] for _ in range(len(self))]

        for point in pontos_problema:
            min_distancia = float('inf')
            min_centroide = None
            for i, centroide in enumerate(self):

                distance = Cromossomo.distance(centroide, point)
                if distance < min_distancia:
                    min_distancia = distance
                    min_centroide = i
            pontos_cluster[min_centroide].append(point)

        colors = plt.cm.tab10.colors

        for i, (centroide, pontos) in enumerate(zip(self, pontos_cluster)):
            color = colors[i % len(colors)]
            if pontos:
                plt.scatter(
                    *zip(*pontos), label=f'Cluster {i}', color=color
                )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    points = [
        (1.0, 2.0), (3.0, 2.0), (1.0, 3.0),
        (5.0, 6.0), (4.0, 6.0), (7.0, 5.0),
        (13.0, 14.0), (14.0, 14.0), (14.0, 13.0)
    ]
    cromossomo = Cromossomo(points, min_k=3, max_k=6)
    cromossomo.plot_centroides(points)
