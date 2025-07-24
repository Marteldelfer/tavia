from genetic.Cromossomo import Cromossomo
import itertools
import random


class Populacao:
    def __init__(
            self, tamanho_populacao: int,
            p_mutacao_centroide: float = 0.02,
            p_mesclar_centroide: float = 0.02,
            p_gerar_centroide: float = 0.02,
            *args, **kwargs
    ):
        self.tamanho_populacao = tamanho_populacao
        self.p_mutacao_centroide = p_mutacao_centroide
        self.p_mesclar_centroide = p_mesclar_centroide
        self.p_gerar_centroide = p_gerar_centroide
        self.cromossomos = [
            Cromossomo(*args, **kwargs) for _ in range(tamanho_populacao)
        ]

    def __iter__(self):
        return iter(self.cromossomos)

    def __len__(self):
        return len(self.cromossomos)

    def fitness(self, pontos_problema: list[tuple[float, ...]]):
        for cromossomo in self.cromossomos:
            cromossomo.fitness(pontos_problema)

    def _selecao_individual(self):
        total_fitness = sum(
            cromossomo.aptidao for cromossomo in self.cromossomos
        )
        random_value = random.random() * total_fitness

        for cromossomo in self.cromossomos:
            random_value -= cromossomo.aptidao
            if random_value <= 0:
                self.pais.append(cromossomo)
                return

    def selecao(self):
        self.pais = []
        for _ in range(self.tamanho_populacao):
            self._selecao_individual()

    def crossover(self, p_crossover: float = 0.8):
        filhos = []
        for pai_a, pai_b in itertools.batched(self.pais, 2):
            if random.random() < p_crossover:
                filhos.extend(Cromossomo.cruzamento(pai_a, pai_b))
            else:
                filhos.extend([pai_a, pai_b])
        self.cromossomos = filhos

    def mutacao(self):
        for cromossomo in self.cromossomos:
            cromossomo.mutacao(
                self,
                self.p_mutacao_centroide,
                self.p_mesclar_centroide,
                self.p_gerar_centroide
            )

    def melhor_cromossomo(self) -> Cromossomo:
        return max(self.cromossomos, key=lambda c: c.aptidao)
