import random
import itertools
import math
import tqdm

Chromossome = list[float]
ProblemData = list[tuple[float, ...]]


def init_population(pop_size: int, k: int, pd: ProblemData) -> list[Chromossome]:
    """
    Generate initial population of genetic algorithm, defined as lists of centroids.

    :param pop_size: Number of cromossome in population.
    :type pop_size: int

    :param k: Number of clusters.
    :type k: int

    :param pd: Data from the problem to be solved, represented as list of cordinates of points
    :type pd: ProblemData

    :returns: Initial population of genetic algorithm.
    :rtype: list[Cromossomes]
    """
    res = []
    for _ in range(pop_size):
        chromossome = list(itertools.chain.from_iterable(
            random.sample(pd, k)))  # pontos aleatórios no dataset
        res.append(chromossome)
    return res


def get_cluster(point: tuple[float, ...], chromossome: Chromossome) -> int:
    min_distance, best_index = float('inf'), -1
    for index, centroid in enumerate(itertools.batched(chromossome, len(point))):
        d = distance(point, centroid)
        if d < min_distance:
            min_distance, best_index = d, index
    return best_index


def clusterize(pd: ProblemData, chromossome: Chromossome) -> list[int]:
    return [get_cluster(p, chromossome) for p in pd]


def get_center(cluster: list[tuple[float, ...]]) -> tuple[float, ...]:
    """
    :param cluster: lists of points inside cluster
    :type cluster: list[tuple[float, ...]]
    """
    if len(cluster) == 0:
        return None

    point_size = len(cluster[0])
    center = [0 for _ in range(point_size)]

    for point in cluster:
        for index, coordinate in enumerate(point):
            center[index] += coordinate
    for index in range(point_size):
        center[index] = center[index] / len(cluster)
    return tuple(center)


def get_centers_chromossome(chromossome: Chromossome, pd: ProblemData) -> Chromossome:
    """

    """
    clusters = []
    point_size = len(pd[0])
    clusterized = clusterize(pd, chromossome)
    centroids = []
    for index, centroid in enumerate(itertools.batched(chromossome, point_size)):
        cluster = [p for p, c in zip(pd, clusterized) if c == index]
        clusters.append(cluster)
        centroids.append(centroid)

    res = []
    for i, c in enumerate(clusters):
        center = get_center(c)
        if center == None:
            res.extend(centroids[i])
        else:
            res.extend(center)
    return res


def get_centers_population(population: list[Chromossome], pd: ProblemData) -> list[Chromossome]:
    return [get_centers_chromossome(c, pd) for c in population]


def individual_fitness(chromossome: Chromossome, pd: ProblemData) -> tuple[Chromossome, float]:
    point_size = len(pd[0])
    clusterized = clusterize(pd, chromossome)
    total_distance = 0

    for index, centroid in enumerate(itertools.batched(chromossome, point_size)):
        cluster_points = [p for p, c in zip(pd, clusterized) if c == index]
        for point in cluster_points:
            total_distance += distance(point, centroid)

    return chromossome, 1 / total_distance


def fitness(population: list[Chromossome], pd: ProblemData) -> list[tuple[Chromossome, float]]:
    """

    """
    new_centers = get_centers_population(population, pd)
    return [individual_fitness(c, pd) for c in new_centers]


def individual_selection(pop_fitness: tuple[Chromossome, float], total_fitness: float) -> Chromossome:
    """

    """
    rdn_seletion = total_fitness * random.random()
    for chromossome, ind_fitness in pop_fitness:
        rdn_seletion -= ind_fitness
        if rdn_seletion <= 0:
            return chromossome


def selection(pop_fitness: tuple[Chromossome, float]) -> list[Chromossome]:
    """

    """
    total_fitness = sum(p[1] for p in pop_fitness)
    return [individual_selection(pop_fitness, total_fitness) for _ in range(len(pop_fitness))]


def individual_crossover(parent_a: Chromossome, parent_b: Chromossome, prob: float = .8) -> tuple[Chromossome]:
    """
    Single point crossover with pre-determined probability
    """
    if random.random() > prob:
        return [parent_a, parent_b]

    crossover_point = random.randrange(len(parent_a))
    child_a = parent_a[:crossover_point] + parent_b[crossover_point:]
    child_b = parent_b[:crossover_point] + parent_a[crossover_point:]

    return child_a, child_b


def crossover(parents: list[Chromossome], prob: float = 0.8) -> list[Chromossome]:
    res = []
    for a, b in itertools.batched(parents, 2):
        res.extend(individual_crossover(a, b, prob))
    return res


def individual_mutation(chromossome: Chromossome, prob: float = .001) -> Chromossome:
    """
    Mutation
    """
    if random.random() > prob:
        return chromossome

    modifier = 2 * random.random()

    gene_index = random.randrange(len(chromossome))

    if chromossome[gene_index] != 0:
        modifier *= chromossome[gene_index]

    if random.random() > .5:
        chromossome[gene_index] += modifier
    else:
        chromossome[gene_index] -= modifier

    return chromossome


def mutation(children: list[Chromossome], prob: float = 0.001) -> list[Chromossome]:
    return [individual_mutation(c, prob) for c in children]


def distance(point: list[float], centroid: list[float]) -> float:
    """Euclidian distance bentween point and centroid"""
    return math.sqrt(sum((p - c)**2 for p, c in zip(point, centroid)))


def genetic(pd: ProblemData, k: int, pop_size: int = 100, n_generations: int = 100) -> Chromossome:
    """

    """
    population = init_population(pop_size, k, pd)

    with tqdm.trange(n_generations) as t:
        for gen in t:
            pop_fitness = fitness(population, pd)

            best_found = min(pop_fitness, key=lambda p: p[1])
            parents = selection(pop_fitness)
            children = crossover(parents)
            population = mutation(children)
            t.set_description("best found: " + f"{best_found[1]:.10f}")
    return best_found


def main():
    import matplotlib.pyplot as plt
    test_data = [((50 + random.gauss(4, 15)), -50 + random.gauss(4, 15))
                 for _ in range(200)]
    test_data.extend([
        (-50 + (random.gauss(4, 15)), -50 + random.gauss(4, 15))
        for _ in range(200)
    ])
    test_data.extend([
        (50 + (random.gauss(4, 15)), 50 + random.gauss(4, 15))
        for _ in range(200)
    ])
    test_data.extend([
        (-50 + (random.gauss(4, 15)), 50 + random.gauss(4, 15))
        for _ in range(200)
    ])

    pop = [[random.randint(1, 3), random.randint(1, 3), random.randint(5, 7), random.randint(
        5, 7), random.randint(8, 10), random.randint(8, 10)] for _ in range(10)]
    pd = [[random.randint(10, 100), random.randint(10, 100)]
          for _ in range(100)]

    colors = ['r', 'b', 'y', 'g']
    res = genetic(test_data, 4, 100)
    print(res)
    clusters = clusterize(test_data, res[0])
    plt.scatter([t[0] for t in test_data], [t[1] for t in test_data], c=[
                colors[i] for i in clusters])
    plt.show()


if __name__ == "__main__":
    main()
