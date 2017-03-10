import numpy as np

import random

from distance import distance
from cluster import Cluster
from config import SEED


def _get_clusters(documents, centroids):
    clusters = [Cluster(centroid, []) for centroid in centroids]

    for doc in documents:
        distances = [distance(doc.vector, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].documents.append(doc)

    return clusters


def _get_centroids(clusters):
    return np.array([cluster.calculate_centroid() for cluster in clusters])


def _normalize(array):
    return np.array(array) / sum(array)


def _sample(array):
    array = _normalize(array)
    array = array.cumsum()
    n = random.random()
    for i in range(len(array)):
        if array[i] >= n:
            return i


def k_means_plus_plus(k, documents, seed=SEED):
    """
    K-means++ is a method of initializing cluster centroids.
    The algorithm works as follows (Wikipedia):
     1. Choose one center uniformly at random from among the data points.
     2. For each data point x, compute D(x), the distance between x and
        the nearest center that has already been chosen.
     3. Choose one new data point at random as a new center, using a weighted
        probability distribution where a point x is chosen with probability
        proportional to D(x)2.
     4. Repeat Steps 2 and 3 until k centers have been chosen.
    """

    random.seed(seed)
    init_centroid_index = random.randint(0, len(documents) - 1)
    centroid_indexes = {init_centroid_index}
    centroids = [documents[init_centroid_index].vector]

    while len(centroids) < k:
        clusters = _get_clusters(documents, centroids)
        distances = [(doc.vector, cluster.distance_to_center(doc) ** 2)
                     for cluster in clusters
                     for doc in cluster.documents]
        centroid_index = _sample([distance[1] for distance in distances])
        if centroid_index not in centroid_indexes:
            centroid_indexes.add(centroid_index)
            centroids.append(distances[centroid_index][0])

    return centroids


def random_documents(k, documents, seed=SEED):
    random.seed(seed)
    return [doc.vector for doc in random.sample(documents, k)]


def cluster(k, documents, centroids=None, init=k_means_plus_plus, seed=None):
    if seed is None:
        seed = SEED

    if centroids is None:
        centroids = init(k, documents, seed)

    print(k)
    while True:
        prev_centroids = centroids
        clusters = _get_clusters(documents, centroids)
        centroids = _get_centroids(clusters)
        print('.', end='', flush=True)
        if np.array_equal(centroids, prev_centroids):
            print()
            return clusters


def similarity_matrix(vectors):
    return [[distance(vectors[i], vectors[j])
             for i in range(len(vectors))]
            for j in range(len(vectors))]


def print_clusters():
    dictionary = load_dictionary()

    for cluster in load_k_means(16):
        print('cluster')
        print(len(cluster.documents))
        print(cluster.top_words(dictionary)[:5])
        print([document.label for document in cluster.top_documents()[:5]])
        print('\n')
