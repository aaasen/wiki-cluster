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


def cluster(k, documents, centroids=None):
    if centroids is None:
        random.seed(SEED)
        centroids = [doc.vector for doc in random.sample(documents, k)]

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
