
import csv
import numpy as np
import random
import pickle
from collections import namedtuple


Cluster = namedtuple('Cluster', 'centroid, documents')
Document = namedtuple('Document', 'label, vector')


def load_dictionary(path):
    with open(path) as f:
        return [row[0] for row in csv.reader(f, delimiter=' ')]


def load_dicts(path):
    "{title}|{key}:{value},{key}:{value}..."

    def pair(row):
        title, pairs_text = row.split('|', maxsplit=1)
        pairs = [pair.split(':') for pair in pairs_text.split(',')]
        return (title, pairs)

    with open(path) as f:
        return [pair(row) for row in f.readlines()]


def load_documents(path, n_words):
    def tf_idf_vector(tf_idf_pairs):
        tf_idf_vector = np.zeros(n_words)
        for word_index, tf_idf in tf_idf_pairs:
            tf_idf_vector[int(word_index)] = tf_idf
        return tf_idf_vector

    return [Document(title, tf_idf_vector(pairs))
            for title, pairs in load_dicts(path)]


def distance(a, b):
    return np.linalg.norm(np.subtract(a, b))


def get_clusters(documents, centroids):
    clusters = [Cluster(centroid, []) for centroid in centroids]

    for doc in documents:
        distances = [distance(doc.vector, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].documents.append(doc)

    return clusters


def get_centroids(clusters):
    def get_centroid(cluster):
        vectors = [document.vector for document in cluster.documents]
        return np.average(vectors, axis=0)

    return np.array([get_centroid(cluster) for cluster in clusters])


def k_means(documents, n, centroids=None):
    if centroids is None:
        random.seed(0)
        centroids = [doc.vector for doc in random.sample(documents, n)]

    while True:
        prev_centroids = centroids
        clusters = get_clusters(documents, centroids)
        centroids = get_centroids(clusters)
        if np.array_equal(centroids, prev_centroids):
            return clusters


def load_k_means(title, document_path, dictionary_path, n):
    path = '{}_{}.pickle'.format(title, n)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        dictionary = load_dictionary(dictionary_path)
        documents = load_documents(document_path, len(dictionary))
        clusters = k_means(documents, n)
        with open(path, 'wb') as f:
            pickle.dump(clusters, f)
        return clusters


def distortion(clusters):
    return sum(sum(np.linalg.norm(np.subtract(doc.vector, cluster.centroid))
                   for doc in cluster.documents)
               for cluster in clusters)


clusters = load_k_means('test_cluster', '20_docs.txt', 'dictionary.txt', 3)
print(clusters)
print(distortion(clusters))


# centroids = k_means(tf_idf_matrix, 20)
# np.save('centroids', centroids)
# centroids = np.load('centroids.npy')
# clusters = cluster(tf_idf, centroids)
# print(clusters)

# clusters = load_dicts('cluster0.txt')
# np.set_printoptions(threshold=np.nan)
# print(tf_idf[0][1])
