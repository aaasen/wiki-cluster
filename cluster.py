
import numpy as np
from matplotlib import pyplot
from sklearn.manifold import MDS

import csv
import random
import pickle
from collections import namedtuple


# This seed is used for all random number generators
# so that results are reproducible.
SEED = 0


Document = namedtuple('Document', 'label, vector')


def distance(a, b):
    return np.linalg.norm(np.subtract(a, b))


class Cluster(namedtuple('Cluster', 'centroid, documents')):
    def top_words(self, dictionary):
        indexes = sorted(range(len(self.centroid)),
                         key=lambda i: -self.centroid[i])
        return [dictionary[i] for i in indexes]

    def distance_to_center(self, document):
        return distance(document.vector, self.centroid)

    def top_documents(self):
        def distance_to_center(document):
            return distance(document.vector, self.centroid)

        return sorted(self.documents, key=distance_to_center)

    def distortion(self):
        return sum(self.distance_to_center(doc) for doc in self.documents)

    def calculate_centroid(self):
        vectors = [document.vector for document in self.documents]
        return np.average(vectors, axis=0)


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


def distortion(clusters):
    return sum(cluster.distortion() for cluster in clusters)


def get_clusters(documents, centroids):
    clusters = [Cluster(centroid, []) for centroid in centroids]

    for doc in documents:
        distances = [distance(doc.vector, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].documents.append(doc)

    return clusters


def get_centroids(clusters):
    return np.array([cluster.calculate_centroid() for cluster in clusters])


def k_means(documents, n, centroids=None):
    if centroids is None:
        random.seed(SEED)
        centroids = [doc.vector for doc in random.sample(documents, n)]

    print(n)
    while True:
        prev_centroids = centroids
        clusters = get_clusters(documents, centroids)
        centroids = get_centroids(clusters)
        print('.', end='', flush=True)
        if np.array_equal(centroids, prev_centroids):
            print()
            return clusters


def load_k_means(document_path, dictionary_path, n):
    path = '{}_{}.pickle'.format(document_path, n)
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


def distortions(document_path, dictionary_path, ns):
    return [distortion(load_k_means(document_path, dictionary_path, n))
            for n in ns]


def similarity_matrix(vectors):
    return [[distance(vectors[i], vectors[j])
             for i in range(len(vectors))]
            for j in range(len(vectors))]


def a():
    dictionary = load_dictionary('dictionary')
    clusters = load_k_means('train', 'dictionary', 16)

    for cluster in clusters:
        print('cluster')
        print(len(cluster.documents))
        print(cluster.top_words(dictionary)[:5])
        print([document.label for document in cluster.top_documents()[:5]])
        print('\n')


def b():
    x = [2 ** n for n in range(10)]
    y = distortions('train', 'dictionary', x)
    pyplot.plot(x, y)
    pyplot.xscale('log')
    pyplot.show()


def mds():
    dictionary = load_dictionary('dictionary')
    documents = load_documents('train', len(dictionary))
    random.seed(SEED)
    sample = np.array([doc.vector for doc in random.sample(documents, 1000)])
    mds = MDS(n_components=2, random_state=SEED)
    points = mds.fit_transform(sample)
    pyplot.plot(points[:, 0], points[:, 1], linestyle='', marker='o')
    pyplot.show()


c()

# centroids = k_means(tf_idf_matrix, 20)
# np.save('centroids', centroids)
# centroids = np.load('centroids.npy')
# clusters = cluster(tf_idf, centroids)
# print(clusters)

# clusters = load_dicts('cluster0.txt')
# np.set_printoptions(threshold=np.nan)
# print(tf_idf[0][1])
