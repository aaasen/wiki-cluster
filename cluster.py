
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

import csv
import random
import pickle
from collections import namedtuple


# This seed is used for all random number generators
# so that results are reproducible.
SEED = 0
random.seed(SEED)

DICTIONARY_PATH = 'dictionary'
DOCUMENT_PATH = 'train'


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

    def __len__(self):
        return len(self.documents)


def load_dictionary(path=DICTIONARY_PATH):
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


def load_documents(path=DOCUMENT_PATH, dictionary_path=DICTIONARY_PATH):
    # Could speed this up since we only need to know how many words exist.
    n_words = len(load_dictionary(dictionary_path))

    def tf_idf_vector(tf_idf_pairs):
        tf_idf_vector = np.zeros(n_words)
        for word_index, tf_idf in tf_idf_pairs:
            tf_idf_vector[int(word_index)] = tf_idf
        return tf_idf_vector

    return [Document(title, tf_idf_vector(pairs))
            for title, pairs in load_dicts(path)]


def document_matrix(documents):
    return [document.vector for document in documents]


def load_document_matrix(path=DOCUMENT_PATH, dictionary_path=DICTIONARY_PATH):
    return document_matrix(load_documents(path, dictionary_path))


def get_clusters(documents, centroids):
    clusters = [Cluster(centroid, []) for centroid in centroids]

    for doc in documents:
        distances = [distance(doc.vector, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].documents.append(doc)

    return clusters


def get_centroids(clusters):
    return np.array([cluster.calculate_centroid() for cluster in clusters])


def k_means(k, documents, centroids=None):
    if centroids is None:
        random.seed(SEED)
        centroids = [doc.vector for doc in random.sample(documents, k)]

    print(k)
    while True:
        prev_centroids = centroids
        clusters = get_clusters(documents, centroids)
        centroids = get_centroids(clusters)
        print('.', end='', flush=True)
        if np.array_equal(centroids, prev_centroids):
            print()
            return clusters


def load_k_means(k, document_path=DOCUMENT_PATH):
    path = '{}_{}.pickle'.format(document_path, k)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        documents = load_documents(document_path)
        clusters = k_means(k, documents)
        with open(path, 'wb') as f:
            pickle.dump(clusters, f)
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


def plot_k_vs_distortion():
    def distortion(clusters):
        return sum(cluster.distortion() for cluster in clusters)

    xs = [2 ** n for n in range(9)]
    ys = [distortion(load_k_means(k)) for k in xs]
    plt.plot(xs, ys)

    plt.title('K vs. Distortion')
    plt.xlabel('K')
    plt.ylabel('Total Distortion')
    plt.tight_layout()
    plt.savefig('k_vs_distortion.png')
    plt.show()


def plot_k_vs_cluster_size():
    ks = [2 ** n for n in range(9)]
    lens = [[len(cluster) for cluster in load_k_means(k)] for k in ks]
    print(lens[4])

    def plot_stat(func):
        plt.plot(ks, list(map(getattr(np, func), lens)), label=func)

    plot_stat('min')
    plot_stat('median')
    plot_stat('max')

    plt.legend()
    plt.title('K vs. Cluster Size')
    plt.xlabel('K')
    plt.ylabel('Cluster Size')
    plt.yscale('log', basey=2)
    plt.xscale('log', basex=2)
    plt.tight_layout()
    plt.savefig('k_vs_cluster_size.png')
    plt.show()


def mds(path=DOCUMENT_PATH, k=10):
    clusters = load_k_means(k, path)
    documents = [doc.vector for cluster in clusters
                 for doc in cluster.documents]
    labels = [i for i in range(len(clusters))
              for j in range(len(clusters[i].documents))]
    mds = MDS(n_components=2, random_state=SEED)
    points = mds.fit_transform(documents)
    print(labels)
    pyplot.scatter(points[:, 0], points[:, 1], c=labels)
    # pyplot.legend([str(i) for i in range(len(clusters))])
    pyplot.show()


def lda():
    lda = LDA(n_topics=10)
    document_matrix()

# mds('train1000')


plot_k_vs_distortion()

# centroids = k_means(tf_idf_matrix, 20)
# np.save('centroids', centroids)
# centroids = np.load('centroids.npy')
# clusters = cluster(tf_idf, centroids)
# print(clusters)

# clusters = load_dicts('cluster0.txt')
# np.set_printoptions(threshold=np.nan)
# print(tf_idf[0][1])
