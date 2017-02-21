
import csv
import numpy as np
import random

random.seed(0)


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


def load_tf_idf(path, n_words):
    def tf_idf_vector(tf_idf_pairs):
        tf_idf_vector = np.zeros(n_words)
        for word_index, tf_idf in tf_idf_pairs:
            tf_idf_vector[int(word_index)] = tf_idf
        return tf_idf_vector

    return [(title, tf_idf_vector(pairs)) for title, pairs in load_dicts(path)]


dictionary = load_dictionary('dictionary.txt')
tf_idf = load_tf_idf('tfidf.txt', len(dictionary))
tf_idf_matrix = [vector for title, vector in tf_idf]

n_clusters = 20


def centroid_index(vector, centroids):
    distances = [np.linalg.norm(np.subtract(vector, centroid))
                 for centroid in centroids]
    return np.argmin(distances)


def cluster(title_tf_idf_tuples, centroids):
    clusters = [[] for centroid in centroids]

    for title, vector in title_tf_idf_tuples:
        clusters[centroid_index(vector, centroids)].append(title)

    return clusters


def k_means_step(matrix, centroids):
    clusters = [[] for centroid in centroids]

    for row in matrix:
        clusters[centroid_index(row, centroids)].append(row)

    return np.array([np.average(cluster, axis=0) for cluster in clusters])


def k_means(matrix, n_clusters, centroids=None):
    if centroids is None:
        centroids = random.sample(matrix, n_clusters)

    while True:
        print(centroids)
        prev_centroids = centroids
        centroids = k_means_step(matrix, centroids)
        if np.array_equal(centroids, prev_centroids):
            return centroids


# centroids = k_means(tf_idf_matrix, 20)
# np.save('centroids', centroids)
centroids = np.load('centroids.npy')
clusters = cluster(tf_idf, centroids)
print(clusters)

# clusters = load_dicts('cluster0.txt')
# np.set_printoptions(threshold=np.nan)
# print(tf_idf[0][1])
