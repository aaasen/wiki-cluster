
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

n_clusters = 20

clusters = load_dicts('cluster0.txt')
a = set()

for title, cluster in clusters:
    a = a | dict(cluster).keys()

print(len(a))
print(sum(len(cluster) for title, cluster in clusters))


# np.set_printoptions(threshold=np.nan)
# print(tf_idf[0][1])
