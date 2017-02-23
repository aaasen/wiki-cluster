import numpy as np

import csv
import pickle

from document import Document
from config import DICTIONARY_PATH, DOCUMENT_PATH
import k_means


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


def load_k_means(k, document_path=DOCUMENT_PATH):
    path = '{}_{}.pickle'.format(document_path, k)
    try:
        with open(path, 'rb') as f:
            print('loading cached clusters from {}'.format(path))
            return pickle.load(f)
    except FileNotFoundError:
        print('no cached clusters found')
        documents = load_documents(document_path)
        clusters = k_means.cluster(k, documents)
        with open(path, 'wb') as f:
            pickle.dump(clusters, f)
        return clusters
