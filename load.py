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
        return [pair(row) for row in f.readlines() if ',' in row]


def write_documents(docs, path):
    def format_tuples(tuples):
        return ','.join('{}:{}'.format(key, value) for key, value in tuples)

    def format_row(title, tuples):
        return '{}|{}\n'.format(title, format_tuples(tuples))

    def format_doc(doc):
        title = doc.label
        tuples = ((i, x) for i, x in enumerate(doc.vector) if x != 0)
        return format_row(title, tuples)

    with open(path, 'w') as f:
        f.writelines(format_doc(doc) for doc in docs)


def load_documents(path=DOCUMENT_PATH, dictionary_path=DICTIONARY_PATH,
                   dictionary=None):
    # Could speed this up since we only need to know how many words exist.
    if dictionary is None:
        dictionary = load_dictionary(dictionary_path)

    n_words = len(dictionary)

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


def load_k_means(k, document_path=DOCUMENT_PATH, seed=None):
    path = '{}_{}{}.pickle'.format(document_path, k,
                                   '' if seed is None else '_{}'.format(seed))
    try:
        with open(path, 'rb') as f:
            print('loading cached clusters from {}'.format(path))
            return pickle.load(f)
    except FileNotFoundError:
        print('no cached clusters found')
        documents = load_documents(document_path)
        clusters = k_means.cluster(k, documents, seed=seed)
        with open(path, 'wb') as f:
            pickle.dump(clusters, f)
        return clusters
