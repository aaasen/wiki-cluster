
import numpy as np

from load import (load_dictionary, load_documents,
                  document_matrix, write_documents)
from explore_clusters import print_row


def get_word_freq(docs):
    matrix = np.array(document_matrix(docs))
    return (matrix != 0).sum(0)


def explore():
    dictionary = load_dictionary()
    documents = load_documents(dictionary=dictionary)
    word_freq = get_word_freq(documents)

    def print_words(indexes):
        for i in indexes:
            print_row([dictionary[i], word_freq[i],
                       '{:.5}\%'.format(100 * word_freq[i] / len(documents))])

    # This is the number of documents that have each word.
    n = 10
    word_freq_sort_index = np.argsort(word_freq)

    print('least common words')
    print_words(word_freq_sort_index[:n])
    print('most common words')
    print_words(word_freq_sort_index[:-n-1:-1])
    print('median common words')
    l = len(word_freq_sort_index) // 2
    print_words(word_freq_sort_index[l-(n//2):l+(n//2)])
    print("median word frequency: {}".format(np.median(word_freq)))


def tfidf_threshold(minimum, path):
    dictionary = load_dictionary()
    documents = load_documents('train', dictionary=dictionary)
    word_freq = get_word_freq(documents)
    below_threshold = word_freq < minimum

    for doc in documents:
        doc.vector[below_threshold] = 0

    write_documents(documents, path)


def main():
    explore()
    # tfidf_threshold(160, 'train_min_160')


main()
