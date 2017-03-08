
import numpy as np

from load import load_dictionary, load_documents, document_matrix
from explore_clusters import print_row

dictionary = load_dictionary()
documents = load_documents(dictionary=dictionary)
matrix = np.array(document_matrix(documents))

# This is the number of documents that have each word.
word_freq = (matrix != 0).sum(0)


def print_words(indexes):
    for i in indexes:
        print_row([dictionary[i], word_freq[i],
                   '{:.5}\%'.format(100 * word_freq[i] / len(documents))])


n = 10
word_freq_sort_index = np.argsort(word_freq)

# print('least common words')
print_words(word_freq_sort_index[:n])
# print('most common words')
print_words(word_freq_sort_index[:-n-1:-1])

# print("median word frequency: {}".format(np.median(word_freq)))
