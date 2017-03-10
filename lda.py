
from sklearn.decomposition import LatentDirichletAllocation as LDA

from load import load_dictionary, load_documents, document_matrix
from config import SEED


def print_row(cells):
    print(' & '.join(str(cell) for cell in cells), end=' \\\\ \\hline \n')


def top_words(model, dictionary, n):
    def top_words_topic(topic):
        return [dictionary[i] for i in topic.argsort()[:-n - 1:-1]]

    return [top_words_topic(topic) for topic in model.components_]


def lda():
    dictionary = load_dictionary()
    documents = load_documents(dictionary=dictionary)
    matrix = document_matrix(documents)
    lda = LDA(n_topics=16, learning_method='batch', random_state=SEED)
    lda.fit(matrix)
    for topic in top_words(lda, dictionary, 5):
        print_row([', '.join(topic)])


lda()
