import sys

from load import load_k_means

if len(sys.argv) != 2:
    print('Usage: python explore_clusters.py [k]')
    exit(2)

k = int(sys.argv[1])
clusters = load_k_means(k)
clusters.sort(key=lambda cluster: -len(cluster))


def print_row(cells):
    print('&'.join(str(cell) for cell in cells), end=' \\\\\n')


for i in range(len(clusters)):
    cluster = clusters[i]

    print_row([
        len(cluster),
        ', '.join(cluster.top_words()[:5]),
        ', '.join([doc.label for doc in cluster.top_documents()[:5]])
    ])
