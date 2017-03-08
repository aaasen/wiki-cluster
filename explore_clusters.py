import sys

from load import load_k_means


def print_row(cells):
    print(' & '.join(str(cell) for cell in cells), end=' \\\\ \\hline \n')


def parbox(strings, width):
    return '\\parbox[t]{{{}}}{{{}}}'.format(width, ' \\\\ '.join(strings))


def main():
    if len(sys.argv) != 2:
        print('Usage: python explore_clusters.py [k]')
        exit(2)

    k = int(sys.argv[1])
    clusters = load_k_means(k)
    clusters.sort(key=lambda cluster: -len(cluster))


    for i in range(len(clusters)):
        cluster = clusters[i]

        if len(cluster) >= 10:
            print_row([
                i,
                len(cluster),
                parbox(cluster.top_words()[:5], '2cm'),
                parbox([doc.label for doc in cluster.top_documents()[:5]], '8cm')
            ])

if __name__ == '__main__':
    main()
