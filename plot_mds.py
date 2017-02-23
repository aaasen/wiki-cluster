
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
