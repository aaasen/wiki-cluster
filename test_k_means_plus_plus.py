
from load import load_documents
import k_means


k = 16
documents = load_documents('train100')
clusters_rand = k_means.cluster(k, documents, init=k_means.random_documents)
clusters_pp = k_means.cluster(k, documents)

for clusters in [clusters_pp, clusters_rand]:
    print(sum(cluster.distortion() for cluster in clusters))
    for cluster in clusters:
        print(len(cluster))
