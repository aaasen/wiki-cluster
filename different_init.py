
import matplotlib.pyplot as plt

from load import load_documents
from k_means import k_means_plus_plus, cluster, random_documents


def distortion(clusters):
    return sum(cluster.distortion() for cluster in clusters)


def distortion_for_seed(seed):
    centroids = k_means_plus_plus(16, documents, seed=seed)
    # centroids = random_documents(16, documents, seed=seed)
    clusters = cluster(16, documents, centroids=centroids)
    return distortion(clusters)


documents = load_documents('train100')
distortions = [distortion_for_seed(i) for i in range(100)]

plt.title('Distortion for Different Random Seeds')
plt.xlabel('Distortion')
plt.hist(distortions, bins=20)
plt.tight_layout()
plt.savefig('writeup/images/different_init.png')
plt.show()
