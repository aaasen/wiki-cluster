
from load import load_k_means
from cluster import Cluster


for seed in range(5, 10):
    clusters = load_k_means(16, 'train_min_16', seed)
    print('seed {}: distortion {}'
          .format(seed, Cluster.total_distortion(clusters)))
