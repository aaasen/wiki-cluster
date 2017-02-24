import matplotlib.pyplot as plt
import numpy as np

from load import load_k_means

ks = [2 ** n for n in range(9)]
lens = [[len(cluster) for cluster in load_k_means(k)] for k in ks]

for i in range(len(ks)):
    print('K={}'.format(ks[i]))
    print(sorted(lens[i], reverse=True))


def plot_stat(func):
    plt.plot(ks, list(map(getattr(np, func), lens)), label=func)

plot_stat('min')
plot_stat('median')
plot_stat('max')

plt.legend()
plt.title('K vs. Cluster Size')
plt.xlabel('K')
plt.ylabel('Cluster Size')
plt.yscale('log', basey=2)
plt.xscale('log', basex=2)
plt.tight_layout()
plt.savefig('writeup/images/k_vs_cluster_size.png')
plt.show()
