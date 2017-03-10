
import matplotlib.pyplot as plt

from load import load_k_means


def distortion(clusters):
    return sum(cluster.distortion() for cluster in clusters)

xs = [2 ** n for n in range(7)]
ys = [distortion(load_k_means(k)) for k in xs]
plt.plot(xs, ys)

plt.title('K vs. Distortion')
plt.xlabel('K')
plt.ylabel('Total Distortion')
plt.tight_layout()
plt.savefig('writeup/images/k_vs_distortion.png')
