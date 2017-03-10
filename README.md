
# Wiki Cluster

Clustering Wikipedia documents using K-Means.

## Reproducing Results

All of the code is written in Python 3.
The only external dependency is numpy.

Training documents are in `train` and the dictionary is in `dictionary`.
`train_min_{n}` files contain all training documents with words that occur less
than `n` times removed. Configuration is in `config.py`.

To regenerate charts, run `python plot_k_vs_cluster_size.py`,
`python plot_k_vs_distortion.py`, or `python different_init.py`
depending on the chart that you would like to regenerate.
The initialization method can be changed within each file.
Note that generating the first chart may take a while,
but that subsequent runs will read from a disk cache and go
very quickly. All charts will be written to the `writeup/images`
directory and automatically incorporated into the writeup.

To generate cluster tables, run `python explore_clusters.py [k]`.
This will generate a table in Latex format.

The `uncommon_words.py` script can be used to filter out uncommon words
or generate statistics about the most and least common words in the dataset.

The `lda.py` script uses sklearn for Latent Dirichlet allocation.
This is just a footnote in the paper.
