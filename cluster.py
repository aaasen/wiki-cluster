
import numpy as np

from collections import namedtuple

from distance import distance
import load


class Cluster(namedtuple('Cluster', 'centroid, documents')):
    def top_words(self):
        dictionary = load.load_dictionary()
        indexes = sorted(range(len(self.centroid)),
                         key=lambda i: -self.centroid[i])
        return [dictionary[i] for i in indexes]

    def distance_to_center(self, document):
        return distance(document.vector, self.centroid)

    def top_documents(self):
        def distance_to_center(document):
            return distance(document.vector, self.centroid)

        return sorted(self.documents, key=distance_to_center)

    def distortion(self):
        return sum(self.distance_to_center(doc) for doc in self.documents)

    def calculate_centroid(self):
        vectors = [document.vector for document in self.documents]
        return np.average(vectors, axis=0)

    def __len__(self):
        return len(self.documents)
