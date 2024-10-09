import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="R15.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
plt.show()

# Initialize an empty list to store inertia values
inertias = []

# Loop over the different values of k
for k in range(1, 20):
    print("------------------------------------------------------")
    print("Appel KMeans pour une valeur de k fixée")
    tps1 = time.time()
    
    # Run KMeans with the current k value
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    
    labels = model.labels_
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

    # Append the inertia to the list
    inertias.append(inertie)

    # Scatter plot of the clustering result
    plt.scatter(f0, f1, c=labels, s=8)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    plt.title("Données après clustering : " + str(name) + " - Nb clusters =" + str(k))
    plt.show()

    print("nb clusters =", k, ", nb iter =", iteration, ", inertie = ", inertie, ", runtime = ", round((tps2 - tps1) * 1000, 2), "ms")

# Plot the relation between number of clusters and inertia
plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), inertias, marker='o')
plt.title("Relation entre le nombre de clusters et l'inertie")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.grid(True)
plt.show()
