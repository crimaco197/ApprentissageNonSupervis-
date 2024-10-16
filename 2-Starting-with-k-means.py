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
name="birch-rg3.arff"

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

# Initialize an empty list to store inertia values, silhouette scores and runtimes for k-means and mini-batch
inertias = []
silhouette_scores = []
runtimes = []
runtimesMini = []

# Loop over the different values of k
for k in range(1, 50):
    print("------------------------------------------------------")
    print("Appel KMeans pour une valeur de k fixée à ", k)
    tps1 = time.time()
    
    # Run KMeans with the current k value
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()

    tps_minibatch_1 = time.time()

    # Run Mini-Batch with the current k value
    modelMini = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
    modelMini.fit(datanp)
    tps_minibatch_2 = time.time()
    
    labels = model.labels_
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_

    # Append the inertia to the list
    inertias.append(inertie)

    # Calculate the silhouette score and append it
    if k > 1:
        silhouette_scores.append(metrics.silhouette_score(datanp, labels))

    # Scatter plot of the clustering result
    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
    #plt.title("Données après clustering : " + str(name) + " - Nb clusters =" + str(k))
    #plt.show()

    # Append the k-means runtime and mini-batch runtime to their respective lists
    runtimes.append(round((tps2 - tps1) * 1000, 2))
    runtimesMini.append(round((tps_minibatch_2 - tps_minibatch_1) * 1000, 2))

    print("nb clusters =", k, ", nb iter =", iteration, ", inertie = ", inertie, ", runtime = ", round((tps2 - tps1) * 1000, 2), "ms")

# Plot the relation between number of clusters and runtime
plt.figure(figsize=(8, 6))
plt.plot(range(2, 50), runtimes[1:50], marker='o')
plt.plot(range(2, 50), runtimesMini[1:50], marker='o')
plt.title("Relation entre le nombre de clusters et le temps de calcul")
plt.xlabel("Nombre de clusters")
plt.ylabel("Temps de calcul")
plt.grid(True)
plt.show()

# Plot the relation between number of clusters and inertia
#plt.figure(figsize=(8, 6))
#plt.plot(range(10, 25), inertias[9:25], marker='o')
#plt.title("Relation entre le nombre de clusters et l'inertie")
#plt.xlabel("Nombre de clusters")
#plt.ylabel("Inertie")
#plt.grid(True)
#plt.show()

# Plot the relation between number of clusters and silhouette scores
#plt.figure(figsize=(8, 6))
#plt.plot(range(2, 25), silhouette_scores, marker='o')
#plt.title("Relation entre le nombre de clusters et le coefficient de silhouette")
#plt.xlabel("Nombre de clusters")
#plt.ylabel("Coefficient de silhouette")
#plt.grid(True)
#plt.show()
