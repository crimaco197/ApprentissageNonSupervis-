import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="xclara.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Initialize an empty list to store Davies-Bouldin index values
DB_Indexes = []

### FIXER la distance
# 

for seuil_dist in range(1, 50):
    tps1 = time.time()
    model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
    model = model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # Nb iteration of this method
    #iteration = model.n_iter_
    k = model.n_clusters_
    leaves=model.n_leaves_

    # Calculate the Davies-Bouldin Index, and add it to the list
    if k != 1:
        DB_Indexes.append(metrics.davies_bouldin_score(datanp, labels))

    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
    #plt.show()

    print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms", ", seuil_dist =", seuil_dist)

# Plot the relation between the distance threshold and the Davies-Bouldin index
plt.figure(figsize=(8, 6))
plt.plot(range(2, len(DB_Indexes)), DB_Indexes[2:len(DB_Indexes) + 1], marker='o')
plt.title("Relation entre le seuil de distance et l'index de Davies-Bouldin")
plt.xlabel("Seuil de Distance")
plt.ylabel("Index de Davies-Bouldin")
plt.grid(True)
plt.show()


###
# FIXER le nombre de clusters
###
k=5
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")



#######################################################################