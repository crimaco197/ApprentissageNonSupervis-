import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="complex9.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

for k in range(2,3):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(datanp)
    distances, indices = neigh.kneighbors(datanp)

    # distances moyenne sur les k plus proches voisins
    # en retirant le point "origine"
    newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0,distances.shape[0])])

    # tirer par ordre croissant
    distancetrie = np.sort(newDistances)

    plt.title("Plus proches voisins " + str(k))
    plt.plot(distancetrie)
    plt.show()

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
#epsilon_values = [4.41, 5.16, 6.02, 6.95, 7.64, 8.23, 8.73, 9.11, 9.53, 9.84, 
#                  10.39, 10.81, 11.28, 11.57, 12.05, 12.47, 12.78, 13.15]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
#epsilon_values = [6.19, 6.94, 7.44, 8.23, 9.03, 9.67, 10.10, 10.70, 11.30, 
#                   11.85, 12.29, 12.70, 13.20, 13.70, 14.03, 14.61, 15.00, 15.30]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
#epsilon_values = [7.19, 8.26, 8.75, 9.53, 10.20, 10.74, 11.52, 11.93, 12.60, 13.18,
#                  13.95, 14.28, 15.03, 15.26, 15.77, 16.44, 16.78, 16.95]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
#epsilon_values = [7.59, 8.66, 9.15, 9.93, 10.60, 11.14, 11.92, 12.33, 13.00, 13.58,
#                  14.35, 14.68, 15.43, 15.66, 16.17, 16.84, 17.18, 17.35]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
#epsilon_values = [6.76, 7.72, 8.40, 8.95, 9.74, 10.35, 11.13, 11.42, 12.02, 12.66,
#                  12.95, 13.13, 13.80, 14.37, 14.96, 15.28, 16.07, 16.41]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample
epsilon_values = [6.26, 7.22, 7.90, 8.45, 9.24, 9.85, 10.63, 10.92, 12.52, 12.16,
                  12.45, 12.63, 13.30, 13.87, 14.46, 14.78, 15.57, 15.91]

# Valeurs epsilon correspondantes pour 2 à 20 valeurs min_sample pour zelnik4.arff
#epsilon_values = [0.0093, 0.0122, 0.0137, 0.0145, 0.0152, 0.0165, 0.0172, 0.0183, 0.0182, 
#                  0.0197, 0.0205, 0.0214, 0.0227, 0.0230, 0.0232, 0.0241, 0.0250, 0.0252]

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

# Initialize an empty array to hold the Davies-Bouldin index scores
DB_Indexes = []

# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
# 
print("------------------------------------------------------")
print("Appel DBSCAN (1) ... ")
for min_pts in range(2,20):
    tps1 = time.time()
    epsilon= epsilon_values[min_pts - 2] #3.1155*(min_pts**0.488) #2  # 4
    #min_pts= 10 #10   # 10
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    DB_Indexes.append(metrics.davies_bouldin_score(datanp[labels != -1], labels[labels != -1]))

    #plt.scatter(f0, f1, c=labels, s=8)
    #plt.title("Données après clustering DBSCAN (1) - Epislon= "+str(epsilon)+" MinPts= "+str(min_pts))
    #plt.show()


# Plot the relation between number of, minimum samples and corresponding epsilon, and silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 20), DB_Indexes, marker='o')
plt.title("Relation entre le nombre de min_samples (et eps correspondante) et l'index de Davies Bouldin")
plt.xlabel("Nombre de min_samples")
plt.ylabel("Index de Davies Bouldin")
plt.grid(True)
plt.show()

####################################################
# Standardisation des donnees

# Initialize an empty array to hold the Davies-Bouldin index scores for the standarized data
DB_Indexes_standard = []

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


print("------------------------------------------------------")
print("Appel DBSCAN (2) sur données standardisees ... ")
for min_pts in range(2,20):
    tps1 = time.time()
    epsilon=epsilon_values[min_pts - 2]*0.009 #0.05
    #min_pts=10 # 10
    model = cluster.DBSCAN(eps=epsilon, min_samples=min_pts)
    model.fit(data_scaled)

    tps2 = time.time()
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print('Number of clusters: %d' % n_clusters)
    print('Number of noise points: %d' % n_noise)

    DB_Indexes_standard.append(metrics.davies_bouldin_score(data_scaled[labels != -1], labels[labels != -1]))

    plt.scatter(f0_scaled, f1_scaled, c=labels, s=8)
    plt.title("Données après clustering DBSCAN (2) - Epsilon= "+str(epsilon)+" MinPts= "+str(min_pts))
    plt.show()


# Plot the relation between the minimum samples and corresponding epsilon, and the Davies-Bouldin index
plt.figure(figsize=(8, 6))
plt.plot(range(2, len(DB_Indexes_standard)), DB_Indexes_standard[2:len(DB_Indexes_standard) + 1], marker='o')
plt.title("Relation entre Min_samples et l'index de Davies-Bouldin")
plt.xlabel("Min_samples")
plt.ylabel("Index de Davies-Bouldin")
plt.grid(True)
plt.show()


