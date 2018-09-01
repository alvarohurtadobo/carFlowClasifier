import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

"""
CSV File: Country,Region,Population,Area (sq. mi.),Pop. Density (per sq. mi.),Coastline (coast/area ratio),Net migration,Infant mortality (per 1000 births),GDP ($ per capita),Literacy (%),Phones (per 1000),Arable (%),Crops (%),Other (%),Climate,Birthrate,Deathrate,Agriculture,Industry,Service
"""

with open('countriesoftheworld.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    pais = []                    # Columna 0
    absisa = []          # Column 8
    ordenada = []         # Column 7
    contador = 0

    for row in readCSV:
        print(row[0])
        pais.append(row[0])
        #absisa.append(row[8])
        absisa.append(contador)
        ordenada.append(row[7].replace(',','.'))
        contador +=1
    print('{} paises cargados'.format(contador))
    toPlot = sorted(zip(ordenada,absisa))
    #plt.scatter(absisa,ordenada)
    #plt.show()

    X = np.array(sorted(zip(absisa,ordenada)))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #print(centroids)
    #print(labels)

    colors = ["g.","r.","c."]
    print('### GRUPO 0 ###')
    for i in range(len(X)):
        #print("coordinate:",X[i], "label:", labels[i])
        if labels[i] == 0:
            print('\t',pais[int(X[i][0])])
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

    print('### GRUPO 1 ###')
    for i in range(len(X)):
        #print("coordinate:",X[i], "label:", labels[i])
        if labels[i] == 1:
            #print('\t',pais[X[i][0]])
            print('\t',pais[int(X[i][0])])


    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

    plt.show()
