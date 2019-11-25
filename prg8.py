import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
data = pd.read_csv('8-kmeansdata.csv')
f1 = data['Distance_Feature']
f2 = data['Speeding_Feature']
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, color='black')
plt.show()
kmeans = KMeans(3).fit(X)
labels = kmeans.predict(X)
print('Graph using Kmeans Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
gmm = GaussianMixture(3).fit(X)
labels = gmm.predict(X)
print('Graph using EM Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()