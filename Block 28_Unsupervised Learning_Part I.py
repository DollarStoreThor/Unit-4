import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

df = pd.read_csv("Mall_customers.csv")

df1 = df.drop(['CustomerID', 'Age'], axis =1)
print(df1.isnull().sum())

sclr = StandardScaler()

sclr.fit(df1)
X = sclr.transform(df1


import scipy.cluster.hierarchy as shc

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

c = ['purple', 'blue', 'teal', 'lime', 'yellow', 'orange', 'black', 'midnightblue', 'hotpink']

all_models = [None]*(K[-1]+1)

for k in K:
    # Building and fitting the elbow model
    kmeanModel = KMeans(n_clusters=k, n_init='auto').fit(X)
    
    y_KMeans = kmeanModel.fit_predict(X)
    all_models[k] = y_KMeans.astype(np.int8)
    #Plotting the points with colors equivalent to their grouping
    plt.scatter(X[:, 0], X[:, 1], c=y_KMeans, alpha=1)
    
    
    #Plotting all the cluster centers with uniqe colors
    for i in range(k):
        plt.scatter(kmeanModel.cluster_centers_[i, 0],kmeanModel.cluster_centers_[i, 1], s=1100-(100*k), c='red', alpha=.5)
    
    plt.title(f'K-means clustering ({k}-clusters)')
    plt.xlabel('Feature 1 - Annual Income')
    plt.ylabel('Feature 2 - Spending Score')
    plt.show()
     
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_


plt.plot(K, distortions, 'bx-')
plt.plot(5, distortions[4], 'ro-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion to Determine Optimal K value')
plt.show()


plt.figure (figsize=(10,7))

x = [0, 2000]  # x-coordinates of the line
y = [5, 5]  # y-coordinates of the line
plt.plot(x, y, 'k--') 

dend = shc.dendrogram(shc.linkage(X, method='ward'))




#fitting the KMeans model with 5 clusters as determined from the elbow graph above
model_KMeans = KMeans(n_clusters=5,  init='k-means++', n_init='auto', random_state=42)
y_KMeans = model_KMeans.fit_predict(X)

#Plotting the points with colors equivalent to their grouping
plt.scatter(X[:, 0], X[:, 1], c=y_KMeans, alpha=1)
#X['Centroid'] = 

df['K-Means Cluster'] = all_models[5]

#Plotting all the cluster centers with uniqe colors
plt.scatter(model_KMeans.cluster_centers_[0, 0],model_KMeans.cluster_centers_[0, 1], s=500, c='purple',label='Cluster 1', alpha=.35)
plt.scatter(model_KMeans.cluster_centers_[1, 0],model_KMeans.cluster_centers_[1, 1], s=500, c='blue',label='Cluster 2', alpha=0.35)
plt.scatter(model_KMeans.cluster_centers_[2, 0],model_KMeans.cluster_centers_[2, 1], s=500, c='teal',label='Cluster 3', alpha=0.35)
plt.scatter(model_KMeans.cluster_centers_[3, 0],model_KMeans.cluster_centers_[3, 1], s=500, c='lime',label='Cluster 4', alpha=0.35)
plt.scatter(model_KMeans.cluster_centers_[4, 0],model_KMeans.cluster_centers_[4, 1], s=500, c='yellow',label='Cluster 5', alpha=0.35)

plt.legend()
plt.title(f'K-means clustering (5-clusters)')
plt.xlabel('Feature 1 - Annual Income')
plt.ylabel('Feature 2 - Spending Score')
plt.show())



import seaborn as sns
sns.color_palette("mako")
data = df.drop('CustomerID', axis=1).groupby(['K-Means Cluster'])
print(df.columns)

sns.pairplot(data=df, hue='K-Means Cluster', kind='scatter', diag_kind='kde', palette='mako')

g = sns.jointplot(data=df, x="Age", y="Spending Score (1-100)", hue="K-Means Cluster", palette='mako')
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

g = sns.jointplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", palette='mako')
g.plot(sns.kdeplot, sns.scatterplot, c='g')









