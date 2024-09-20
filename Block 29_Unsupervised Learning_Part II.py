#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


#Load the dataset
ccData = pd.read_csv("CC GENERAL.csv")

# Display the number of rows and columns in the dataset
print("Rows:", ccData.shape[0])
print("Columns:", ccData.shape[1])
ccData.head()

ccData.drop('CUST_ID', axis = 1 , inplace = True)
ccData
ccData.info()

#Dont want to make up a random credit limit for a user, so just droped the whole row
cc_limit = ccData['CREDIT_LIMIT']
ccData.drop(5203, axis = 0, inplace = True)

#filled in the rest of MINIMUM_PAYMENTS and the NaN vaules with 0 
ccData = ccData.fillna(0)
ccData.info()


ccData_scaled = StandardScaler().fit(ccData).transform(ccData)

ccData_scaled_norm = normalize(ccData_scaled)
ccData_scaled_norm

pca = PCA(3)
pca_final = pca.fit_transform(ccData_scaled_norm)

cov = pca.get_covariance()
print(pca.get_feature_names_out())
for i in np.arange(3):
    index =  np.argmax(np.absolute(pca.get_covariance()[i]))
    max_cov = pca.get_covariance()[i][index]
    column = ccData.columns[index]
    print("Principal Component", i+1, "maximum covariance:", "{:.2f}".format(max_cov), "from column", column)


### 1D plot
ax = plt.axes()
im = ax.imshow(cov, cmap="RdYlBu_r", vmin=-.1, vmax=.1)
ax.set_xticks(list(range(0, 17)))
ax.set_yticks(list(range(0, 17)))
plt.colorbar(im).ax.set_ylabel("$Covariance$", rotation=90)
ax.set_title("Feature Analysis - Covariance")
plt.tight_layout()

### 2D plot
plt.figure(figsize=(8,6))
plt.scatter(pca_final[:,0],pca_final[:,1],c= pca_final[:,0:1], cmap='RdYlBu_r', s=10, alpha = 0.6)
plt.xlabel('First Principal Component - Balance')
plt.ylabel('Second Principal Component - Balance Frequency')
plt.title("First Two Principal Components")

### 3D plot
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
ax.scatter(
    pca_final[:,0],
    pca_final[:,1],
    pca_final[:,2],
    c=pca_final[:,0:1],
    cmap='RdYlBu_r',
    s=10,
    alpha = .6
)
ax.set_title("First Three Principal Components")
ax.set_xlabel("Balance")
ax.set_ylabel("Balance Frequency")
ax.set_zlabel("Purchases")
plt.show()



### KMEANS CLUSTERING
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import cdist

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 12)
all_models = [None]*(K[-1]+1)

for k in K:
    # Building and fitting the elbow model
    kmeanModel = KMeans(n_clusters=k, n_init='auto').fit(pca_final)
    y_KMeans = kmeanModel.fit_predict(pca_final)
    all_models[k] = y_KMeans.astype(np.int8)
    
    #Plotting the points with colors equivalent to their grouping
    plt.scatter(pca_final[:, 0], pca_final[:, 1], c=y_KMeans, alpha=0.25, s = 10)
    plt.title(f'K-means clustering ({k}-clusters)')
    plt.xlabel('First Principal Component - Balance')
    plt.ylabel('Second Principal Component - Balance Frequency')
    plt.show()
    distortions.append(sum(np.min(cdist(pca_final, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / ccData_scaled_norm.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(pca_final, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / ccData_scaled_norm.shape[0]
    mapping2[k] = kmeanModel.inertia_


## ELBOW PLOT
plt.plot(K, inertias, 'bx-')
plt.plot(6, inertias[4], 'ro-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertias to Determine Optimal K value')
plt.show()



import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import cdist
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
all_models = [None]

# Building and fitting the elbow model
kmeanModel = KMeans(n_clusters=6, n_init='auto').fit(pca_final)
y_KMeans = kmeanModel.fit_predict(pca_final)
all_models[0] = y_KMeans.astype(np.int8)

#Plotting the points with colors equivalent to their grouping
plt.scatter(pca_final[:,0],pca_final[:,1], c=all_models[0], s=20, alpha = 0.6)
plt.title(f'K-means clustering on PCA Transformed Data (6-clusters)')
plt.xlabel('First Principal Component - Balance')
plt.ylabel('Second Principal Component - Balance Frequency')
plt.show()

distortions.append(sum(np.min(cdist(pca_final, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / ccData_scaled_norm.shape[0])
inertias.append(kmeanModel.inertia_)

mapping1[k] = sum(np.min(cdist(pca_final, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / ccData_scaled_norm.shape[0]
mapping2[k] = kmeanModel.inertia_



