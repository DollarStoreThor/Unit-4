import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


### Exploratory Data Analysis (EDA)
df = pd.read_csv("heart.csv")
print(f"DF Head: \n", df.head(), "\n")
print(f"DF Desc: \n", df.describe(), "\n")
df.isnull().sum()

X = df.drop('AHD', axis=1)
y = df.drop(X, axis=1)
y = y.where(y.AHD == 'Yes', 0).where(y.AHD == 'No', 1)

df.update(y)
df = df.astype({'AHD':np.int64})
df.dtypes


### Data Visualization
sns.catplot(data=df, y="ChestPain", x="Age", hue="AHD", alpha=0.5)
sns.catplot(data=df, y="ChestPain", x="Age", hue="AHD", kind="violin", bw_adjust=.5, cut=0, split=True,)
sns.catplot(data=df, y="ChestPain", x="Age", hue="AHD", kind="boxen")

piv = df.drop(['ChestPain', 'Thal'], axis=1)

def showClusterMaps(cluster_map = False, _method = 'single'):
    if cluster_map:
        AHD = df.AHD
        piv2 = piv.drop('AHD', axis =1 )
        AHD_lut = dict(zip(AHD.unique(), "rbg"))
        AHD_row_colors = AHD.map(AHD_lut)
        if _method == 'centroid' or 'median':
            sns.clustermap(piv2, row_colors=AHD_row_colors, method=_method, z_score=0, center=0)
        else:
            sns.clustermap(piv2, row_colors=AHD_row_colors, metric="correlation", method=_method, z_score=0, center=0)


def showCorrelationalPlots(pair_grid = False, correlation = False, slow_pandas_scatter = False): 
    if pair_grid:
        pg = sns.PairGrid(df, diag_sharey=False)
        pg.map_upper(sns.scatterplot, s=15)
        pg.map_lower(sns.kdeplot)
        pg.map_diag(sns.kdeplot, lw=2)
        
    if correlation:
        sns.set_theme(style="whitegrid")
        # Compute a correlation matrix and convert to long-form
        corr_mat = df.corr().stack().reset_index(name="correlation")

        # Draw each cell as a scatter point with varying size and color
        g = sns.relplot(
            data=corr_mat,
            x="level_0", y="level_1", hue="correlation", size="correlation",
            palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
            height=10, sizes=(50, 250), size_norm=(-1, 1),
        )

        # Tweak the figure to finalize
        g.set(xlabel="", ylabel="", aspect="equal")
        g.despine(left=True, bottom=True)
        g.ax.margins(.02)
        for label in g.ax.get_xticklabels():
            label.set_rotation(90)
            
    if slow_pandas_scatter:
        pd.plotting.scatter_matrix(df, figsize=(20, 20))


# CALLS TO THE ABOVE VISULAIZATION METHODS
cluster_map_methods = {1:'single', 2:'complete', 3:'average', 4:'weighted', 5:'centroid', 6:'median', 7:'ward'}            
showClusterMaps(cluster_map = True, _method = cluster_map_methods[7])      
showCorrelationalPlots(pair_grid = False, correlation = True, slow_pandas_scatter = True)




### Data Cleaning and Preprocessing
min_values = df.min()
max_values = df.max()

range_df = pd.DataFrame({'min': min_values, 'max': max_values})

print(range_df)

le = LabelEncoder()
X['ChestPain'] = le.fit_transform(df.ChestPain)
le = LabelEncoder()
X['Thal'] = le.fit_transform(df.Thal)

def replace_outliers_with_median(data):
    for col in data.columns:
        print(col)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = data[col].median()
        
    return data
  
# Removing the outliers from the Data
sclr = StandardScaler()
sclr.fit(X)
X = sclr.transform(X)

# Convert y to numerical labels if it's categorical
le = LabelEncoder()
y = le.fit_transform(y)  # This will convert labels to numeric values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure y_train and y_test are 1D arrays
y_train = y_train.ravel()
y_test = y_test.ravel()


###  Model Development
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
y_pred = clf.predict_proba(X_test)

#gets the output from the model in a 1D form for the classification report
predicted_labels = y_pred.argmax(axis=1)
predicted_labels


### Model Evaluation
print(classification_report(y_true=y_test.ravel(), y_pred=predicted_labels))

