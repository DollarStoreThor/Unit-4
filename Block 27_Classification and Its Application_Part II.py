# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Importing dataset
df = pd.read_csv('adult.csv')
# Preview dataset
df.head()
# Shape of dataset
df.info()
# Features data-type
df.dtypes
# Statistical summary
df.describe(


#set the '?' value to NaN
df.mask(df == '?', inplace = True)
print(df.isnull().sum())
#iterate through the columns in our dataframe
for col in df.columns:
    #where we see the Nan values we set just that value equal to the mode of the column
    df[col].fillna(value=df[col].mode()[0], inplace=True)

print(df.isnull().sum())

#quick visulal to check if the varibale being optimized for is represented equally
sns.catplot(data=df, x='income', kind='count')
# Creating a distribution plot for 'Age'
sns.catplot(data=df, y='age', kind='count')
# Creating a distribution plot for 'Age'
sns.violinplot(data=df, x='age', y='workclass', hue='income', split=True, gap=.1, inner="quart")

# Creating a barplot for 'Education'
import seaborn.objects as so
sns.catplot(data=df, y='education', kind='count', hue='workclass')
p = so.Plot(data=df, x="age", y="education").add(so.Dots())
p.facet("income")

# Creating a barplot for 'Years of Education'
sns.countplot(df, x="education.num" , hue='income')
sns.pairplot(df, hue='income')

# Creating a pie chart for 'Marital status'
values = df.groupby('marital.status').count()['age']
plt.pie(values, labels=values.index, autopct='%.0f%%')

# Creating a barplot for 'Hours per week'
h = so.Plot(df, x="hours.per.week")
h.add(so.Bars(), so.Hist(bins=10), color='income')

# Creating a count plot of income across age
h = so.Plot(df, x="age")
h.add(so.Bars(), so.Hist(bins=10), color='income')


# Creating a countplot of income across education
h = so.Plot(df, y="education")
h.add(so.Bars(), so.Hist(bins=16), color='income')

# Creating a countplot of income across years of education
h = so.Plot(df, x="education.num")
h.add(so.Bars(), so.Hist(bins=16), color='income')


# Creating a countplot of income across Marital Status
h = so.Plot(df, y="marital.status")
h.add(so.Bars(), so.Hist(bins=7), color='income')



# Creating a heatmap of correlations
only_num = df.select_dtypes(include=['int64']).columns
only_num = df[only_num]
sns.heatmap(only_num.corr(), annot=True)


### Label Encoding categorical columns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X):
        return self.transform(X)

     
columns_to_encode = df.select_dtypes(include=['object']).columns.tolist()
df = MultiColumnLabelEncoder(columns = columns_to_encode).fit_transform(df)
df.info()
df.head(20)

# Setting up our features and labels
X = df.drop(labels=['income'], axis=1)
y =  df['income']


# Scaleing our features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
df_feat = pd.DataFrame(scaled_features,columns=X.columns)
df_feat.info()
df_feat.head()


### Taking care of the class imbalances
from imblearn.over_sampling import SMOTE, RandomOverSampler
# Apply SMOTE to oversample the minority class
oversample = SMOTE(random_state=42)
X_SMOTE_df, y_SMOTE_df = oversample.fit_resample(X, y)
# Apply RandomOverSampler to oversample randomly
rand_oversample = RandomOverSampler(random_state=42)
X_rand_ovr_df, y_rand_ovr_df = rand_oversample.fit_resample(X, y)

X_SMOTE_df.info()
y_SMOTE_df.info()

X_rand_ovr_df.info() 
y_rand_ovr_df.info()

### Train test and splitting our data
from sklearn.model_selection import train_test_split, GridSearchCV
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_SMOTE_df, y_SMOTE_df, train_size=0.8, random_state=42)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(X_rand_ovr_df, y_rand_ovr_df, train_size=0.8, random_state=42)

from sklearn.model_selection import train_test_split, GridSearchCV
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_SMOTE_df, y_SMOTE_df, train_size=0.8, random_state=42)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(X_rand_ovr_df, y_rand_ovr_df, train_size=0.8, random_state=42)

from sklearn.model_selection import train_test_split, GridSearchCV
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_SMOTE_df, y_SMOTE_df, train_size=0.8, random_state=42)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(X_rand_ovr_df, y_rand_ovr_df, train_size=0.8, random_state=42)

from sklearn.model_selection import train_test_split, GridSearchCV
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_SMOTE_df, y_SMOTE_df, train_size=0.8, random_state=42)
X_rand_train, X_rand_test, y_rand_train, y_rand_test = train_test_split(X_rand_ovr_df, y_rand_ovr_df, train_size=0.8, random_state=42)

from sklearn.naive_bayes import GaussianNB
GNB_smote = GaussianNB().fit(X_smote_train, y_smote_train)
GNB_rand = GaussianNB().fit(X_rand_train, y_rand_train)

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
feature_names = list(X.columns)
DTC_smote = DecisionTreeClassifier(random_state=42).fit(X_smote_train, y_smote_train)
DTC_rand = DecisionTreeClassifier(random_state=42).fit(X_rand_train, y_rand_train)

from sklearn.ensemble import RandomForestClassifier
RFC_smote = RandomForestClassifier(random_state=42).fit(X_smote_train, y_smote_train)
RFC_rand = RandomForestClassifier(random_state=42).fit(X_rand_train, y_rand_train)

### Visualization and Reporting
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
y_rand_pred_lr = lr_rand.predict(X_rand_test)
print('LogisticRegressionCV, Random: \n', classification_report(y_rand_test, y_rand_pred_lr))
y_smote_pred_lr = lr_smote.predict(X_smote_test)
print('LogisticRegressionCV, SMOTE: \n', classification_report(y_smote_test, y_smote_pred_lr))
plt.figsize(30, 30)
ax = plt.gca()
lr_rand_disp = RocCurveDisplay.from_estimator(lr_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
lr_smote_disp = RocCurveDisplay.from_estimator(lr_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


y_rand_pred_knc = knc_rand.predict(X_rand_test)
print('K-nearest Classifciaton, Random: \n', classification_report(y_rand_test, y_rand_pred_knc))
y_smote_pred_knc = knc_smote.predict(X_smote_test)
print('K-nearest Classifciaton, SMOTE: \n', classification_report(y_smote_test, y_smote_pred_knc))
ax = plt.gca()
knc_rand_disp = RocCurveDisplay.from_estimator(knc_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
knc_smote_disp = RocCurveDisplay.from_estimator(knc_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


y_smote_pred_smv = smv_smote.predict(X_smote_test)
print('Support Vector Classifier accuracy, SMOTE: \n',classification_report(y_smote_test, y_smote_pred_smv))
y_rand_pred_smv = smv_rand.predict(X_rand_test)
print('Support Vector Classifier accuracy, Rand: \n',classification_report(y_rand_test, y_rand_pred_smv))
ax = plt.gca()
svm_rand_disp = RocCurveDisplay.from_estimator(smv_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
svm_smote_disp = RocCurveDisplay.from_estimator(smv_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


y_smote_pred_GNB = GNB_smote.predict(X_smote_test)
print('Gausian Naive Bayes, SMOTE: \n',classification_report(y_smote_test, y_smote_pred_GNB))
y_rand_pred_GNB = GNB_rand.predict(X_rand_test)
print('Gausian Naive Bayes, Rand: \n',classification_report(y_rand_test, y_rand_pred_GNB))
ax = plt.gca()
GNB_rand_disp = RocCurveDisplay.from_estimator(GNB_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
GNB_smote_disp = RocCurveDisplay.from_estimator(GNB_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


y_smote_pred_DTC = DTC_smote.predict(X_smote_test)
print('Decission Tree Classifier accuracy, SMOTE: \n',classification_report(y_smote_test, y_smote_pred_DTC))
y_rand_pred_DTC = DTC_rand.predict(X_rand_test)
print('Decission Tree Classifier accuracy, Rand: \n',classification_report(y_rand_test, y_rand_pred_DTC))
ax = plt.gca()
DTC_rand_disp = RocCurveDisplay.from_estimator(DTC_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
DTC_smote_disp = RocCurveDisplay.from_estimator(DTC_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


y_smote_pred_RFC = RFC_smote.predict(X_smote_test)
print('Random Forest Classifier accuracy, SMOTE: \n',classification_report(y_smote_test, y_smote_pred_RFC))
y_rand_pred_RFC = RFC_rand.predict(X_rand_test)
print('Random Forest Classifier accuracy, Rand: \n',classification_report(y_rand_test, y_rand_pred_RFC))
ax = plt.gca()
RFC_rand_disp = RocCurveDisplay.from_estimator(RFC_rand, X_rand_test, y_rand_test, ax=ax, alpha=0.8)
RFC_smote_disp = RocCurveDisplay.from_estimator(RFC_smote, X_smote_test, y_smote_test, ax=ax, alpha=0.8)


### Final Model Selection
RFC_rand.get_params()
#random_state=42).fit(X_rand_train, y_rand_train
FinalModel = RandomForestClassifier()


#{'ccp_alpha': 0.0, 'class_weight': 'balanced', 'max_features': 'sqrt', 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 10}
Param_grid = {'max_features': ['sqrt', 'log2', None],
              'ccp_alpha': [0.0, 0.01, 0.001],
              'warm_start': [True]}

grid = GridSearchCV(FinalModel, Param_grid, refit = True, verbose = 2, cv=5, n_jobs=-1)
grid_search=grid.fit(X_rand_train, y_rand_train)
print(grid_search.best_params_)


print("Confusion Matrix(before Hyperparameter tuning): ")
cm = confusion_matrix(y_rand_test, y_rand_pred_RFC)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
ax = plt.gca() 
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False) 
plt.show()
y_rand_pred_RFC = RFC_rand.predict(X_rand_test)
print('Area Under Curve(AUC) before Hyperparameter tuning: ', roc_auc_score(y_rand_test, y_rand_pred_RFC))
print('Random Forest Classifier accuracy, Rand (BEFORE Hyper-parameter tuning): \n',classification_report(y_rand_test, y_rand_pred_RFC))


y_rand_pred_RFC_hyper = grid_search.predict(X_rand_test)
print('Grid Search Area Under Curve(AUC) AFTER Hyperparameter tuning: ',  roc_auc_score(y_rand_test, y_rand_pred_RFC_hyper))
print('Random Forest Classifier accuracy, Rand (AFTER Hyperparameter tuning): \n',classification_report(y_rand_test, y_rand_pred_RFC_hyper))
cm = confusion_matrix(y_rand_test, y_rand_pred_RFC_hyper)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['<=50K', '>50K'])
print("Confusion Matrix(after hyper tuning): ")
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
ax = plt.gca() 
ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False) 
#plt.title("Confusion Matrix(after hyper tuning): ")
plt.show()

