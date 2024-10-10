import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, log_loss

df = pd.read_csv("sepsis.csv")
df.head()
df.info()

label_counts = df['SepsisLabel'].value_counts()

label_counts = df['SepsisLabel'].value_counts()
fig, ax = plt.subplots()
labels = ['Sepsis', 'Non-Spesis']
ax.pie(label_counts, labels=labels, autopct='%1.1f%%')
plt.title('Original Data (Classes are Imbalanced) - \n Patients with / without Sepsis')
plt.show()

from imblearn.over_sampling import RandomOverSampler

y = df['SepsisLabel']
X = df.drop('SepsisLabel', axis=1)

X, y = RandomOverSampler().fit_resample(X, y)

label_counts = y.value_counts()
fig, ax = plt.subplots()
labels = ['Sepsis', 'Non-Spesis']
ax.pie(label_counts, labels=labels, autopct='%1.1f%%')
plt.title('Resampled Data (Class Balancing) -\n Patients with / without Sepsis')
plt.show()

df.shape
X.shape
y.shape

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
y

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

ABC = AdaBoostClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()

ABC.fit(X_train, y_train)
GBC.fit(X_train, y_train)
RFC.fit(X_train, y_train)


#Making Predictions

ypred_proba_abc = ABC.predict_proba(X_test)
ypred_proba_gbc = GBC.predict_proba(X_test)
ypred_proba_rfc = RFC.predict_proba(X_test)

scored_abc = ABC.score(X_test, y_test)
scored_gbc = GBC.score(X_test, y_test)
scored_rfc = RFC.score(X_test, y_test)


ypred_abc = ABC.predict(X_test)
ypred_gbc = GBC.predict(X_test)
ypred_rfc = RFC.predict(X_test)

log_loss_abc = log_loss(y_test, ypred_abc)
log_loss_gbc = log_loss(y_test, ypred_gbc)
log_loss_rfc = log_loss(y_test, ypred_rfc)

print(f'Scoring for our three ensemble learning classifiers: AdaBoost, GradientBoosting, RandomForest\nRFC: {100*scored_rfc:1.4f}% accuracy,\nGBC: {100*scored_gbc:1.4f}% accuracy,\nABC: {100*scored_abc:1.4f}% accuracy\n\n')

my_vals = pd.DataFrame(data = [scored_rfc, scored_gbc, scored_abc], index =['RFC','GBC','ABC'], columns =['Accuracy'])
my_vals['Log Loss'] = [log_loss_rfc, log_loss_gbc, log_loss_abc]


print(f'AdaBoost  - log loss:{log_loss_abc}\nxxxxxxxxxxx\n', classification_report(y_test, ypred_abc), '\n\n')
print(f'GradientBoosting  - log loss: {log_loss_gbc}\nxxxxxxxxxxx\n',classification_report(y_test, ypred_gbc), '\n\n')
print(f'RandomForest - log Loss: {log_loss_rfc}\nxxxxxxxxxxx\n',classification_report(y_test, ypred_rfc), '\n\n')


sns.barplot(my_vals, x ='Accuracy', y = my_vals.index)
ax = sns.barplot(my_vals, x ='Log Loss', y = my_vals.index)
