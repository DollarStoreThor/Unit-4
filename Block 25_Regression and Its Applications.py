### IMPORTING SOME NECESSARY LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

### READING IN THE DATASET
df  = pd.read_csv('cuisines.csv')
df.head()

print(df.info())
print("# of NA's: ", df.isna().sum().sum())
print("Size of df: ", df.size)

df.drop('Unnamed: 0', axis=1, inplace = True)

df.cuisine.value_counts().plot.barh()

df_totals = df.groupby('cuisine').sum()

df_chinese = df_totals.loc['chinese']
df_indian = df_totals.loc['indian']
df_japanese = df_totals.loc['japanese']
df_thai = df_totals.loc['thai']
df_korean = df_totals.loc['korean']

print('df_chinese shape:', df_chinese.shape, '\n')
print('df_indian shape:', df_indian.shape, '\n')
print('df_japanese shape:', df_japanese.shape, '\n')
print('df_thai shape:', df_thai.shape, '\n')
print('df_korean shape:', df_korean.shape, '\n')

### ANALYZING INGREIDENT USAGE
def create_ingredient(data=pd.DataFrame, t='int16', dev=False):
    
    #create a mask to remove all occurances of 'no usages in given dataframe'
    new_data = data.mask(data == 0).dropna()
    new_data = new_data.astype(t)
    new_data = pd.DataFrame(new_data)
    
    if dev:
        print('\n', new_data)
        new_data.head()    
    
    return new_data
  
japanese_ingredients = create_ingredient(data=df_japanese, dev=True)
indian_ingredients = create_ingredient(data=df_indian, dev=True)
chinese_ingredients = create_ingredient(data=df_chinese, dev=True)
thai_ingredients = create_ingredient(data=df_thai, dev=True)
korean_ingredients = create_ingredient(data=df_korean, dev=True)


f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 30), sharex=False)

j_10 = japanese_ingredients.sort_values(by='japanese', ascending=False).head(10)
i_10 = indian_ingredients.sort_values(by='indian', ascending=False).head(10)
c_10 = chinese_ingredients.sort_values(by='chinese', ascending=False).head(10)
t_10 = thai_ingredients.sort_values(by='thai', ascending=False).head(10)
k_10 = korean_ingredients.sort_values(by='korean', ascending=False).head(10)



sns.barplot(data=j_10, x=j_10.index , y=j_10.values.flatten(), ax=ax1)
ax1.set_ylabel("Japanese")

sns.barplot(data=i_10, x=i_10.index , y=i_10.values.flatten(), ax=ax2)
ax2.set_ylabel("Indian")

sns.barplot(data=c_10, x=c_10.index , y=c_10.values.flatten(), ax=ax3)
ax3.set_ylabel("Chinese")

sns.barplot(data=t_10, x=t_10.index , y=t_10.values.flatten(), ax=ax4)
ax4.set_ylabel("Thai")

sns.barplot(data=k_10, x=k_10.index , y=k_10.values.flatten(), ax=ax5)
ax5.set_ylabel("Korean")


features_df = df.drop(['cuisine','garlic', 'cayenne', 'ginger', 'rice', 'soy_sauce', ],  axis=1)
label_df = df.cuisine
features_df

from imblearn.over_sampling import SMOTE

# Apply SMOTE to oversample the minority class
oversample = SMOTE(random_state=42)
transformed_feature_df, transformed_label_df = smote.fit_resample(features_df, label_df)

from collections import Counter

print("Original dataset shape:", Counter(label_df), )
print("Resampled dataset shape:", Counter(transformed_label_df))

### TRAINING A LOGISTIC REGRESSION MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(transformed_feature_df, transformed_label_df, test_size = 0.3)

lr = LogisticRegression(multi_class='ovr', solver='liblinear')

model = lr.fit(X_train, np.ravel(y_train))
accuracy = model.score(X_test, y_test)

print(f'Model accurcay: {accuracy:.3f}')


### MAKING PREDICTIONS
print(X_test.iloc[50], y_test.iloc[50])

test = X_test.iloc[50].values.reshape(-1,1).T


proba = model.predict_proba(test)
classes = model.classes_

result_df = pd.DataFrame(data=proba, columns=classes)
top_res = result_df.T.sort_values(by=[0], ascending=False)
top_res

### EVALUATE THE MODEL
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
