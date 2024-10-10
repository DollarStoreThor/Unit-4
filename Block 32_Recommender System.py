import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statistics
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

#Ratings csv
print('\n ~Ratings Head~ \n', ratings.head(10))
print('\n~Is null~\n',ratings.isnull().sum())
print('\n~dtypes~\n',ratings.dtypes)

ratings = ratings.drop('timestamp', axis=1)

print('\n ~ Ratings Head ~ \n', ratings.head(10))
print('\n~ Is null ~\n',ratings.isnull().sum())
print('\n~ dtypes ~\n',ratings.dtypes)
print('\n~ describe ~\n',ratings.describe())

#Movies csv
print('\n ~ Movies Head ~ \n', movies.head(10))
print('\n~ Is null ~\n',movies.isnull().sum())
print('\n~ dtypes ~\n',movies.dtypes)


#Merging DFs toghter on the movieID feature
df = pd.merge(movies, ratings, on='movieId', suffixes = ('_movies', '_ratings'))

print(df.head())
print(df.shape)

#Datavisulatizations
avg_movie_rating = pd.DataFrame(df.groupby('movieId')['rating'].mean())
avg_movie_rating.head(20)
ravg = avg_movie_rating.apply(lambda x: round(x, 1))

#calculations
overall_ravg = round(ravg['rating'].mean(), 2)
most_ravg = ravg['rating'].mode()[0]
median_ravg = ravg['rating'].median()
print(f'Average rating Average: {overall_ravg}\nMode rating Average: {most_ravg}\nMedian rating Average: {median_ravg}')

##PLOT
fig, ax = plt.subplots()

sns.countplot(data=ravg,  x='rating') 
plt.title("Count of Average User Ratings - \nLeft-Skewed & Whole Numbers Overrepresented")

ax.set_xticks([2, 12, 22, 32, 42])
plt.xlabel('Ratings')
plt.ylabel('Count')


df_genres = pd.DataFrame(df['genres'])
df_genres_counts = pd.DataFrame(pd.get_dummies(data=df_genres).sum())
df_unique_genres = df_genres['genres'].unique()
list_raw_genres = df_unique_genres.tolist()


seperated_generes = {}
for current_movie_genres in list_raw_genres:

    genres = current_movie_genres.split('|')
    for g in genres:
        if g in seperated_generes.keys():
            seperated_generes[g] = seperated_generes[g] + 1
        else:
            seperated_generes[g] = 1
df_gen = pd.DataFrame(seperated_generes, index=range(0, len(seperated_generes))).to_numpy()
identity_matrix = np.identity(df_gen.shape[1])
df_genI = df_gen * identity_matrix

unique_genres = seperated_generes.keys()
# Create a co-occurrence matrix
cooccurrence_matrix = pd.DataFrame(0, index=unique_genres, columns=unique_genres).to_numpy()
# Fill the matrix
for genres in unique_genres:
    for i in range(len(genres)):
        for j in range(len(genres)):
            if i!= j:
                cooccurrence_matrix[i, j] += 1

genre_coocurence = pd.DataFrame(cooccurrence_matrix, index=seperated_generes.keys(), columns=seperated_generes.keys())

plt.figure(figsize=(9,9))
sns.heatmap(data=genre_coocurence, annot=True, fmt=".0f", linewidth=.9, xticklabels=seperated_generes.keys(), yticklabels=seperated_generes.keys())
plt.xticks(rotation=45, ha='right')  # Rotates x-axis labels for better readability
plt.title('Co-occurrence of Genres \n(Min of Two Generes Occur on these films to be Depicted)')
plt.tight_layout()


genre_coocurence = pd.DataFrame(cooccurrence_matrix+df_genI, index=seperated_generes.keys(), columns=seperated_generes.keys())

plt.figure(figsize=(9,9))
sns.heatmap(data=genre_coocurence, annot=True, fmt=".0f", linewidth=.9, robust=True, xticklabels=seperated_generes.keys(), yticklabels=seperated_generes.keys())
plt.xticks(rotation=45, ha='right')  # Rotates x-axis labels for better readability
plt.title('Occurrences of Genres -\n(All Depicted)')
plt.tight_layout()

sns.histplot(data=df , x=df['userId'])

sns.scatterplot(data=df , x=df['userId'], y=(df.groupby('userId')['userId'].count()), hue=df.groupby('userId')['rating'].std(), alpha=0.85)
plt.ylabel('Occurences')
plt.xlabel('User ID')
plt.title('User ID vs Occurences - \nDepicts how a few users do most of the ratings')
ax = plt.subplot(111)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Stdev of Rating')
plt.tight_layout()

sns.scatterplot(data=df , x=df['userId'], y=(df.groupby('userId')['userId'].count()), hue=df.groupby('userId')['rating'].std(), alpha=0.85)
plt.ylabel('Occurences')
plt.xlabel('User ID')
plt.title('User ID vs Occurences - \nDepicts how a few users do most of the ratings')
ax = plt.subplot(111)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Stdev of Rating')
plt.tight_layout()

sns.scatterplot(data=df , x=df['userId'], y =df.groupby('userId')['rating'].mean(), hue=df.groupby('userId')['rating'].std(), alpha=0.85)
plt.ylabel('Average Rating')
plt.xlabel('User ID')
plt.title('User ID vs Average Rating \nDepicts Skewness towards Average: 3.26')
ax = plt.subplot(111)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Stdev of Rating')
plt.tight_layout()

##USER ITEM MATRIX FOR COLLABORATIVE FILTERING
unique_userId = df['userId'].unique()
unique_userId.sort()
unique_userId

unique_movieId = df['movieId'].unique()
unique_movieId.sort()
unique_movieId

##HEAT MAP OF USER VS ITEMS
User_Item_df = ratings.pivot(index='movieId', columns='userId', values = 'rating')
plt.figure(figsize=(9,9))
sns.heatmap(data=User_Item_df, annot=False, robust=True)
plt.xticks(rotation=60, ha='right')  # Rotates x-axis labels for better readability
plt.title("Before NaN Conversions")
plt.tight_layout()

##HEAT MAP OF USER VS ITEMS WITH FILLED NAN CONVERSION TO MEAN USER RATING
User_Item_df = ratings.pivot(index='movieId', columns='userId', values = 'rating')
plt.figure(figsize=(9,9))
sns.heatmap(data=User_Item_df, annot=False, robust=True)
plt.xticks(rotation=60, ha='right')  # Rotates x-axis labels for better readability
plt.title("Before NaN Conversions")
plt.tight_layout()


UBCF_User_Item_df.T
UBCF_User_Item_Pearson = UBCF_User_Item_df.corr(method='pearson', min_periods=1, numeric_only=False)
sns.heatmap(data=UBCF_User_Item_Pearson, annot=False, robust=True)


##HELPER FUNCTION FOR THE USER COLLABORATIVE RATING PREDICTION
def UserBasedCollabFilter(data=pd.DataFrame, user=int, return_self = False, numReturns = int):
    
    if return_self:
        return data[:][user].sort_values(ascending=False)[0:numReturns]
    else:
        return data[:][user].sort_values(ascending=False)[1:numReturns+1]

Fifty_corrs_with_u1 = UserBasedCollabFilter(UBCF_User_Item_Pearson, user=1, return_self=False, numReturns = 50)



## DEFINING OUR USER BASED COLABORITIVE RATING PREDICTION FUNCTION
def userBCFRatingPredict(data=pd.Series, movieId=int, weightedAVG = True,  decimalNums = 1):
    '''
    1. take userId Series data from userBasedCollabFilter method:
    2. loop through the users which were highly correlated with input user from prev method
    3. while looking at the other users, need to pull the rank each user gave the given movieId
    4. perform a weighted average using the correlations in data as the weights
    '''
    score = 0
    weight = 0
    
    if weightedAVG:
        for u in data.index:
            score += UBCF_User_Item_df[u][movieId] * data[u]
            weight += data[u]
        return round((score/weight), decimalNums)
    else:
        for u in data.index:
            score += UBCF_User_Item_df[u][movieId] 
        return round((score/len(data)), decimalNums)

print(f'Predicted Rating for User(1) on Movie(1 - Toy Story (1995)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 1, decimalNums=3)} / 5.0')    
print(f'Predicted Rating for User(1) on Movie(32 - Twelve Monkeys (a.k.a. 12 Monkeys) (1995)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 32, decimalNums=3)} / 5.0')
print(f'Predicted Rating for User(1) on Movie(47 - Seven (a.k.a. Se7en) (1995)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 47, decimalNums=3)} / 5.0')
print(f'Predicted Rating for User(1) on Movie(71 - Fair Game (1995)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 71, decimalNums=3)} / 5.0')
print(f'Predicted Rating for User(1) on Movie(76091 - Mother (Madeo) (2009)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 76091, decimalNums=3)} / 5.0')
print(f'Predicted Rating for User(1) on Movie(79553 - Ip Man 2 (2010)) Based on Top(50) Similar Users ---> {userBCFRatingPredict(data = Fifty_corrs_with_u1, movieId = 79553, decimalNums=3)} / 5.0')





## Perform Item-based Collaborative Filtering
User_Item_df = ratings.pivot(index='movieId', columns='userId', values = 'rating')

plt.figure(figsize=(9,9))
sns.heatmap(data=User_Item_df, annot=False, robust=True)
plt.xticks(rotation=60, ha='right')  # Rotates x-axis labels for better readability
plt.title("Before NaN Conversions")
plt.tight_layout()

User_Item_df = User_Item_df.T
IBCF_User_Item_df = User_Item_df.fillna(User_Item_df.mean())

plt.figure(figsize=(9,9))
sns.heatmap(data=IBCF_User_Item_df, annot=False, robust=True)
plt.xticks(rotation=60, ha='right')  # Rotates x-axis labels for better readability
plt.title("IBCF_User_Item_df -\nAfter NaN Conversion to Item's Mean Rating Score")
plt.tight_layout()

IBCF_User_Item_Pearson = IBCF_User_Item_df.corr(method='pearson', min_periods=1, numeric_only=False)
IBCF_User_Item_Pearson.head()


#HELPER FUNCTION FOR ITME BASED RATING PREDICTION
def ItemBasedCollabFilter(data=pd.DataFrame, movie=int, return_self = False, numReturns = int):
    
    if return_self:
        return data[:][movie].sort_values(ascending=False)[0:numReturns]
    else:
        return data[:][movie].sort_values(ascending=False)[1:numReturns+1]
    

Ten_corrs_with_movie1 = ItemBasedCollabFilter(IBCF_User_Item_Pearson, movie=1, return_self=False, numReturns = 10)
Ten_corrs_with_movie480 = ItemBasedCollabFilter(IBCF_User_Item_Pearson, movie=480, return_self=False, numReturns = 10)
Ten_corrs_with_movie4306 = ItemBasedCollabFilter(IBCF_User_Item_Pearson, movie=4306, return_self=False, numReturns = 10)
Ten_corrs_with_movie5218 = ItemBasedCollabFilter(IBCF_User_Item_Pearson, movie=5218, return_self=False, numReturns = 15)
Ten_corrs_with_movie103688 = ItemBasedCollabFilter(IBCF_User_Item_Pearson, movie=103688, return_self=False, numReturns = 10)



def itemBCFRatingPredict(data=pd.Series, movieId=int, numMovies = 10):
    '''
    1. take movie Series data from itemBasedCollabFilter method:
    2. loop through the movies which were highly correlated with input movieId from prev method
    3. Return the list of numMovies number of movies
    '''
    
    ret_list = []
    for m in data.index:
        ret_list.append(movies.where(movies.movieId == m).dropna()['title'].values[0])
    
    
    return ret_list

print(f'Predicted Similar Movies for Movie(1 - Toy Story (1995)) Based on Top(10) Similar Movies ---> {itemBCFRatingPredict(data = Ten_corrs_with_movie1, movieId=1, numMovies = 10)}\n')    
print(f'Predicted Similar Movies for Movie(480 - Jurassic Park (1993)) Based on Top(10) Similar Movie ---> {itemBCFRatingPredict(data = Ten_corrs_with_movie480, movieId=480, numMovies = 10)}\n')    
print(f'Predicted Similar Movies for Movie(4306 - Shrek (2001)) Based on Top(10) Similar Movie ---> {itemBCFRatingPredict(data = Ten_corrs_with_movie4306, movieId=4306, numMovies = 10)}\n')    
print(f'Predicted Similar Movies for Movie(5218 - Ice Age (2002)) Based on Top(15) Similar Movie ---> {itemBCFRatingPredict(data = Ten_corrs_with_movie5218, movieId=5218, numMovies = 15)}\n')    
print(f'Predicted Similar Movies for Movie(103688 - Conjuring, The (2013)) Based on Top(10) Similar Movie ---> {itemBCFRatingPredict(data = Ten_corrs_with_movie103688, movieId=103688, numMovies = 10)}\n')    




