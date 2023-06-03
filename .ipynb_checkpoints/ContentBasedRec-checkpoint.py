import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df_gscore = pd.read_csv('genome-scores.csv')
df_gtags = pd.read_csv('genome-tags.csv')
df_links = pd.read_csv('links.csv')
df_movies = pd.read_csv('movies.csv')
df_rating = pd.read_csv('ratings.csv')
df_tagspd = pd.read_csv('tags.csv')

## Creating the genome dataset

df_gtagscore=df_gtags.merge(df_gscore, how='right', on='tagId').drop('tag', axis =1)
df_gtagscore.drop_duplicates(subset=['tagId', 'movieId'], inplace=True)

# pivoting the datafram
df_gtagscore = df_gtagscore.pivot(index = 'movieId', columns = 'tagId', values='relevance')

## adding realease year and genres as columns

df_movies.genres= df_movies.genres.str.split('|')
dummies = pd.get_dummies(df_movies.genres.apply(pd.Series).stack()).sum(level=0)
df_movies = pd.concat([df_movies, dummies], axis=1).drop('genres', axis=1)

from sklearn.preprocessing import MinMaxScaler
df_movies.drop('title', axis =1)

df_movies = pd.read_csv('movies.csv')
mask = df_movies['title'].str.contains('09–')

df_movies = pd.read_csv('movies.csv')
def condition(x):
    if x[-2:]=='a)':
        return np.nan
    elif x[-2:]=='l)':
        return np.nan
    elif x[-3:-1]=='7-':
        return 2007
    elif x[-4:-2]=='9–':
        return 2009
    elif x[-2:]=='))':
        return x[-6:-2]
    elif x[-1:]==')':
        return x[-5:-1]
    elif x[-1:]==' ':
        return x[-6:-2]
    else:
        return np.nan
df_movies['releaseyear'] = df_movies['title'].apply(condition)

scaler02 = MinMaxScaler()
df_movies.releaseyear = scaler02.fit_transform(df_movies[['releaseyear']])

df_ContBaseRec = pd.merge(df_gtagscore, df_movies, how='inner', on='movieId').set_index('movieId')

df_ContBaseRec.info()