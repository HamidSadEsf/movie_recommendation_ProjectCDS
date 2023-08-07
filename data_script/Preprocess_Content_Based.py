import numpy as np
import pandas as pd

def conditions(x):
        if x[-2:] == 'a)':
            return np.nan
        elif x[-2:] == 'l)':
            return np.nan
        elif x[-3:-1] == '7-':
            return 2007
        elif x[-4:-2] == '9–':
            return 2009
        elif x[-2:] == '))':
            return x[-6:-2]
        elif x[-1:] == ')':
            return x[-5:-1]
        elif x[-1:] == ' ':
            return x[-6:-2]
        else:
            return np.nan


def get_df():
    df_gscore = pd.read_csv('data/genome-scores.csv')
    df_gtags = pd.read_csv('data/genome-tags.csv')

    # Creating the genome dataset

    df_gtagscore = df_gtags.merge(df_gscore, how='right', on='tagId').drop('tag', axis=1)
    df_gtagscore.drop_duplicates(subset=['tagId', 'movieId'], inplace=True)

    # pivoting the dataframe
    df_gtagscore = df_gtagscore.pivot(index='movieId', columns='tagId', values='relevance')

    # adding release years as columns
    df_movies = pd.read_csv('data/movies.csv')
   
    
    df_movies['releaseyear'] = df_movies['title'].apply(conditions).fillna(1993)
    
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    scaler = MinMaxScaler()
    df_movies.releaseyear = scaler.fit_transform(df_movies[['releaseyear']])
    
    # adding genres as columns
    df_movies.genres = df_movies.genres.str.split('|')
    dummies = pd.get_dummies(df_movies.genres.apply(pd.Series).stack()).sum(level=0)
    df_movies = pd.concat([df_movies, dummies], axis=1).drop('genres', axis=1)

    # merging the tag score dataset and the new dataset with release year and genres to a new database
    df_movies = df_movies.drop('title', axis=1)
    df_ContBaseRec = pd.merge(df_gtagscore, df_movies, how='inner', on='movieId').set_index('movieId')
    df_ContBaseRec.columns = df_ContBaseRec.columns.astype(str)
    return df_ContBaseRec