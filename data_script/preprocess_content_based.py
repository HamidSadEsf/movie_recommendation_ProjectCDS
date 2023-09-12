import numpy as np
import pandas as pd

def get_releaseyear(X):
    
    """
    Get the release year from the titles column of the Movielens movies.csv
    
    Parameters
    ----------
    X:  pandas.DataSeries
        The 'title' column of the Movielens movies.csv
    Returns
    ----------
    pandas.DataSeries
        The extracted release years from the 'title' column as Strings
        
    """
    
    # Conditions to extract the release year string from the title, depending on how the string ends. 
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
    # Applying the conditions to the title column, while filling the only NaN with 1993.
    # The release years for this particular movie were looked up on the internet
    
    return X.apply(conditions).fillna(1993)


def get_df_ContBaseRec():
    
    """
    Preprocessing the dataset for the content-based recommendation system.
    It combines the genome-scores.csv and genome-tags.csv into a data frame.
    The rows represent the Movies, thus the index corresponds to the Movie Ids. 
    Columns represent the tags, thus the column names correspond to the Tag Ids.
    The Value in the cells is the relevance of the Tags for the respective movies
    
    Parameters
    ----------
    no Params
    
    Returns
    ----------
    pandas.DataFrame
        The preprocessed dataset to be used in the content-based recommendation system
        Index: movieId
        Columns: tagIds
        each value in the cells represents the relevance of the tag for the corresponding movie 
    """
    
    # Importing the genomes score and genome tags data sets as pandas.DataFrame
    df_gscore = pd.read_csv('data/external/genome-scores.csv')
    df_gtags = pd.read_csv('data/external/genome-tags.csv')

    # Creating the genome dataset

    df_gtagscore = df_gtags.merge(df_gscore, how='right', on='tagId').drop('tag', axis=1)
    df_gtagscore.drop_duplicates(subset=['tagId', 'movieId'], inplace=True)

    # Pivoting the data frame
    df_gtagscore = df_gtagscore.pivot(index='movieId', columns='tagId', values='relevance')

    # Adding release years as columns
    df_movies = pd.read_csv('data/external/movies.csv')
   
    # Getting and attaching the release year as a new column
    df_movies['releaseyear'] = get_releaseyear(df_movies['title'])
    
    # Normalize the release year values
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    scaler = MinMaxScaler()
    df_movies.releaseyear = scaler.fit_transform(df_movies[['releaseyear']])
    
    # Adding genres as columns
    df_movies.genres = df_movies.genres.str.split('|')
    dummies = pd.get_dummies(df_movies.genres.apply(pd.Series).stack()).sum(level=0)
    df_movies = pd.concat([df_movies, dummies], axis=1).drop('genres', axis=1)

    # Merging the tag score dataset and the new dataset with release year and genres to a new database
    df_movies = df_movies.drop('title', axis=1)
    df_ContBaseRec = pd.merge(df_gtagscore, df_movies, how='inner', on='movieId').set_index('movieId')
    df_ContBaseRec.columns = df_ContBaseRec.columns.astype(str)
    
    # Save the final dataset to disk
    df_ContBaseRec.to_csv('./data/processed/df_ContBaseRec.csv', index=False)
    
    return df_ContBaseRec