import pandas as pd
def cold_starters(amount = 10):
    
    """
    Making a recommendation list for Cold starters. 
    Ranks movies from the last two years based on their rating and rating amount
    it cuts movies with a rating of less than 3.5 and a rating amount of less than 0.75% of the total of that year. 
    
    Parameters
    ----------
    X:  integer
        The number of movies to be recommended. If 0 it returns the whole ranked table
        (Default is 10)
    Returns
    ----------
    pandas.Dataframe
        The recommended movies and their score.
        Index: movieId
        rows: movies 
        Columns:
            title:
                String
                title(release year)
            genres:
                String 
                genres1|genre2|...|genren
            score
                Float
                the calculated score between 0-1
        
    """
    # Function to normalize data
    def Min_Max(obj):
        norm_obj =  (obj - obj.min()) / (obj.max() - obj.min())
        return norm_obj
    
    # Make a df from the rating database, grouped by movie Id with an average rating and amount of rating
    df_rating = pd.read_csv('data/ratings.csv').groupby('movieId').agg({'rating': ['mean', 'count']})
    # Flatten the multi-index columns to have a single level of columns
    df_rating.columns = ['avg_rating', 'rating_count']
    
    # Getting the source data frame with the release year column
    from data_script.Preprocess_Content_Based import get_releaseyear
    df_movies = pd.read_csv('data/movies.csv')
    df_movies['releaseyear'] = get_releaseyear(df_movies['title']).astype(int)
    
    # Filtering movies from the last two years
    df_movies = df_movies[df_movies['releaseyear'] >= df_movies['releaseyear'].max()-1].set_index('movieId')
        
    if amount == 0:
        # Returning the ranked table without pre-filtering the data
        
        # Normalize the average rating
        df_rating.avg_rating= Min_Max(df_rating.avg_rating)
        
        # Merge the two data frames based on an index
        df= pd.merge(df_movies, df_rating,  left_index=True, right_index=True)
    
        # Normalize the rating_count column within each year
        df['rating_count'] = df.groupby('releaseyear', group_keys=False)['rating_count'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
        # Normalize release year to be labeled to use it within the normalized score
        df['releaseyear'] = Min_Max(df.releaseyear)
        
        # Calculating the score based on avg rating, rating amount and release year using weights
        df['score'] = df.avg_rating + 1.25 * df.rating_count + 0.5 * df.releaseyear
        df['score'] = Min_Max(df['score'])
        # Dropping the unnecessary columns
        df = df.drop(['avg_rating', 'rating_count', 'releaseyear'], axis = 1)
        
        # Returning the ranked table
        return df.sort_values(by='score', ascending=False)
        
    else: 
        # Returning the recommendation list by pre-filtering the data
        
        # Filtering out low ratings
        df_rating = df_rating[(df_rating.avg_rating >= 3)]
        # Normalize the average rating
        df_rating.avg_rating= Min_Max(df_rating.avg_rating)
        
        
        # Merge the two data frames based on an index
        df= pd.merge(df_movies, df_rating,  left_index=True, right_index=True)
        
        # Normalize the rating_count column within each year
        df['rating_count'] = df.groupby('releaseyear', group_keys=False)['rating_count'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # Filtering the top 0.25% of rating count of each year independently
        df = df[df.rating_count >= df.rating_count.quantile(0.75)]
        
        # Normalize release year to be labeled to use it within the normalized score
        df['releaseyear'] = Min_Max(df.releaseyear)
        
        # Calculating the score based on avg rating, rating amount and release year using weights
        df['score'] = df.avg_rating + 1.25 * df.rating_count + 0.5 * df.releaseyear
        df['score'] = Min_Max(df['score'])
        # Dropping the unnecessarily columns
        df = df.drop(['avg_rating', 'rating_count', 'releaseyear'], axis = 1)
        
        # Returning a recommendation list with the predefined amount of recommendation
        return df.sort_values(by='score', ascending=False).head(amount)