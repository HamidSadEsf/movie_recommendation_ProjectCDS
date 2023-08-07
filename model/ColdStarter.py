import pandas as pd
def cold_starters(amount = 10):
    #make a df from the rating database, grouped by movie Id with average rating and amount of rating
    df_rating = pd.read_csv('data/ratings.csv').groupby('movieId').agg({'rating': ['mean', 'count']})
    # Flatten the multi-index columns to have a single level of columns
    df_rating.columns = ['avg_rating', 'rating_count']
    # filtering out low view and low rating amount movies
    df_rating = df_rating[(df_rating.avg_rating > 4) & (df_rating.rating_count > df_rating.rating_count.quantile(0.98))]
    # function to normalize data
    def Min_Max(obj):
        norm_obj =  (obj - obj.min()) / (obj.max() - obj.min())
        return norm_obj
    #normalize 
    df_rating.avg_rating= Min_Max(df_rating.avg_rating)
    df_rating.rating_count= Min_Max(df_rating.rating_count)

    # getting the source dataframe
    from data_script.Preprocess_Content_Based import get_df
    df = get_df()
    df= pd.merge(df, df_rating, left_index=True, right_index=True)
    # filtering out movies with older release year
    df = df[df.releaseyear > df.releaseyear.quantile(0.95)]
    #calculating the score based on avg rating, rating amount and realese year using weights
    df['score'] = df.avg_rating + 1.25 * df.rating_count - 1.5 * df.releaseyear
    #searching the movies in the dataframe and return
    df_movies = pd.read_csv('data/movies.csv')
    return df_movies[df_movies['movieId'].isin(df.sort_values(by='score').head(amount-1).index)]