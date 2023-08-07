import pandas as pd
def cold_starters(amount = 10):
    #make a df from the rating database, grouped by movie Id with average rating and amount of rating
    df_rating = pd.read_csv('data/ratings.csv').groupby('movieId').agg({'rating': ['mean', 'count']})
    # Flatten the multi-index columns to have a single level of columns
    df_rating.columns = ['avg_rating', 'rating_count']
    # filtering out low view and low rating amount movies
    df_rating = df_rating[(df_rating.avg_rating >= 3)]
    # function to normalize data
    def Min_Max(obj):
        norm_obj =  (obj - obj.min()) / (obj.max() - obj.min())
        return norm_obj
    #normalize 
    df_rating.avg_rating= Min_Max(df_rating.avg_rating)
    # getting the source dataframe with release year column
    from data_script.Preprocess_Content_Based import conditions
    df_movies = pd.read_csv('data/movies.csv')
    df_movies['releaseyear'] = df_movies['title'].apply(conditions).fillna(1993).astype(int)
    #fitlering movies from the last two years
    df_movies = df_movies[df_movies['releaseyear'] >= df_movies['releaseyear'].max()-1].set_index('movieId')
    #merge the two dataframes based on index
    df= pd.merge(df_movies, df_rating,  left_index=True, right_index=True)
    # normalize the rating_count column within each year
    df['rating_count'] = df.groupby('releaseyear', group_keys=False)['rating_count'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # filtering out the firsts three quartils
    df = df[df.rating_count >= df.rating_count.quantile(0.75)]
    #normalize release year
    df['releaseyear'] = Min_Max(df.releaseyear)
    #calculating the score based on avg rating, rating amount and realese year using weights
    df['score'] = df.avg_rating + 1.25 * df.rating_count + 0.5 * df.releaseyear
    df['score'] = Min_Max(df['score'])
    #droping the unnecesserie column
    df = df.drop(['avg_rating', 'rating_count', 'releaseyear'], axis = 1)

    return df.sort_values(by='score', ascending=False).head(amount)