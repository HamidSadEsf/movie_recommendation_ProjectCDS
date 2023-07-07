import numpy as np
import pandas as pd

df_gscore = pd.read_csv('genome-scores.csv')
df_gtags = pd.read_csv('genome-tags.csv')
df_links = pd.read_csv('links.csv')
df_movies = pd.read_csv('movies.csv')
df_rating = pd.read_csv('ratings.csv')
df_tagspd = pd.read_csv('tags.csv')

# Creating the genome dataset

df_gtagscore=df_gtags.merge(df_gscore, how='right', on='tagId').drop('tag', axis =1)
df_gtagscore.drop_duplicates(subset=['tagId', 'movieId'],inplace=True)

# pivoting the dataframe
df_gtagscore = df_gtagscore.pivot(index = 'movieId', columns = 'tagId', values='relevance')

# adding release years as columns
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
df_movies['releaseyear'] = df_movies['title'].apply(condition).fillna(1993)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
df_movies =df_movies.drop('title', axis =1)
scaler = MinMaxScaler()
df_movies.releaseyear = scaler.fit_transform(df_movies[['releaseyear']])

# adding genres as columns
df_movies.genres= df_movies.genres.str.split('|')
dummies = pd.get_dummies(df_movies.genres.apply(pd.Series).stack()).sum(level=0)
df_movies = pd.concat([df_movies, dummies], axis=1).drop('genres', axis=1)

# merging the tag score dataset and the new dataset with release year and genres to a new database
df_ContBaseRec = pd.merge(df_gtagscore, df_movies, how='inner', on='movieId').set_index('movieId')
df_ContBaseRec.columns = df_ContBaseRec.columns.astype(str)

# Clustering
# importing the library

from sklearn.cluster import KMeans

# training the Model

kmeans = KMeans(n_clusters=18, n_init='auto')
kmeans.fit(df_ContBaseRec)
labels = kmeans.labels_
df_movies = pd.read_csv('movies.csv')
df_labeledMovies= pd.DataFrame({'movieId': df_ContBaseRec.index, 'labels': labels}).merge(df_movies, on='movieId').set_index('movieId')

# recommending movies based on several movies

df_movies = pd.read_csv('movies.csv')


def recommendation(given_movies, number_of_recommendations, df=df_ContBaseRec, lambda_val=np.random.rand()):
    given_movies_ids = []
    for title, year in given_movies:
        given_movie_id = df_movies[
            df_movies.title.str.contains(r'\b{}\b.*\b{}\b.$'.format(title, year), case=False)].movieId.item()
        given_movies_ids.append(given_movie_id)

    # calculating the mean point (vector) of the given movies
    mean_point = df.loc[given_movies_ids].mean()

    # movie recommendation with cousin similarity
    from sklearn.metrics.pairwise import cosine_similarity
    # Get the relevance vector of the given movie
    given_movie_vector = mean_point.values.reshape(1, -1)
    # Calculate cosine similarity between the given movie and all other movies
    similarity_scores = cosine_similarity(df.values, given_movie_vector)
    # calculate the dissimilarity score dictionary
    dissimilarity_scores = similarity_scores - 1

    # distances as relevance score with nearest neighbors distances
    import sklearn.neighbors
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=df.shape[0])
    nn.fit(df)
    # getting the distances of any point in the df to mean point
    distances, indices = nn.kneighbors(mean_point.values.reshape(1, -1))
    # normalize the data between 0 and 1
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    # calculate a relevance_score
    relevance_score = (1 - distances)

    # Diversification
    score = pd.Series((relevance_score + (lambda_val * dissimilarity_scores.reshape(1, -1)))[0], name="score")

    df_scored = pd.concat([df_labeledMovies.reset_index(names='movieId'), score.reset_index(drop=True)],
                          axis=1).set_index('movieId')

    # Remove the given movie from the list of recommendation
    df_scored = df_scored.drop(given_movies_ids)
    # deleting the movies which are not in the same clusters as the given movies
    given_movies_clusters = df_labeledMovies.loc[given_movies_ids, "labels"].unique()
    df_scored = df_scored[df_scored["labels"].isin(given_movies_clusters)]

    # getting the movies from the labled set which shows the title and label
    recommendations = df_scored.sort_values('score', ascending=False).head(number_of_recommendations)
    return recommendations
