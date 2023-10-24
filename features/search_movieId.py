import pandas as pd
def search_movieId(given_movies):
    """
    get the Ids of movies for a list of movies with their title and release year.
    
    Parameters
    ----------
    given_movies:  list of movie title.
        ['movie Title_1',...,'movie Title_n']
    
    Returns
    ----------
    given_movies_ids: list of integers
        A list of the movieIds of the given movies in the same order.
        [movieId_1,...,movieId_n]            
    """
    df_movies = pd.read_csv('data/external/movies.csv')
    given_movies_ids = []
    for title in given_movies:
        given_movie_id = df_movies[
            df_movies.title == title].movieId.item()
        given_movies_ids.append(given_movie_id)
    return given_movies_ids
