import pandas as pd
def search_movieId(given_movies):
    """
    get the Ids of movies for a list of movies with their title and release year
    
    Parameters
    ----------
    given_movies:  list of lists of two strings
        [['movie Title_1','release year_2'],...,['movie Title_n','release year_n']]]
    
    Returns
    ----------
    given_movies_ids: list of integers
        A list of the movieIds of the given movies in the same order
        [movieId_1,...,movieId_n]            
    """
    df_movies = pd.read_csv('data/movies.csv')
    given_movies_ids = []
    for title, year in given_movies:
        given_movie_id = df_movies[
            df_movies.title.str.contains(r'\b{}\b.*\b{}\b.$'.format(title, year), case=False)].movieId.item()
        given_movies_ids.append(given_movie_id)
    return given_movies_ids
