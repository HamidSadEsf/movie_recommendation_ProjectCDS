import pandas as pd
def search_movieId(given_movies):
    # import movie database
    df_movies = pd.read_csv('data/movies.csv')
    # given _movies: a list of [['movie Title','release year']] as string
    given_movies_ids = []
    for title, year in given_movies:
        given_movie_id = df_movies[
            df_movies.title.str.contains(r'\b{}\b.*\b{}\b.$'.format(title, year), case=False)].movieId.item()
        given_movies_ids.append(given_movie_id)
    return given_movies_ids
