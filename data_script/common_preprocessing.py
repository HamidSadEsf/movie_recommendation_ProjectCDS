import preprocess_collaborative
# import clustering_collaborative #TODO

print("Prerocessing for Content-Based Recommender...")
from preprocess_content_based import get_df_ContBaseRec
df = get_df_ContBaseRec()

from clustering_content_based import get_labeledMovies
get_labeledMovies(df, clusters=15)
