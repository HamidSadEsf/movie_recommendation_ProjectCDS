import pandas as pd
from ..model.ContentBasedRec import ContentBasedRecommender 
def get_CBMatrix():
    CBR = ContentBasedRecommender()
    CBR.load_database()
    final_rating = pd.read_csv(r'data\processed\final_ratings.csv')
    CBMatrix = pd.DataFrame(index = CBR.df.index).sort_index()
    for userid in final_rating['userId'].unique()[:20]:
        cbr = CBR.recommendation(userid).set_index('movieId').sort_index()
        CBMatrix[userid]= cbr.score
    CBMatrix = CBMatrix.transpose().rename_axis("userId", axis=0).rename_axis("movieId", axis=1)
    CBMatrix.to_csv('./data/processed/CBMatrix.csv', index_label="userId")

    return CBMatrix