import pandas as pd
def cold_starters(df, amount = 10):
    def Min_Max(obj):
        nor_obj =  (obj - obj.min()) / (obj.max() - obj.min())
        return nor_obj
    df['score'] = Min_Max(df.avg_rating) + Min_Max(df.rating_amount) + Min_Max(df.realease_year)
    return df.sort_values(by='score').head(10)