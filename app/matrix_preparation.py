import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import numpy as np

class Dataprep:
    def __init__(self):
        self.reco_df = pd.read_csv('datasetreco.csv',sep=';')
    def data_preparation(self):
        
        self.reco_df["score"]=self.reco_df["p_views"]+(self.reco_df["p_carts"]*5)+(self.reco_df["p_purchases"]*10)

        main = self.reco_df.groupby('category').max('score')[['score']].reset_index()

        self.main_df = self.reco_df.merge(main,on=['category'])
        self.main_df['score_norm'] = self.main_df['score_x']/self.main_df['score_y']

        self.main_df = self.main_df[self.main_df['user_purchases']>=1]
        self.main_df.drop_duplicates(inplace=True)

        self.data_filtered_last_purchase = self.main_df.groupby('user_id').max('Date')['product_id'].reset_index()

        return self.data_filtered_last_purchase, self.main_df

    def dimensionality_reduction(self):
        self.pivot_data=pd.pivot_table(self.main_df, values='score_norm', index=['product_id'],
                       columns=['user_id'],fill_value=0)
        svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42) #reduction de dimensionnalité a 10 colonnes

        self.decomposed_matrix= svd.fit_transform(self.pivot_data)

        return self.pivot_data
    def correlation_matrix(self):
        self.correlation_matrix = np.corrcoef(self.decomposed_matrix)

        return self.correlation_matrix
