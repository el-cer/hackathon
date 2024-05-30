import pandas as pd 
from sklearn.decomposition import TruncatedSVD
import numpy as np


class Users:
    def __init__(self):
        self.reco_df = pd.read_csv('datasetreco.csv',sep=';')
    def get_data(self):
        self.main_df = self.reco_df[self.reco_df['user_purchases']>=1]
        self.main_df.drop_duplicates(inplace=True)
        groupby_df = self.main_df.groupby('user_id').max('Date_x')['product_id'].reset_index()
        total_user_id = groupby_df.loc[:10,'user_id']
        return total_user_id


class Model:
    def __init__(self,user_id,data_filtered_last_purchase, main_df, pivot_data, correlation_matrix):
        self.reco_df = pd.read_csv('datasetreco.csv',sep=';')
        self.user_id = int(user_id)
        self.data_filtered_last_purchase, self.main_df, self.pivot_data, self.correlation_matrix = data_filtered_last_purchase, main_df, pivot_data, correlation_matrix

    def get_recommendation(self):

        product_ID = self.data_filtered_last_purchase.loc[self.data_filtered_last_purchase['user_id']==self.user_id,'product_id'].values[0]
        product_names = list(self.pivot_data.index)
        product_ID_index = product_names.index(product_ID)

        correlation_product_ID = self.correlation_matrix[product_ID_index]

        Recommend = list(self.pivot_data.index[correlation_product_ID > 0.9])
        
        list_of_tuple = [(index, correlation_product_ID[product_names.index(index)]) for index in Recommend]

        
        sorted_data = sorted(list_of_tuple, key=lambda x: x[1],reverse=True)
        number_product_to_recommend = 5
        Recommend_list = []

        
        for i in sorted_data:
            Recommend_list.append(i[0])
        Recommend = Recommend_list[:number_product_to_recommend+1]


        list_product = [self.main_df.loc[self.main_df['product_id'] == i, ['category', 'subcategory', 'subsubcategory', 'price', 'p_purchases', 'product_id']] for i in Recommend]

        product_buyed = self.main_df.loc[self.main_df['product_id']==product_ID,['category','subcategory','subsubcategory','price','p_purchases','product_id']].drop_duplicates()

        list_product = Model.clean_name(list_product)

        return Recommend,list_product
    @staticmethod
    def clean_name(product_name_recommanded):
        product_recommended = [i.drop_duplicates().values for i in product_name_recommanded]

        return product_recommended


        

        