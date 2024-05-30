import pandas as pd 
import numpy as np

class Users:
    def __init__(self):
        self.reco_df = pd.read_csv('datasetreco.csv',sep=';')
        
    def get_data(self):
        self.main_df = self.reco_df[self.reco_df['user_purchases']>=1]
        self.main_df.drop_duplicates(inplace=True)
        groupby_df = self.main_df.groupby('user_id').max('Date_x')['product_id'].reset_index()
        total_user_id = groupby_df.loc[:15,'user_id']
        return total_user_id


class Model:
    def __init__(self,user_id,data_filtered_last_purchase, main_df, pivot_table, correlation_matrix):
        self.reco_df = pd.read_csv('datasetreco.csv',sep=';')
        self.user_id = int(user_id)
        self.data_filtered_last_purchase, self.main_df, self.pivot_data, self.correlation_matrix = data_filtered_last_purchase, main_df, pivot_table, correlation_matrix

    

    def get_recommendation_score(self,product_ID):

        product_names = list(self.pivot_data.index)
        product_ID_index = product_names.index(product_ID)

        correlation_product_ID = self.correlation_matrix[product_ID_index]
        return correlation_product_ID
    def get_unique_product(self,n_event=3):
        
        user_data=self.reco_df.loc[self.reco_df['user_id'] == self.user_id]
        

        filter_reco_df=user_data[["Date","product_id"]]
        unique_product=filter_reco_df["product_id"].unique()

        product_list=[]
        for user_product_id in unique_product:

            get_recent_product=filter_reco_df.loc[filter_reco_df["product_id"]==user_product_id]
            get_recent_product=get_recent_product.loc[get_recent_product["Date"]==get_recent_product.Date.max()].iloc[0]

            product_list.append((get_recent_product["Date"],get_recent_product["product_id"]))

        product_list = sorted(product_list, key=lambda x: x[0],reverse=False)
        product_list=[x[1] for x in product_list]
        max_product=min(len(product_list),n_event)

        return product_list[0:max_product]

    def get_recommendation(self,n_event=3,nb_recomendation=5):
        product_list= self.get_unique_product(n_event)
        recomendation_list=[]
        correlation_product_ID= self.get_recommendation_score(product_list[0])

        recomendation_list=correlation_product_ID

        for product_index in range(1,len(product_list)):
            correlation_product_ID= self.get_recommendation_score(product_list[product_index])
            recomendation_list= np.add(correlation_product_ID,recomendation_list)

        list_of_tuple=[]
        product_names = list(self.pivot_data.index)


        for index in range(len(recomendation_list)):
            list_of_tuple.append((product_names[index],recomendation_list[index]))

        sorted_data = sorted(list_of_tuple, key=lambda x: x[1],reverse=True)

        recommend=[x[0] for x in sorted_data]
        recommend=recommend[:nb_recomendation]
        product_list=[]

        product_recommended = [self.main_df.loc[self.main_df['product_id'] == i, ['category', 'subcategory', 'subsubcategory', 'price', 'p_purchases', 'product_id']] for i in recommend]

        product_recommended = Model.clean_name(product_recommended)
        

        return recommend,product_recommended
    @staticmethod
    def clean_name(product_name_recommanded):
        product_recommended = [i.drop_duplicates().values for i in product_name_recommanded]

        return product_recommended