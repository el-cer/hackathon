import streamlit as st

import model_reco_v1 
import model_reco_v2
from matrix_preparation import Dataprep
# Style
user = model_reco_v2.Users()
list_of_user = user.get_data()

mp = Dataprep()
data_filtered_last_purchase, main_df= mp.data_preparation()
pivot_table = mp.dimensionality_reduction()
correlation_matrix = mp.correlation_matrix()

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Titre de l'application
st.title('Recommandations par client')

# Titre de l'application

# ajoute les users, tu peux mettre une liste ou un df 


list_of_model_version = ["V1", "V2"]

model_input = st.selectbox("", options=list_of_model_version)
# input tu mets la list dans la selectbox
text_input = st.selectbox("", options=list_of_user)

# Afficher les entrées
st.write('produits recommandé pour l\'utilisateur ', text_input)
if model_input=='V1':
    model = model_reco_v1.Model(text_input,data_filtered_last_purchase, main_df, pivot_table, correlation_matrix)
else:
    model = model_reco_v2.Model(text_input,data_filtered_last_purchase, main_df, pivot_table, correlation_matrix)
# Get recommendations
Recommend, list_product_recommended = model.get_recommendation()


list_of_recommended_item = [list_product_recommended]
#product_recommended[0][0][3] access to idem
# list_of_tuple = [(index, correlation_product_ID[product_names.index(index)]) for index in Recommend]

# print(list_of_recommended_item)

if list_of_recommended_item != None:
    for item in list_of_recommended_item:
        for item_number in range(0, 5):
            categories = ', '.join(item[item_number][0][0:3])
            html_content = f"""
            <style>
            .custom-div {{
                background-color: #E3E3E3;
                padding: 10px;
                border-radius: 12px;
                margin-bottom:12px;
            }}

            .custom-div p {{
                color:#3A3A3C;
            }}

            #product_id{{
                font-size:24px;
                font-weight: 700;
            }}

            #categories{{
                font-size:20px;
            }}

            #price {{
                font-size:25px;
                color: #26E8A0;
            }}

            #product_sell{{
                font-size:18px;
                color: grey;
                
            }}

            </style>
            <div class="custom-div">
                <p id="product_id">Product id : {item[item_number][0][5]}</p>
                <p id="categories">Categories : {categories}</p>
                <p id="product_sell">Nombre de produit vendu en 2020 : {item[item_number][0][4]}</p>
                <p id="price">{item[item_number][0][3]}€</p>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
