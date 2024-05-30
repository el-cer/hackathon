from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/eliot/Desktop/dossier_projet_hackaton/hackathon/app/datasetreco.csv',sep=';')
reco_df = df.copy()

reco_df["score"]=reco_df["p_views"]+(reco_df["p_carts"]*5)+(reco_df["p_purchases"]*10) #rating

main = reco_df.groupby('category').max('score')[['score']].reset_index()
main_df = reco_df.merge(main, on=['category'])
main_df['score_norm'] = main_df['score_x']/main_df['score_y']

main = main_df[reco_df['is_purchase']>=1]
pivot_data=pd.pivot_table(main, values='score_norm', index=['product_id'],
                       columns=['user_id'], fill_value=0)


svd = TruncatedSVD(n_components=25, n_iter=7, random_state=42)
decomposed_matrix = svd.fit_transform(pivot_data)
explained_variance = svd.explained_variance_ratio_
explained_variance = 1 - explained_variance #have the exolained variance on the good order

nb_composants_to_save = np.argmax(explained_variance > 0.95) + 1 #result is 5

plt.figure(figsize=(15, 6))
plt.plot(explained_variance)
plt.scatter(nb_composants_to_save -1, 0.96, c="r", label="Seuil pour garder 95 % du poids informationnel")

plt.xlabel('SVD components')  # Ajout du label pour l'axe des x
plt.ylabel('Information cumulée')  # Ajout du label pour l'axe des y
plt.show()
