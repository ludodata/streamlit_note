
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix





def main():
    st.title("Détection de faux billets")
    st.subheader("Auteur : Ludovic Cancy")
    
    # Importe les données pour construire le modèle
    data_model = pd.read_csv("data_good.csv") 
    
    # Création d'une instance de LabelEncoder
    encoder = LabelEncoder()

    # Transformation des données booléennes en entiers
    data_model['is_genuine'] = encoder.fit_transform(data_model['is_genuine'])

    # Récupère les données
    X_wth_genuine = data_model.iloc[:, 1:].values
    
    # Standardise les données
    scaler_standard = StandardScaler()   
    scaler_standard.fit(X_wth_genuine)
    X_wth_genuine_std = scaler_standard.transform(X_wth_genuine)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_wth_genuine_std, data_model['is_genuine'], test_size=0.2, random_state=5, stratify=data_model['is_genuine'])

    # On entraîne le modèle sur les données d'entraînement
    estimator2 = LogisticRegression()
    estimator2.fit(X_train, y_train)

    #score
    # Accuracy score
    tr_score_logi = estimator2.score(X_train, y_train).round(4)

    #prediction test sur le model
    y_pred_test = estimator2.predict(X_test)

    #matrice de confusion du modele
    mat2 = confusion_matrix(y_test, y_pred_test)
    mat2 = pd.DataFrame(mat2)
    mat2.columns = [ f"pred_{i}" for i in mat2.columns]
    mat2.index = [ f"test_{i}" for i in mat2.index]








    # on utilise un kmeans
    kmeans2 = KMeans(n_clusters=2,random_state=42,n_init='auto') # random pour avoir toujours la meme initialisation
    kmeans2.fit(X_wth_genuine_std)
    

    

    # prédire les clusters pour l'ensemble de test
    y_pred_kmeans = kmeans2.predict(X_test)
    
    # Fonction d'importation des données
    def load_data(file):
        data = pd.read_csv(file)
        return data

    # Affichage du formulaire de téléchargement du fichier CSV
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        # Charger les données à partir du fichier CSV téléchargé
        df = load_data(uploaded_file)
        df_bis = df.drop(columns=['id'])
       
        df_bis= df_bis.values
        
        # Affichage des données brutes
        st.subheader("Jeu de données")
        st.write(df)

        # Appliquer le StandardScaler sur les colonnes numériques
        df_standard = scaler_standard.transform(df_bis)

        # Prédire les valeurs de is_genuine pour le nouveau dataframe
        y_pred_logi = estimator2.predict(df_standard)
        y_prob_best = estimator2.predict_proba(df_standard)

        y_pred_kmeans2 = kmeans2.predict(df_standard)

        
        df['est_vrai_reglog'] = y_pred_logi
        df['proba_est_vrai'] = y_prob_best[:, 1] 
        df['est_vrai_reglog'] = df['est_vrai_reglog'].astype(bool)

        df['kmeans'] = y_pred_kmeans2

        

       

        if st.sidebar.checkbox("Predictions", False):
            # Affichage des prédictions
            st.subheader("Voici les predictions")
            st.write(df)
            #st.subheader("Accuracy")
            #st.write(tr_score_logi)
            #st.subheader("Matrice")
            #st.write(mat2)
            

        
            


if __name__ == "__main__":
    main()


