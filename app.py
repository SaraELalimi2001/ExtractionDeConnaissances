import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score
import plotly.express as px
import plotly.figure_factory as ff
import streamlit.components.v1 as components
import json
from streamlit_lottie import st_lottie
import requests 

   
# Charger un jeu de données (exemple avec iris)
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
# Fonction pour entraîner et évaluer le modèle sélectionné
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, **kwargs):
    st.success(f"Vous avez choisi le modèle : {model_name}")

    # Ajuster les hyperparamètres spécifiques au modèle sélectionné
    if model_name == "Random Forest":
        model.set_params(n_estimators=kwargs.get("n_estimators", 100),
                         max_depth=kwargs.get("max_depth", None),
                         min_samples_split=kwargs.get("min_samples_split", 2),
                         min_samples_leaf=kwargs.get("min_samples_leaf", 1))
    elif model_name == "SVM":
        model.set_params(C=kwargs.get("C", 1.0),
                         kernel=kwargs.get("kernel", "rbf"))

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # creation des colonnes
    c1,c2=st.columns(2)
    # Afficher les métriques
    accuracy = accuracy_score(y_test, y_pred)
    c1.info(f"Précision : {accuracy:.2f}")
    # Utiliser Plotly Express pour afficher la matrice de confusion en tant que heatmap interactif
    
    
    cm = confusion_matrix(y_test, y_pred)

    fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Setosa", "Versicolor", "Virginica"],
    y=["Setosa", "Versicolor", "Virginica"],
    colorscale="Blues",
)

    # Ajouter le texte avec le nombre d'instances dans chaque cellule
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(cm[i][j]),
                showarrow=False,
                font=dict(size=10),
            )

    # Mise en page
    fig.update_layout(
        title=dict(text="Matrice de Confusion", font=dict(color="White"),y=0.95),
        width=300,
        height=220,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    # Afficher la figure
    c1.plotly_chart(fig)

    c2.info("Prédiction de la classe")
    with c2:
        col1,col2=st.columns(2)
        param1 = col1.slider("Longueur du sépale", min_value=min(iris.data[:, 0]), max_value=max(iris.data[:, 0]), value=iris.data[:, 0].mean())
        param2 = col1.slider("Largeur du sépale", min_value=min(iris.data[:, 1]), max_value=max(iris.data[:, 1]), value=iris.data[:, 1].mean())
        param3 = col2.slider("Longueur du pétale", min_value=min(iris.data[:, 2]), max_value=max(iris.data[:, 2]), value=iris.data[:, 2].mean())
        param4 = col2.slider("Largeur du pétale", min_value=min(iris.data[:, 3]), max_value=max(iris.data[:, 3]), value=iris.data[:, 3].mean())
        # Utiliser les valeurs des sliders pour faire une prédiction
        prediction_flowers = model.predict([[param1, param2, param3, param4]])

    # Afficher la classe prédite en fonction de la valeur
    if prediction_flowers[0] == 0:
        c2.success("Classe prédite : Setosa")
    elif prediction_flowers[0] == 1:
        c2.success("Classe prédite : Versicolor")
    elif prediction_flowers[0] == 2:
        c2.success("Classe prédite : Virginica")


with st.sidebar:
    selected = option_menu("Main Menu", ["Home","Modèle", 'Settings'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=0)
    
if selected == "Home" or selected is None : 
    st.write("# Welcome ! 👋")

    st.markdown(
       
       "Cette application vous offre la possibilité de choisir parmi plusieurs modèles de machine learning. "
    "Vous pourrez explorer et analyser les performances de chaque modèle en affichant des métriques telles que la précision, "
    "et visualiser la matrice de confusion pour évaluer sa capacité à faire des prédictions précises."
)
    
    url = requests.get( 
        "https://lottie.host/7a447ab7-8427-48dc-8dab-544eef7ccff4/EK5ucNy81A.json") 
    # Creating a blank dictionary to store JSON file, 
    # as their structure is similar to Python Dictionary 
    url_json = dict() 
    
    if url.status_code == 200: 
        url_json = url.json() 
        st_lottie(url_json, 
        reverse=True, 
        height=400, 
        width=600, 
        speed=1, 
        loop=True, 
        quality='high'
        )
    else: 
        print("Error in the URL") 
    
    
     
    
     
elif selected == "Modèle":
    st.header("Choisissez un modèle")

    # Liste des modèles disponibles
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Méthode Bayésienne": GaussianNB(),
        "Ridge": RidgeClassifier(),
        "Régression Logistique": LogisticRegression()
    }

    # Sélectionner le modèle
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle", list(models.keys()))

    # Ajouter des sliders pour ajuster les hyperparamètres spécifiques à chaque modèle
    
    if selected_model == "Random Forest":
        st.sidebar.subheader("Ajustement des Hyperparamètres")
        # Ajuster le nombre d'estimateurs (n_estimators)
        n_estimators = st.sidebar.slider("Nombre d'estimateurs ", min_value=1, max_value=100, value=10)

        # Ajuster la profondeur maximale (max_depth)
        max_depth = st.sidebar.slider("Profondeur maximale", min_value=1, max_value=20, value=5)

        # Ajuster le nombre minimal d'échantillons pour la division (min_samples_split)
        min_samples_split = st.sidebar.slider("min_samples_split", min_value=2, max_value=20, value=2)

        # Ajuster le nombre minimal d'échantillons dans une feuille (min_samples_leaf)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", min_value=1, max_value=20, value=1)

        # Créer un dictionnaire d'hyperparamètres pour le modèle Random Forest
        hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }
        evaluate_model(selected_model, models[selected_model], X_train, X_test, y_train, y_test, **hyperparameters)
    elif selected_model == "SVM":
        st.sidebar.subheader("Ajustement des Hyperparamètres SVM")
        # Ajuster le paramètre de régularisation (C)
        C = st.sidebar.slider("Régularisation (C)", min_value=0.1, max_value=10.0, value=1.0)

        # Ajuster le noyau (kernel)
        kernel = st.sidebar.selectbox("Noyau", ["linear", "poly", "rbf", "sigmoid"], index=2)

        # Ajuster d'autres hyperparamètres spécifiques au SVM si nécessaire

        # Créer un dictionnaire d'hyperparamètres pour le modèle SVM
        hyperparameters_svm = {
            "C": C,
            "kernel": kernel
            # Ajoutez d'autres hyperparamètres spécifiques au SVM si nécessaire
        }
        evaluate_model(selected_model, models[selected_model], X_train, X_test, y_train, y_test, **hyperparameters_svm)
    else:
    # Entraîner et évaluer le modèle sélectionné avec les hyperparamètres ajustés
        evaluate_model(selected_model, models[selected_model], X_train, X_test, y_train, y_test)

