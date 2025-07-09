#######################
import streamlit as st
import pandas as pd
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
##############################
import lime 
import lime.lime_tabular
from joblib import load
import dill 
import plotly.graph_objs as go

##############
from fct_plot import make_donut
from fct_process import extraction, extract_bounds
import requests
import os



#########################################################################
@st.cache_data
def fetch_ids():
    """Fonction pour récupérer la liste des IDs depuis l'API Flask."""
    API_URL = "https://oc-api-score.azurewebsites.net/client"
    response = requests.get(API_URL)
    if response.status_code == 200:
        ids = response.json()  # La réponse est attendue sous forme de JSON
        return ids
    else:
        st.error('Failed to retrieve IDs')
        return []
    
@st.cache_data
def fetch_info(selected_ID):
    API_URL = f"https://oc-api-score.azurewebsites.net/info?id={selected_ID}"
    response = requests.get(API_URL)
    if response.status_code == 200:
        info=response.json()
        return info
    else:
        st.error("failed to retrieve client info")
        return []

    ################## Create client overview ###############################
def client_overview(selected_ID):     
    dict_sel= {"CODE_GENDER":"Aucun", "AMT_INCOME_TOTAL":0, "AMT_CREDIT":0, "AMT_ANNUITY":0, "AMT_GOODS_PRICE":0, "DAYS_BIRTH":0}
    col_sel=["CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    col_names= ["Sexe", "Annual Income", "Credit Amount", "Annuities", "Goods Price", "Age"]
    #client= df[df["SK_ID_CURR"]== selected_ID]
    client_att = fetch_info(selected_ID)#client[col_sel].iloc[0] # Sélectionner les attributs

    if client_att["CODE_GENDER"] ==0:
        dict_sel["CODE_GENDER"] = "Femme"
    else:
        dict_sel["CODE_GENDER"] = "Homme"

    if client_att["AMT_INCOME_TOTAL"] > 0:
        dict_sel["AMT_INCOME_TOTAL"] = round(client_att["AMT_INCOME_TOTAL"])
    if client_att["AMT_CREDIT"] > 0:
        dict_sel["AMT_CREDIT"] = round(client_att["AMT_CREDIT"])
    if client_att["AMT_ANNUITY"] > 0:
        dict_sel["AMT_ANNUITY"] = round(client_att["AMT_ANNUITY"])
    if client_att["AMT_GOODS_PRICE"] > 0:
        dict_sel["AMT_GOODS_PRICE"] = round(client_att["AMT_GOODS_PRICE"])
    if client_att["DAYS_BIRTH"]<0:
        dict_sel["DAYS_BIRTH"] = round(-client_att["DAYS_BIRTH"]/365)
    
    # Création des inputs
    st.write("### Input Data")
    col1,col2,col3= st.columns(3)
    
    gender= col1.text_input(col_names[0],dict_sel["CODE_GENDER"], disabled=True)
    age = col1.number_input(col_names[5],dict_sel["DAYS_BIRTH"], disabled=True)

    income = col2.number_input(col_names[1], dict_sel["AMT_INCOME_TOTAL"], disabled=True)
    goods = col2.number_input(col_names[4],dict_sel["AMT_GOODS_PRICE"], disabled=True)

    credit= col3.number_input(col_names[2],dict_sel["AMT_CREDIT"], disabled=True)
    annuities= col3.number_input(col_names[3],dict_sel["AMT_ANNUITY"], disabled=True)
    
    del col1,col2,col3
    st.write("")
    col1,col2,col3 = st.columns((2,1,1))
    ## création détails et mettre à gauche####
    with col1:
        st.write("### Details client")  
    with col2:
        st.write("")
    with col3:
        st.write("")

    del col1,col2,col3
    col1,col2,col3= st.columns(3)

    credit_income_percent= round(dict_sel["AMT_CREDIT"]*100/dict_sel["AMT_INCOME_TOTAL"],2)
    annuity_income_percent= round(dict_sel["AMT_ANNUITY"]*100/dict_sel["AMT_INCOME_TOTAL"],2)
    credit_term = round(dict_sel["AMT_CREDIT"]/dict_sel["AMT_ANNUITY"])

    col1.metric(label="Credit/Income %", value=f"{credit_income_percent:,.0f}%")
    col2.metric(label="Annuity/Income %", value=f"{annuity_income_percent:,.0f}%")
    col3.metric(label="Credit Term", value=f"{credit_term:,.0f} Years")
            
#########################################################################
@st.cache_data
def request_proba(selected_ID):
    API_URL = f"https://oc-api-score.azurewebsites.net/predict?id={selected_ID}"  # Assurez-vous que l'URL correspond à votre configuration
    response = requests.get(API_URL)
    if response.status_code == 200:
        proba = response.json()
        return proba
    else:
        st.error('Failed to retrieve proba')
        return []





    ###### Afficher les probabilités de défault ############
def show_proba(selected_ID):
    #######################
    # Initialisation des données de probabilité pour le premier élément de la liste
    #list_IDS = df["SK_ID_CURR"].unique().tolist()
    #default_id = list_IDS[0]
    #condition = df['SK_ID_CURR'] == default_id
    #elt = df[condition]["PAYBACK_PROBA"].tolist()[0]
    #elt = list( request_proba(default_id).values())[1]
    
  # Dashboard Main Panel
    col1, col2= st.columns((1, 1), gap='medium')

    #st.markdown('#### Probabilité de remboursement')
    #condition = df['SK_ID_CURR'] == selected_ID
    elt = list( request_proba(selected_ID).values())[0]
    if elt:  # Assurez-vous que la liste n'est pas vide
        proba_remboursement = round(elt*100)
        proba_default = round((1-elt)*100)
    else:  # Valeurs par défaut si la liste est vide pour l'ID sélectionné
        proba_remboursement = 0
        proba_default  = 100

    # Création des chartes de proba
    donut_chart_greater = make_donut(proba_remboursement , 'Remboursement', 'green')
    donut_chart_less = make_donut(proba_default, 'Défaut', 'red')

## variable pour afficher le statut du crédit
    #credit_status = df[condition]["IF_0_CREDIT_IS_OKAY"].tolist()
    if elt>0.51:
        credit_status = 0
    else:
        credit_status = 1
    # Affichage des résultats dans Streamlit
    with col1:
        st.markdown('###### Proba remboursement')
        st.altair_chart(donut_chart_greater, use_container_width=True)
    with col2:
        st.markdown('###### Proba défaut')
        st.altair_chart(donut_chart_less, use_container_width=True)

    del col1,col2
    col1,col2= st.columns((1,2))
    with col1:
        st.write("")
    with col2:
        if credit_status  == 0:
            st.markdown("<h3 style='color:green; font-size:24px;'>Crédit accordé</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red; font-size:24px;'>Crédit non accordé</h3>", unsafe_allow_html=True)




@st.cache_data
def get_explications(selected_ID):
    API_URL = f"https://oc-api-score.azurewebsites.net/explain?id={selected_ID}"
    response = requests.get(API_URL)
    if response.status_code == 200:
        explanations=response.json()
        return explanations
    else:
        st.error("failed to retrieve explanations info")
        return []




@st.cache_data#(allow_output_mutation=True, show_spinner=False)     
def load_explanations(selected_ID):
    
    exp_list= get_explications(selected_ID)
    features_names,lime_threshold, features_impact, exp_list = extraction(exp_list)
    return features_names, lime_threshold, features_impact, exp_list 
######################## Voir les explications ################
def show_explanations(selected_ID):
    features_names, lime_threshold, features_impact, exp_list= load_explanations(selected_ID)
    del lime_threshold,exp_list  
    dict_lime = dict(zip(features_names, features_impact))
    colors_original_order = ['darkgreen' if x < 0 else 'darkred' for x in dict_lime.values()]
    df_lime= pd.DataFrame(dict_lime.items(),columns=["Feature","Value"])
    

    # Créer le graphique Plotly
    fig = go.Figure([go.Bar(x=df_lime['Value'], y=df_lime['Feature'], orientation='h', marker_color=colors_original_order)])
    
    # Configurer le graphique
    fig.update_layout(
        title={
            'text': 'Features contribution on credit decision',
            'y':0.9,  # Position verticale du titre
            'x':0.5,  # Position horizontale du titre
            'xanchor': 'center',  # Ancre le titre au centre horizontalement
            'yanchor': 'top'  # Ancre le titre en haut verticalement
        },
        title_font_size=24,  # Augmenter la taille de la police du titre
        dragmode=False,  # Désactiver les interactions de zoom/drag
        showlegend=False,  # Cacher la légende
        xaxis_showticklabels=True,  # Montrer les étiquettes de l'axe X
        yaxis_showticklabels=True  # Montrer les étiquettes de l'axe Y
    )


    config = {
        'staticPlot': True,  # Rend le graphique statique (non interactif)
        'displayModeBar': False  # Cache la barre d'outils
    }
    graph = st.plotly_chart(fig, use_container_width=True, config=config)

@st.cache_data  
def feature_dist(selected_feature):
    API_URL = f"https://oc-api-score.azurewebsites.net/feature?feature={selected_feature}"
    response = requests.get(API_URL)
    if response.status_code == 200:
        feature=response.json()
        return feature
    else:
        st.error("failed to retrieve feature info")
        return []
   
def highlight_instance(selected_ID,selected_feature):
    features_names, lime_threshold, features_impact, exp_list=load_explanations(selected_ID)
    df=pd.DataFrame()
    df["SK_ID_CURR"]= fetch_ids()
    for feature in features_names:
        df[feature]= feature_dist(feature)
    
    condition = df['SK_ID_CURR'] == selected_ID
    features_values = df[condition][features_names].iloc[0].tolist() ## Récupérer les valeur de ces 5 features pour ce client
    dict_lime = dict(zip(features_names, features_values)) # Mettre ces valeurs dans un dictionnaire
    dict_impact = dict(zip(features_names, features_impact)) # Mettre les impacts des features dans un dict
    dict_threshold = dict(zip(features_names, lime_threshold)) # Mettre le lime threshold dans un dict
    feature_value = dict_lime[selected_feature] # Stocker la valeur de ce feature pour ce client
    feature_impact = dict_impact[selected_feature] #Stocker l'impact de ce feature pour la décision de crédit
    lime_threshold = float(dict_threshold[selected_feature])
    
    # Eliminate top 1% values to remove outliers
    # Calculer les seuils des 1% les plus bas et 1% les plus hauts
    df = df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)  # transformer les bool en 0/1
    limite_basse = df[selected_feature].quantile(0.01)
    limite_haute = df[selected_feature].quantile(0.99)


    #threshold = np.percentile(df[selected_feature], 99.99)  # Find the 95th percentile value
    filtered_df = df[(df[selected_feature] >= limite_basse) & (df[selected_feature] <= limite_haute)] #df[df[selected_feature] <= threshold]  # Keep only the data <= 99th percentile to avoid extreme values
    
    # Define the figure size
    plt.figure(figsize=(10, 6))
    
    # Plot the distribution of the feature for the dataset with outliers removed
    sns.histplot(filtered_df[selected_feature], kde=True, color = 'darkblue', 
                 edgecolor='black', linewidth=1)
    
    if feature_value <= limite_haute and feature_value >= limite_basse:
        # Determine the color and label based on the lime_impact value
        if feature_impact < 0:
            color = 'green'
            label = 'Positive Contribution  '
        else:
            color = 'red'
            label = 'Negative Contribution  '
        # Highlight the value for the specified original instance if it is within the new range
        plt.axvline(feature_value, color=color, linestyle='-', linewidth=1)
        # Add a label next to the line
        #plt.text(feature_value, plt.gca().get_ylim()[1] * 0.70, label, color=color, horizontalalignment='right')
        
        # Highlight the new instance value in yellow
        if (lime_threshold >= limite_basse) & (lime_threshold <= limite_haute) :
            plt.axvline(lime_threshold, color='yellow', linestyle='--', linewidth=1)
            
        else:
            for e in exp_list:
                if extract_bounds(e[0]):
                    lime_inf,lime_sup= extract_bounds(e[0])
                    if (lime_inf >= limite_basse) & (lime_sup<=limite_haute):
                        plt.axvline(lime_inf, color='yellow', linestyle='--', linewidth=1)
                        plt.axvline(lime_sup, color='yellow', linestyle='--', linewidth=1)
        

        if (lime_threshold >= limite_basse) & (lime_threshold <= limite_haute) :
            plt.legend(['Distribution', f'Client ID "{selected_ID}"', 'Lime Threshold'], loc='upper left')
        else:
            plt.legend(['Distribution', f'Client ID "{selected_ID}"', 'Lime Inf', "Lime Sup"], loc='upper left')
     
    plt.title(f'Distribution of {selected_feature} with highlighted instances (ID: {selected_ID})')
    plt.xlabel(selected_feature)
    plt.ylabel('Density')
    st.pyplot(plt)

    

def features_client(selected_ID,selected_feature):
    
         

    df=pd.DataFrame()
    df["SK_ID_CURR"]= fetch_ids()

    df[selected_feature]= feature_dist(selected_feature)
    
    condition = df['SK_ID_CURR'] == selected_ID
    feature_value= df[condition][selected_feature].iloc[0]

    # Eliminate top 1% values to remove outliers
    # Calculer les seuils des 1% les plus bas et 1% les plus hauts
    df = df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)  # transformer les bool en 0/1
    limite_basse = df[selected_feature].quantile(0.01)
    limite_haute = df[selected_feature].quantile(0.99)


    #threshold = np.percentile(df[selected_feature], 99.99)  # Find the 95th percentile value
    filtered_df = df[(df[selected_feature] >= limite_basse) & (df[selected_feature] <= limite_haute)] #df[df[selected_feature] <= threshold]  # Keep only the data <= 99th percentile to avoid extreme values
    
    # Define the figure size
    plt.figure(figsize=(10, 6))
    
    # Plot the distribution of the feature for the dataset with outliers removed
    sns.histplot(filtered_df[selected_feature], kde=True, color = 'darkblue', 
                edgecolor='black', linewidth=1)
    plt.axvline(feature_value, color="blue", linestyle='-', linewidth=2)
    
    plt.title(f'Distribution of {selected_feature} with highlighted instances (ID: {selected_ID})')
    plt.xlabel(selected_feature)
    plt.ylabel('Density')
    st.pyplot(plt)


    
##########################################
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.warning("OPENAI_API_KEY environment variable is not set.")
@st.cache_resource
def create_prompt(selected_ID):
    features_names, lime_threshold, features_impact, exp_list=load_explanations(selected_ID)
 
    features= exp_list
    prompt = "Generate a report explaining the impact of the following features on the model's prediction: "
    prompt += ", ".join([f"{feature}: {impact:.2f}" for feature, impact in features])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Correctly accessing the response content
    generated_text = response.choices[0].message['content']
    st.write("## Report")
    st.write(generated_text)


