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
import io
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import dill 
import plotly.graph_objs as go

##############
from fct_plot import make_donut
from fct_process import extraction, extract_bounds



################################
account_name = "fruitsamurai97depot"
account_key=''
with open("azure_container_key.txt", "r") as my_key:
    account_key= my_key.read()
container_name= "assets"
################################

connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

#use the client to connect to the container
container_client = blob_service_client.get_container_client(container_name)



################ Load data~###########
# Utilisez @st.cache pour charger et préparer les données
@st.cache_data
def load_data():

    
    test_df_name = "test_df.csv"
    #### load test data set ############
    sas_test = generate_blob_sas(account_name = account_name,
                                container_name = container_name,
                                blob_name = test_df_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))

    sas_test_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + test_df_name + '?' + sas_test
    df= pd.read_csv(sas_test_url)
    return df

@st.cache_resource
def load_model():
    model_blob_name = "modele_base.joblib" 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_blob_name)
    stream = io.BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    stream.seek(0)  # Go back to the start of the stream
    clf = load(stream)
    #clf = load('modele_base.joblib')
    return clf


################## Load LIME explainer ############################
@st.cache_resource
def load_explainer(): 
    explainer_blob_name = 'lime_explainer.pkl'
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=explainer_blob_name)
    stream = io.BytesIO()
    blob_client.download_blob().download_to_stream(stream)
    stream.seek(0)  # Réinitialise le pointeur au début du stream pour la lecture
    explainer = dill.load(stream)  # Chargez directement à partir du stream
    return explainer

#########################################################################



    ################## Create client overview ###############################
def client_overview(df,selected_ID):         
    dict_sel= {"CODE_GENDER":"Aucun", "AMT_INCOME_TOTAL":0, "AMT_CREDIT":0, "AMT_ANNUITY":0, "AMT_GOODS_PRICE":0, "DAYS_BIRTH":0}
    col_sel=["CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    col_names= ["Sexe", "Annual Income", "Credit Amount", "Annuities", "Goods Price", "Age"]
    client= df[df["SK_ID_CURR"]== selected_ID]
    client_att = client[col_sel].iloc[0] # Sélectionner les attributs

    if client_att[0] ==0:
        dict_sel["CODE_GENDER"] = "Femme"
    else:
        dict_sel["CODE_GENDER"] = "Homme"

    if client_att[1] > 0:
        dict_sel["AMT_INCOME_TOTAL"] = round(client_att[1])
    if client_att[2] > 0:
        dict_sel["AMT_CREDIT"] = round(client_att[2])
    if client_att[3] > 0:
        dict_sel["AMT_ANNUITY"] = round(client_att[3])
    if client_att[4] > 0:
        dict_sel["AMT_GOODS_PRICE"] = round(client_att[4])
    if client_att[5]<0:
        dict_sel["DAYS_BIRTH"] = round(-client_att[5]/365)
    
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
            


    ###### Afficher les probabilités de défault ############
def show_proba(df,selected_ID):
    #######################
    # Initialisation des données de probabilité pour le premier élément de la liste
    list_IDS = df["SK_ID_CURR"].unique().tolist()
    default_id = list_IDS[0]
    condition = df['SK_ID_CURR'] == default_id
    elt = df[condition]["PAYBACK_PROBA"].tolist()[0]
    if elt:  # Assurez-vous que la liste n'est pas vide pour le premier ID
        default_proba_remboursement = round(elt*100)
        default_proba_default = round((1 - elt)*100)
    else:  # Valeurs par défaut si la liste est vide
        default_proba_remboursement = 0
        default_proba_default = 100
  # Dashboard Main Panel
    col1, col2= st.columns((1, 1), gap='medium')

    #st.markdown('#### Probabilité de remboursement')
    condition = df['SK_ID_CURR'] == selected_ID
    elt = df[condition]["PAYBACK_PROBA"].tolist()[0]
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
    credit_status = df[condition]["IF_0_CREDIT_IS_OKAY"].tolist()
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
        if credit_status and credit_status[0] == 0:
            st.markdown("<h3 style='color:green; font-size:24px;'>Crédit accordé</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red; font-size:24px;'>Crédit non accordé</h3>", unsafe_allow_html=True)
@st.cache_data#(allow_output_mutation=True, show_spinner=False)     
def load_explanations(df,selected_ID,_clf,_explainer):
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index',"IF_0_CREDIT_IS_OKAY","PAYBACK_PROBA"]]
    test_x = df[feats]
    test_x_np = test_x.to_numpy()
    condition = df['SK_ID_CURR'] == selected_ID
    client_instance = test_x_np[df[condition].index[0]]  
    exp= _explainer.explain_instance(
        data_row=client_instance, 
        predict_fn=_clf.predict_proba, 
        num_features=5
    )
    features_names,lime_threshold, features_impact, exp_list = extraction(exp.as_list())
    return features_names, lime_threshold, features_impact, exp_list 
######################## Voir les explications ################
def show_explanations(features_names ,features_impact):
        
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

    
    
   
def highlight_instance(df,selected_ID, features_names,lime_threshold, features_impact, exp_list,selected_feature):

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

    

        
def test_affichage(df,selected_ID):
    condition = df['SK_ID_CURR'] == selected_ID

    st.write(df[condition])
    