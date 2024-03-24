#######################

import streamlit as st
import pandas as pd
import altair as alt
#import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
##############################
import lime 
import lime.lime_tabular
from joblib import load
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from joblib import dump, load
import io
from lightgbm import plot_importance
from datetime import datetime, timedelta

################################
account_name = "fruitsamurai97depot"
account_key=''
with open("azure_container_key.txt", "r") as my_key:
    account_key= my_key.read()
container_name= "assets"
################################


#col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')
# Plots
# Donut chart
########################
@st.cache_data
def make_donut(input_response, input_text, input_color):
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

##############################


################ Load data~###########
# Utilisez @st.cache pour charger et préparer les données

def load_data():

    connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    #use the client to connect to the container
    container_client = blob_service_client.get_container_client(container_name)

    

    ### load my model from Azure #######
    test_df_name = "test_df.csv"
    
    ######################################

    #### load test data set ############
    sas_test = generate_blob_sas(account_name = account_name,
                                container_name = container_name,
                                blob_name = test_df_name,
                                account_key=account_key,
                                permission=BlobSasPermissions(read=True),
                                expiry=datetime.utcnow() + timedelta(hours=1))

    sas_test_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + test_df_name + '?' + sas_test
    df= pd.read_csv(sas_test_url)
    #####################################

   
    ########################################
    return df

######### Loading Data ####################
df= load_data()
#columns_description = import_columns()
col_selection = [c for c in df.columns if c not in ['TARGET','IF_0_CREDIT_IS_OKAY', 'PAYBACK_PROBA','LIME_FEATURES_NAMES','LIME_FEATURES_THRESHOLD','LIME_FEATURES_IMPACT']]
test_df= df[col_selection]
list_IDS = df["SK_ID_CURR"].unique().tolist()
##############################################



#def import_columns():
#    col_desc = pd.read_csv("Assets/Data/Columns_description.csv", encoding='ISO-8859-1')
#    col_desc=col_desc[["Row","Description"]]
#    col_desc= col_desc.rename(columns={"Row":"Feature"})
#    return col_desc

###### Afficher les probabilités de défault ############
def show_proba(selected_ID):
    #######################
    # Initialisation des données de probabilité pour le premier élément de la liste
    default_id = list_IDS[0]
    condition = df['SK_ID_CURR'] == default_id
    elt = df[condition]["PAYBACK_PROBA"].tolist()[0]
    if elt:  # Assurez-vous que la liste n'est pas vide pour le premier ID
        default_proba_remboursement = round(elt*100)
        default_proba_default = round((1 - elt)*100)
    else:  # Valeurs par défaut si la liste est vide
        default_proba_remboursement = 0
        default_proba_default = 100

    #######################
    # Dashboard Main Panel
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
        


    ################## Create client overview ###############################
def client_overview(selected_ID):   
        
    dict_sel= {"CODE_GENDER":"Aucun", "AMT_INCOME_TOTAL":0, "AMT_CREDIT":0, "AMT_ANNUITY":0, "AMT_GOODS_PRICE":0, "DAYS_BIRTH":0}
    #col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')
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
    
    gender= col1.text_input(col_names[0],dict_sel["CODE_GENDER"])
    age = col1.number_input(col_names[5],dict_sel["DAYS_BIRTH"])

    income = col2.number_input(col_names[1], dict_sel["AMT_INCOME_TOTAL"])
    goods = col2.number_input(col_names[4],dict_sel["AMT_GOODS_PRICE"])

    credit= col3.number_input(col_names[2],dict_sel["AMT_CREDIT"])
    annuities= col3.number_input(col_names[3],dict_sel["AMT_ANNUITY"])
    
    del col1,col2,col3
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

    col1.metric(label="Credit Income %", value=f"{credit_income_percent:,.2f}%")
    col2.metric(label="Annuity Income %", value=f"{annuity_income_percent:,.0f}%")
    col3.metric(label="Credit Term", value=f"{credit_term:,.0f} Years")
            





######################## Voir les explications ################
            

def show_explanations(selected_ID):
    #col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

    condition = df['SK_ID_CURR'] == selected_ID
    st.markdown('#### Best 5 features')
    st.write(df[condition]["LIME_FEATURES_NAMES"])
    #fig = exp.as_pyplot_figure()
    #st.pyplot(fig)  # Affichez la figure dans Streamlit


def test_affichage(df):
    st.write(df)
