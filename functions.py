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




################ Load data~###########
# Utilisez @st.cache pour charger et préparer les données

def load_data():
    df = pd.read_csv("Assets/Data/test.csv")
    train_df = pd.read_csv("Assets/Data/df_train_base.csv")  
    return df, train_df

def import_columns():
    col_desc = pd.read_csv("Assets/Data/Columns_description.csv", encoding='ISO-8859-1')
    col_desc=col_desc[["Row","Description"]]
    col_desc= col_desc.rename(columns={"Row":"Feature"})
    return col_desc

###### Afficher les probabilités de défault ############
def show_proba(df,list_IDS,selected_ID):

    #######################
    # Initialisation des données de probabilité pour le premier élément de la liste
    default_id = list_IDS[0]
    condition = df['SK_ID_CURR'] == default_id
    elt = df[condition]["Proba de remboursement"].tolist()[0]
    if elt:  # Assurez-vous que la liste n'est pas vide pour le premier ID
        default_proba_remboursement = round(elt*100)
        default_proba_default = round((1 - elt)*100)
    else:  # Valeurs par défaut si la liste est vide
        default_proba_remboursement = 0
        default_proba_default = 100

    #######################
    # Dashboard Main Panel
  # Dashboard Main Panel
    #col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

    #st.markdown('#### Probabilité de remboursement')
    condition = df['SK_ID_CURR'] == selected_ID
    elt = df[condition]["Proba de remboursement"].tolist()[0]
    if elt:  # Assurez-vous que la liste n'est pas vide
        proba_remboursement = round(elt*100)
        proba_default = round((1-elt)*100)
    else:  # Valeurs par défaut si la liste est vide pour l'ID sélectionné
        proba_remboursement = 0
        proba_default  = 100

    # Création des chartes de proba
    donut_chart_greater = make_donut(proba_remboursement , 'Remboursement', 'green')
    donut_chart_less = make_donut(proba_default, 'Défaut', 'red')

    # Affichage des résultats dans Streamlit
    #markdown('#### Probabilité de remboursement')
    st.altair_chart(donut_chart_greater, use_container_width=True)

    #markdown('#### Probabilité de défaut')
    st.altair_chart(donut_chart_less, use_container_width=True)

    credit_status = df[condition]["credit accordé == 0"].tolist()
    if credit_status and credit_status[0] == 0:
        st.markdown("<h3 style='color:green; font-size:24px;'>Crédit accordé</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red; font-size:24px;'>Crédit non accordé</h3>", unsafe_allow_html=True)





    ################## Create client overview ###############################
def client_overview(df, selected_ID):
        #df_desc = pd.read_csv("Assets/Data/Columns_description.csv", encoding='ISO-8859-1')
        #df_desc=df_desc[["Row","Description"]]
        #df_desc= df_desc.rename(columns={"Row":"Feature"})
        
        
    dict_sel= {"CODE_GENDER":"Aucun", "AMT_INCOME_TOTAL":0, "AMT_CREDIT":0, "AMT_ANNUITY":0, "AMT_GOODS_PRICE":0, "DAYS_BIRTH":0}
    #col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')
    col_sel=["CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH"]
    col_names= ["Sexe", "Total Income", "Credit Amount", "Annuities", "Goods Price", "Age"]
    client= df[df["SK_ID_CURR"]== selected_ID]
    client_att = client[col_sel].iloc[0] # Sélectionner les attributs

    if client_att[0] ==0:
        dict_sel["CODE_GENDER"] = "Femme"
    else:
        dict_sel["CODE_GENDER"] = "Homme"

    if client_att[1] > 0:
        dict_sel["AMT_INCOME_TOTAL"] = client_att[1]
    if client_att[2] > 0:
        dict_sel["AMT_CREDIT"] = client_att[2]
    if client_att[3] > 0:
        dict_sel["AMT_ANNUITY"] = client_att[3]
    if client_att[4] > 0:
        dict_sel["AMT_GOODS_PRICE"] = client_att[4]
    if client_att[5]<0:
        dict_sel["DAYS_BIRTH"] = round(-client_att[5]/365)
    
    # Création des inputs
    
    for col_name, sel_key in zip(col_names, col_sel):
        user_input = st.text_input(col_name, dict_sel[sel_key])
        # Vous pouvez utiliser user_input comme vous le souhaitez, par exemple pour mettre à jour dict_sel ou pour d'autres logiques.

   # st.text_input("Sexe")
    #st.write(dict_sel["CODE_GENDER"])
            





######################## Voir les explications ################
            

def show_explanations(selected_ID):
    #col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

    df,train_df = load_data()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df["TARGET"], test_size=0.2, random_state=42)
    test_x = df[feats]
    train_x_np, test_x_np = train_x.to_numpy(), test_x.to_numpy()
    clf = load('Assets/Models/modele_base.joblib')
    ### Partie 2 (afficher les explication)            
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=train_x_np,feature_names=test_x.columns.tolist(),  class_names=['Crédit autorisé', 'Default'],
        verbose=True,
        mode='classification'
        )
    condition = df['SK_ID_CURR'] == selected_ID
    idx_client = df.index[condition].tolist()[0]
    instance = test_x_np[idx_client]
    exp = explainer.explain_instance(data_row=instance, predict_fn=clf.predict_proba, num_features=6)

    #with col2:
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)  # Affichez la figure dans Streamlit
