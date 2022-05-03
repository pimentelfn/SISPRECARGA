import autots
from autots import AutoTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_icon="📈", page_title="Previsão de Carga")
st.sidebar.image("light.png")

# Titulo
st.title("Previsão de Carga das Subestações")



#Barra lateral
st.sidebar.header("Selecione Subestação")
st.sidebar.write("========================")

#Escolha da coluna
aa=pd.read_csv('a1.csv', sep=';')
aa=aa.drop(columns=['data'])
setds=aa.columns
#Escolha
choice = st.sidebar.selectbox("Selecione subestação",setds)

st.sidebar.header("Subestação Selecionada:  {}".format(choice))

options = st.sidebar.selectbox('Selecione conjunto de modelos',['fast', 'superfast','probabilistic', 'multivariate', 'all'])

df = pd.read_csv('a2.csv', sep=";",dtype={choice: float})
dd=df[[choice]]
ss=df[['soma']]


model_list = [
    'LastValueNaive',
    'AverageValueNaive',
]


metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
}


model = AutoTS(
    forecast_length=12,
    frequency='infer',
    prediction_interval=0.95,
    ensemble=['simple', 'horizontal-min'],
    max_generations=5,
    num_validations=4,
    validation_method='seasonal 12',
    #model_list= options,
    model_list= model_list,
    #model_list= 'superfast',
    transformer_list= 'fast',
    models_to_validate=0.2,
    n_jobs='auto'
)

model = model.fit(df, date_col='data', value_col=choice, id_col=None)

st.header("Subestação Selecionada ")
st.line_chart(data=dd, width=0, height=0, use_container_width=True)

prediction = model.predict()
forecasts = prediction.forecast

st.header("Previsões")
st.dataframe(forecasts)

st.header("Série de previsões")
st.line_chart(data=forecasts, width=0, height=0, use_container_width=True)
