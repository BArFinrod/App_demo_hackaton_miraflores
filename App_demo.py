#%%
import pandas as pd
import folium
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np

#%%


df_list = [[-12.116763, -77.044949, 5.0],
           [-12.116747, -77.043542, 0.0],
           [-12.116325, -77.045386, 2.0]]

hm_wide = HeatMap(df_list,
                  min_opacity=0.1,
                  radius=30,
                  gradient = {0.2:'cyan', 0.6: 'purple', 0.8:'red',1:'yellow'}, use_local_extrema=True,
                  name='Delitos denunciados'
                 )

f = folium.Figure(height=750)
hmap = folium.Map(location=[-12.116747, -77.043542], zoom_start=17, tiles='stamentoner', min_zoom=16, max_zoom=20)
hmap.add_child(hm_wide)
f.add_child(hmap)
scatter_map = folium.CircleMarker(location=[-12.116747, -77.043542],
                        radius=25,
                        weight=5, color='red').add_to(hmap)

st_folium(f)
# st.components.v1.html(f._repr_html_(), height=750)
# %%
st.header("1. Etapa de calibración")
st.markdown("Con el objetivo de calibrar sus opiniones y sumarlas al modelo, coloque a continuación \
        el número de delitos que UD. PIENSA que hubo en la cuadra en donde se ubica su vivienda durante...")
df_gsectorial = pd.DataFrame({'Dato real':[10.0, 2.0, 3.0, 1.0]}, index=['mes pasado','semana antepasada','semana pasada','ayer'])
df_e1 = pd.DataFrame({'Calibración ciudadano 1':[9.0,1.0,3.0,2.0]}, index=df_gsectorial.index)
# df_growth = pd.DataFrame.from_dict(dict([[2004,5.449063],[2005,6.842634],[2006,6.101323],[2007,15.599755],[2008,11.146391],
#                           [2009,0.776626],[2010,5.917537],[2011,4.365751],[2012,4.728066],[2013,2.702974],
#                           [2014,0.637255],[2015,3.298967],[2016,25.924020],[2017,3.719005],[2018,2.545244],
#                           [2019,-0.325247],[2020,-15.644018],[2021,13.206235],[2022,4.963000]]), orient='index', columns=['Crecimiento'])

df_e2 = pd.DataFrame(index=df_gsectorial.index, columns=['Calibración usted'])
edited_df_e2 = st.data_editor(df_e2)
edited_df_e2['Calibración usted'] = edited_df_e2['Calibración usted'].astype(float)
if edited_df_e2['Calibración usted'].hasnans:
    st.markdown("Escriba valores en todas las celdas")
else:
    try:
        var_e1 = ((df_e1['Calibración ciudadano 1'] - df_gsectorial['Dato real'])**2).sum()
        var_e2 = ((edited_df_e2['Calibración usted'].astype(float) - df_gsectorial['Dato real'])**2).sum()
        cov_e1e2 = ((df_e1['Calibración ciudadano 1'] - df_gsectorial['Dato real'])*(edited_df_e2['Calibración usted'].astype(float) - df_gsectorial['Dato real'])).sum()
        Sigma_e1e2 = np.array([[var_e1, cov_e1e2],[cov_e1e2, var_e2]])

        sigma_inv = np.linalg.inv(Sigma_e1e2)
        mu_0 = 0.0 # marzo 2023
        sigma_0 = 2.025# 4.1 # marzo 2023

        # xe1 = np.outer(np.linspace(mu_0-3*np.sqrt(var_e1), mu_0+3*np.sqrt(var_e1), 30), np.ones(30))
        # xe2 = np.outer(np.linspace(mu_0-3*np.sqrt(var_e2), mu_0+3*np.sqrt(var_e2), 30), np.ones(30)).T
        # flikelihood = sp.stats.multivariate_normal(mean=[mu_0, mu_0], cov=Sigma_e1e2).pdf(np.dstack((xe1, xe2)))
        # zmax = flikelihood.max()
    except:
        st.markdown("escriba solo números")

st.header("2. Etapa de agregación de opiniones al modelo")
forecast_e2_str = st.text_input("Coloque el número de delitos que ud. piensa que han ocurrido la zona en la última semana", '')
st.header("3. Resultados")

if forecast_e2_str!='':
    forecast_e1 = 2.0
    forecast_e2 = float(forecast_e2_str)
    array_expert = np.array([forecast_e1, forecast_e2])############################
    
    mu_new = ((array_expert.T@sigma_inv).sum() + (sigma_0**(-2))*mu_0)/(sigma_inv.sum() + sigma_0**(-2))
    sigma_new = (sigma_inv.sum() + sigma_0**(-2))**(-1)

    df_list = [[-12.116763, -77.044949, 5.0],
           [-12.116747, -77.043542, mu_new],
           [-12.116325, -77.045386, 2.0]]

    hm_wide = HeatMap(df_list,
                  min_opacity=0.1,
                  radius=30,
                  gradient = {0.2:'cyan', 0.6: 'purple', 0.8:'red',1:'yellow'}, use_local_extrema=True,
                  name='Delitos denunciados'
                 )

    f = folium.Figure(height=750)
    hmap = folium.Map(location=[-12.116747, -77.043542], zoom_start=17, tiles='stamentoner', min_zoom=16, max_zoom=20)
    hmap.add_child(hm_wide)
    f.add_child(hmap)
    scatter_map = folium.CircleMarker(location=[-12.116747, -77.043542],
                            radius=25,
                            weight=5, color='red').add_to(hmap)

    st_folium(f)


    df_show = df_gsectorial.merge(df_e1, left_index=True, right_index=True)
    df_show = df_show.merge(edited_df_e2, left_index=True, right_index=True)
    st.table(df_show)

    dfsummary = pd.DataFrame([mu_0, forecast_e1, forecast_e2, mu_new],
                             index=['Reporte inicial','Reporte del ciudadano 1','Reporte del ciudadano 2','Reporte final del modelo'],
                             columns=['Reportes'])
    st.table(dfsummary)
