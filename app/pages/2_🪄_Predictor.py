## Page principal
import streamlit as st
import pandas as pd
import requests
import json
from utils.read_parquet import load_parquet

API_URL = "http://127.0.0.1:8000/predict"

# Carga de datos de test
test_df = load_parquet()

# Page
st.title("🪄 Predicción Riesgo de Póliza")


# selección de póliza
st.header("Selección de póliza a predecir")
cols_vista = ['policy_tenure', 'age_of_car', 'age_of_policyholder', 'area_cluster', 'model',
              'is_claim', 'engine_type', 'fuel_type']

sel = st.dataframe(test_df[cols_vista],  on_select="rerun", selection_mode="single-row")
if len(sel.selection.rows)>0:
    row = sel.selection.rows[0]
    policy = test_df.iloc[row]

else:
    st.warning("#### Selecciona una póliza para continuar")
    st.stop()

st.subheader("Póliza seleccionada")
st.dataframe(policy.astype(str))  # mezcla números y texto: Arrow exige un tipo por columna
y = policy['is_claim']

# llamado a api
st.subheader("Predicción de riesgo")
payload = json.loads(policy.drop("is_claim").to_json())
try:
    response = requests.post(url=API_URL, json=payload, timeout=10)
except requests.exceptions.ConnectionError:
    st.error("No se pudo conectar con la API. ¿Está corriendo uvicorn?")
    st.stop()

if response.status_code != 200:
    st.error(f"Error de la API ({response.status_code}): {response.text}")
    st.stop()

# presentación de resultados
data = response.json()['data']
risk_p = data['probabilidad']
riesgo_relativo = data['riesgo_relativo']
prediction = data['alto_riesgo']
drivers = pd.DataFrame(data['drivers'])
drivers['valor_ingresado'] = drivers['valor_ingresado'].astype(str)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Probabilidad de siniestro", value=f"{risk_p:.2%}")
with col2:
    st.metric(label="Riesgo vs promedio", value=f"{riesgo_relativo:.1f}x")
with col3:
    st.metric(label="Clasificación", value="Alto riesgo" if prediction else "Normal")
with col4:
    st.metric(label="¿Tuvo siniestro?", value="Sí" if y == 1 else "No")

st.caption("El modelo estima riesgo, no predice el futuro: una póliza de alto riesgo tiene ~2x más "
           "chances de siniestro que el promedio, no la certeza de tenerlo. "
           "Por eso es normal ver pólizas de alto riesgo sin siniestro.")

st.dataframe(drivers, hide_index=True)
