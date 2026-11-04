## Page info
import streamlit as st

st.title("🚀 Intro")

st.markdown("""
### ¿Qué hace esta app?

Estima la **probabilidad de que una póliza de seguro de auto tenga un siniestro** y explica
**por qué** el modelo le asigna ese riesgo. Es un proyecto de punta a punta: análisis exploratorio,
modelado, una API con FastAPI y este frontend en Streamlit.

### Datos

- Dataset público de Kaggle: *Car Insurance Claim Prediction* — **58,592 pólizas**, 1 fila por póliza.
- Variables del asegurado (antigüedad de la póliza, edad), del auto (modelo, motor, equipamiento de
  seguridad, NCAP) y de la zona (cluster y densidad de población).
- Variable objetivo `is_claim`: solo **~6.4%** de las pólizas tuvo un siniestro → problema muy desbalanceado.
- Split **70 / 15 / 15** (train / validación / test) antes de cualquier transformación, para evitar *leakage*.

> Las variables `policy_tenure`, `age_of_car` y `age_of_policyholder` vienen normalizadas (0–1) en el
> dataset original, sin unidades reales. Se leen como *más baja = más reciente / más joven*.

### Modelo

- **XGBoost** dentro de un `Pipeline` de scikit-learn (escalado, *target encoding* y *one-hot encoding*).
- Desbalance tratado con `scale_pos_weight` (sin SMOTE ni re-muestreo).
- Métrica principal: **PR-AUC**, porque con 6% de positivos la accuracy y el ROC-AUC dan una imagen
  demasiado optimista.
- Se comparó contra Random Forest: empate estadístico en score → se eligió XGBoost por ser
  **~4x más rápido** en inferencia y más liviano.

### Resultados en test (datos nunca vistos)

| Métrica | Modelo | Azar |
|---|---|---|
| PR-AUC | **0.097** | 0.062 (prevalencia) |
| Precisión en el top 20% más riesgoso | **10.4%** | 6.2% |
| Siniestros capturados en ese top 20% | **34%** | 20% |

El modelo **ordena** el riesgo ~1.6x mejor que el azar. Como clasificador individual es modesto
(la mayoría de las pólizas de "alto riesgo" igual no tienen siniestro), pero sirve para
**separar grupos**: el riesgo estimado varía ~9x entre los deciles más bajos y más altos.

### Cómo leer una predicción

- **Probabilidad**: calibrada (se corrigió la inflación que introduce `scale_pos_weight`), así que es
  comparable con la tasa real de siniestros.
- **Riesgo relativo**: probabilidad ÷ tasa promedio (~6.3%). Ej.: *2x* = el doble de riesgo que una póliza típica.
- **Alto riesgo**: la póliza supera el umbral de **9%**, que corresponde al 20% más riesgoso de la cartera.
- **Drivers (SHAP)**: las variables que más empujaron el riesgo hacia arriba ⬆️ o hacia abajo ⬇️
  *para esa póliza en particular*. Explican al modelo, no prueban causalidad.

### Cómo usar la app

Ve a **🪄 Predictor**, elige una póliza del set de test y verás su predicción, sus drivers y si
**realmente** tuvo un siniestro. Recuerda: una probabilidad de 20% significa que la mayoría de esas
pólizas *no* tienen siniestro; un caso aislado no confirma ni refuta al modelo.
""")
