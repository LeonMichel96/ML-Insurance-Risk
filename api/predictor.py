# predictor.py
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import shap


## variables constantes
W = 14.85388
THRESHOLD = 0.09
TASA_BASE = 0.06307602282147559
BASE_DIR = Path(__file__).resolve().parent.parent

## cargar modelo
model_path = BASE_DIR / "models" / "xgb_pipeline.joblib"
pipeline = joblib.load(model_path)
preprocessor = pipeline.named_steps['preprocessing']
xgb_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(xgb_model)

## Funciones

# Preprocesamiento
def preprocessing(X):
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, 'toarray'):      # por si ColumnTransformer devolvio matriz sparse
        X_transformed = X_transformed.toarray()

    # nombres de las columnas ya transformadas (el OneHotEncoder expandio columnas)
    feature_names = preprocessor.get_feature_names_out()

    X_transformed = pd.DataFrame(
        X_transformed, columns=feature_names, index=X.index
    ).astype(float)

    return X_transformed

# Drivers principales del score de una póliza (lo que mostraría Streamlit)
def top_drivers(X, n=5):
    """Las n variables que más mueven el score de la póliza i (posición en X)."""
    
    # pasar X transformado
    # shap_values es un objeto que contiene valores (empujones de cada columna), valores base, copia de X_test transformed y nombres de col
    X_transformed = preprocessing(X)
    shap_values = explainer(X_transformed)
    
    # quitar prefijos: 'num_transformer__policy_tenure' -> 'policy_tenure'
    nombres_limpios = [col.split('__')[1] for col in X_transformed.columns]

    # columnas del one-hot -> variable original: 'segment_C1' -> 'segment'
    # recorre las 47 columnas que resultan del preprocessing
    # para los casos de ohe con mas de 2 columnas, a cada una le asigna el nombre original
    variable_original = []
    for name in nombres_limpios:                  # ← loop de AFUERA: los 47 nombres
        for column in X.columns:               # ← loop de ADENTRO: las 34 columnas originales
            if name == column or name.startswith(column + '_'):
                variable_original.append(column)
                break

    nombres_comunes = {
        'policy_tenure': 'Antigüedad de la póliza',
        'age_of_car': 'Edad del auto',
        'age_of_policyholder': 'Edad del asegurado',
        'area_cluster': 'Zona',
        'population_density': 'Densidad de población',
        'max_power': 'Potencia máxima',
        'max_torque': 'Torque máximo',
        'gross_weight': 'Peso bruto',
        'ncap_rating': 'Estrellas NCAP',
        'segment': 'Segmento',
        'model': 'Modelo',
        'engine_type': 'Tipo de motor',
        'make': 'Fabricante',
        'fuel_type': 'Combustible',
    }
    
    impacto = (
        pd.Series(shap_values.values[0], index=variable_original)
        .groupby(level=0).sum()                       # agrupa por nombre original con el objetivo de sumar las columnas del one-hot de una misma variable
        .sort_values(key=abs, ascending=False)        # ordena por tamaño del empujón, sin importar el signo
        .head(n)
        )
    
    return pd.DataFrame({
        'variable': [nombres_comunes.get(v, v) for v in impacto.index],
        'valor_ingresado': X.iloc[0][impacto.index].values,   # lo que el usuario puso en el formulario
        'impacto': impacto.values.astype(float).round(2),
        'efecto': np.where(impacto.values > 0, '⬆️ sube el riesgo', '⬇️ baja el riesgo'),
        })

# Realizar prediccion de la probabilidad de riesgo y categorizar
def prediction(X):
    """Regresa dos valores, probabilidad de riesgo y clasificación"""
    
    # transformar X
    X_transformed = preprocessing(X)
    
    # calcular probabilidades infladas
    p_inf = xgb_model.predict_proba(X_transformed)[:,1]
    
    # ajustar probabilidades
    odds = p_inf / (1 - p_inf)
    p_cal = (odds / W) / (1 + odds / W)
    
    # predicción final con threshold
    y_pred = np.where(p_cal >= THRESHOLD, 1, 0)
    
    return p_cal, y_pred

# Respuesta final (para una sola poliza)
def risk_prediction(X):
    """Función final que junta todos los outputs: probabilidad de riesgo, riesgo relativo, etiqueta y main drivers"""
    # probabilidad calibrada y label
    p_cal, y_pred = prediction(X)
    p_cal = float(p_cal[0])
    y_pred = bool(y_pred[0])
    
    # top drivers
    top_drivers_df = top_drivers(X)
    top_drivers_dict =top_drivers_df.to_dict(orient='records')
    
    final_dict = {
        "probabilidad": p_cal,
        "riesgo_relativo": round(p_cal/TASA_BASE,2),
        "alto_riesgo": y_pred,
        "drivers": top_drivers_dict
    }
    
    return final_dict