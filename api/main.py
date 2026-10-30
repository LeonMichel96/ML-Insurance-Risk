# main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import Any, Dict, Optional, List, Union, Literal

from api.predictor import risk_prediction

app = FastAPI(title="Riesgo por póliza")

## Definicion de clase Poliza
# detecta errores de tipos de dato antes de ingresarlos al modelo
# corrige tipos de dato
# genera documentacion

SiNo = Literal["Yes", "No"]
class Poliza(BaseModel):
    policy_tenure: float
    age_of_car: float
    age_of_policyholder: float
    area_cluster: Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
                          "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22"]
    population_density: int
    make: Literal["1", "2", "3", "4", "5"]
    segment: Literal["A", "B1", "B2", "C1", "C2", "Utility"]
    model: Literal["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11"]
    fuel_type: Literal["CNG", "Diesel", "Petrol"]
    max_torque: float
    max_power: float
    engine_type: Literal["1.0 SCe", "1.2 L K Series Engine", "1.2 L K12N Dualjet", "1.5 L U2 CRDi",
                         "1.5 Turbocharged Revotorq", "1.5 Turbocharged Revotron", "F8D Petrol Engine",
                         "G12B", "K Series Dual jet", "K10C", "i-DTEC"]
    is_esc: SiNo
    is_adjustable_steering: SiNo
    is_tpms: SiNo
    is_parking_sensors: SiNo
    is_parking_camera: SiNo
    rear_brakes_type: Literal["Disc", "Drum"]
    transmission_type: Literal["Automatic", "Manual"]
    steering_type: Literal["Electric", "Manual", "Power"]
    gross_weight: int
    is_front_fog_lights: SiNo
    is_rear_window_wiper: SiNo
    is_rear_window_washer: SiNo
    is_rear_window_defogger: SiNo
    is_brake_assist: SiNo
    is_power_door_locks: SiNo
    is_central_locking: SiNo
    is_power_steering: SiNo
    is_driver_seat_height_adjustable: SiNo
    is_day_night_rear_view_mirror: SiNo
    is_ecw: SiNo
    is_speed_alert: SiNo
    ncap_rating: int
    
## Definicion de clases de respuesta
class Driver(BaseModel):
    variable: str
    valor_ingresado: Union[float, str]
    impacto: float
    efecto: str

class Prediccion(BaseModel):
    probabilidad: float
    riesgo_relativo: float
    alto_riesgo: bool
    drivers: List[Driver]

class ErrorInfo(BaseModel):
    code: int
    message: str

class PrediccionResponse(BaseModel):
    data: Optional[Prediccion]
    error: Optional[ErrorInfo]

## Formatos de respuesta
def error_response(code: int, message: str, status_code: Optional[int] = None) -> JSONResponse:
    sc = status_code if status_code is not None else code
    return JSONResponse(
        status_code=sc,
        content={"data": None, "error": {"code": code, "message": message}},
    )

## path health
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status":"ok"}

## path predict
@app.post("/predict", response_model=PrediccionResponse)
def predict(poliza: Poliza):
    
    # generar df input
    X = pd.DataFrame([poliza.model_dump()])
    
    # generar respuesta
    try:
        result = risk_prediction(X)        
        return {"data": result, "error": None}
    
    except Exception as e:
        return error_response(code=500, message=str(e), status_code=500)