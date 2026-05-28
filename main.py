import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# 1. Inicializar la app de FastAPI
app = FastAPI(
    title="API de Predicción de Default Crediticio - Proyecto Final",
    description="API en producción para evaluar el riesgo de impago bancario mediante un modelo XGBoost Pipeline."
)

# 2. Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Cargar el Pipeline entrenado (.pkl)
MODEL_PATH = "xgboost_default_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el archivo del modelo en: {MODEL_PATH}")

model_pipeline = joblib.load(MODEL_PATH)

# 4. Definir la estructura de datos de entrada
class LoanApplication(BaseModel):
    year: int = Field(..., description="Año de la solicitud", example=2019)
    loan_amount: float = Field(..., description="Monto del préstamo solicitado", example=150000.0)
    term: float = Field(..., description="Duración del préstamo en meses", example=36.0)
    income: float = Field(..., description="Ingreso anual del solicitante", example=55000.0)
    Credit_Score: int = Field(..., description="Puntaje crediticio del solicitante", example=650)
    dtir1: Optional[float] = Field(default=None, description="Debt-to-income ratio", example=35.5)
    property_value: Optional[float] = Field(default=None, description="Valor de la propiedad financiada", example=200000.0)
    LTV: Optional[float] = Field(default=None, description="Loan-to-value ratio", example=75.0)
    loan_limit: str = Field(..., description="cf o ncf", example="cf")
    Gender: str = Field(..., description="Male, Female, Joint, o Sex Not Available", example="Male")
    approv_in_adv: str = Field(..., description="pre o nopre", example="nopre")
    loan_type: str = Field(..., description="type1, type2, o type3", example="type1")
    loan_purpose: str = Field(..., description="p1, p2, p3, o p4", example="p1")
    Credit_Worthiness: str = Field(..., description="l1 o l2", example="l1")
    open_credit: str = Field(..., description="opc o nopc", example="nopc")
    business_or_commercial: str = Field(..., description="ob/c o nob/c", example="nob/c")
    Neg_ammortization: str = Field(..., description="neg_amm o not_neg", example="not_neg")
    interest_only: str = Field(..., description="int_only o not_int", example="not_int")
    lump_sum_payment: str = Field(..., description="lpsm o not_lpsm", example="not_lpsm")
    construction_type: str = Field(..., description="sb o mh", example="sb")
    occupancy_type: str = Field(..., description="pr, sr, o ir", example="pr")
    Secured_by: str = Field(..., description="home o land", example="home")
    total_units: str = Field(..., description="1U, 2U, 3U, o 4U", example="1U")
    co_applicant_credit_type: str = Field(..., description="CIB o EXP", example="CIB")
    age: str = Field(..., description="Rango de edad (ej: 25-34, 35-44...)", example="35-44")
    submission_of_application: str = Field(..., description="to_inst o not_inst", example="to_inst")
    Region: str = Field(..., description="North, South, Central, o North-East", example="North")
    Security_Type: str = Field(..., description="direct o indirect", example="direct")


@app.get("/")
def index():
    return {
        "status": "online",
        "message": "API de Predicción de Default Crediticio (XGBoost Pipeline) lista para operar."
    }


@app.post("/predict")
def make_prediction(application: LoanApplication):
    data_dict = application.model_dump()

    sentinel_value = -999.0
    for col in ['dtir1', 'property_value', 'LTV']:
        if data_dict[col] is None:
            data_dict[col] = sentinel_value

    input_df = pd.DataFrame([data_dict])

    prediction = model_pipeline.predict(input_df)[0]
    probabilities = model_pipeline.predict_proba(input_df)[0]
    default_probability = probabilities[1]

    return {
        "prediction_code": int(prediction),
        "prediction_label": "Default (Impago)" if prediction == 1 else "No Default (Saldado)",
        "default_probability": float(default_probability),
        "credit_risk_status": "HIGH RISK" if default_probability >= 0.5 else "LOW RISK"
    }