import json
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import WEB_DIR, METRICS_PATH, BEST_MODEL_PATH
from src.predictor import CreditRiskPredictor
from api.schemas import (
    CreditApplicantInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse
)

app = FastAPI(
    title="Enterprise ML Credit Risk Analytics API",
    description="Production REST API for Machine Learning credit default risk scoring & MLOps evaluation.",
    version="1.0.0"
)

# CORS Middleware (Enable for Web Dashboard access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = CreditRiskPredictor()
        print("✅ FastAPI Startup: CreditRiskPredictor loaded successfully.")
    except Exception as e:
        print(f"⚠️ FastAPI Startup Warning: Model not found. Train model first using run_pipeline.py. Details: {e}")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    model_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_name=type(predictor.model).__name__ if model_loaded else "None",
        api_version="1.0.0"
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Analytics & MLOps"])
def get_model_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=44, detail="Model metrics artifact not found. Please train the model first.")
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
    return metrics_data

@app.post("/predict", response_model=PredictionResponse, tags=["Inference Engine"])
def predict_single_applicant(applicant: CreditApplicantInput):
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=500, detail="Prediction Engine is uninitialized. Train the model first.")
    try:
        input_dict = applicant.dict()
        result = predictor.predict_single(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference Engine"])
async def predict_batch_file(file: UploadFile = File(...)):
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=500, detail="Prediction Engine is uninitialized.")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        contents = await file.read()
        df_batch = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        predictions = predictor.predict_batch(df_batch)
        
        low_count = sum(1 for p in predictions if p["risk_level"] == "LOW RISK")
        med_count = sum(1 for p in predictions if p["risk_level"] == "MEDIUM RISK")
        high_count = sum(1 for p in predictions if p["risk_level"] == "HIGH RISK")
        
        return {
            "total_processed": len(predictions),
            "low_risk_count": low_count,
            "medium_risk_count": med_count,
            "high_risk_count": high_count,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV Batch processing failed: {str(e)}")

# Mount static web directory for Dashboard UI
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/", tags=["Dashboard UI"])
def serve_dashboard():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Enterprise ML API is running. Web Dashboard under construction."}
