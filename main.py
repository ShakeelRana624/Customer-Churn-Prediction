from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import warnings
from preprocess import preprocess_input
import os
from mangum import Mangum

warnings.filterwarnings('ignore')

app = FastAPI(title="Customer Churn Prediction API")
handler = Mangum(app)

# Setup templates directory
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Load model
model_path = os.path.join(base_dir, "churn_model.pkl")
model = joblib.load(model_path)

# Pydantic model for JSON requests (optional, but good for API)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the frontend HTML form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: CustomerData):
    """API endpoint to predict churn from JSON payload."""
    input_df = preprocess_input(data.dict())
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "churn": "Yes" if prediction == 1 else "No",
        "probability": round(probability * 100, 2)
    }

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: str = Form(...) # Form fields stringify empty values
):
    """Handle form submission and return result to the HTML template."""
    
    # Pack the form data into a dict mimicking JSON structure
    data_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    input_df = preprocess_input(data_dict)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    churn_result = "Yes" if prediction == 1 else "No"
    probability_pct = round(probability * 100, 2)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result": churn_result, 
        "probability": probability_pct,
        "form_data": data_dict
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
