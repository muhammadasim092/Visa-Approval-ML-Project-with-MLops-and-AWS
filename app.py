
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run
from pydantic import BaseModel

import os
from typing import Optional
from visa_approval_prediction.constants import APP_HOST, APP_PORT, MODEL_REGISTRY_DIR, MODEL_FILE_NAME
from visa_approval_prediction.pipline.prediction_pipeline import visaData, visaClassifier
from visa_approval_prediction.pipline.training_pipeline import TrainPipeline
from visa_approval_prediction.entity.shap_explainer import VisaShapExplainer
from visa_approval_prediction.utils.main_utils import load_objects

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shap_explainer = None

def get_shap_explainer():
    global shap_explainer
    if shap_explainer is None:
        model_path = os.path.join(MODEL_REGISTRY_DIR, MODEL_FILE_NAME)
        if os.path.exists(model_path):
            model = load_objects(file_path=model_path)
            shap_explainer = VisaShapExplainer(model)
    return shap_explainer


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None
        

    async def get_visa_data(self):
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "visa.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_visa_data()
        
        visa_data = visaData(
                                continent= form.continent,
                                education_of_employee = form.education_of_employee,
                                has_job_experience = form.has_job_experience,
                                requires_job_training = form.requires_job_training,
                                no_of_employees= form.no_of_employees,
                                company_age= form.company_age,
                                region_of_employment = form.region_of_employment,
                                prevailing_wage= form.prevailing_wage,
                                unit_of_wage= form.unit_of_wage,
                                full_time_position= form.full_time_position,
                                )
        
        visa_df = visa_data.get_visa_input_data_frame()

        model_predictor = visaClassifier()

        value = model_predictor.predict(dataframe=visa_df)[0]

        status = None
        if value == 1:
            status = "Visa-approved"
        else:
            status = "Visa Not-Approved"

        return templates.TemplateResponse(
            "visa.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


class VisaPredictRequest(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str
    company_age: int


def _build_insights(data: VisaPredictRequest, result: str):
    strengths = []
    weaknesses = []
    suggestions = []

    if data.has_job_experience == "Y":
        strengths.append("Applicant has prior job experience")
    else:
        weaknesses.append("No prior job experience")
        suggestions.append("Highlight any transferable skills or internships")

    if data.education_of_employee in ("Master's", "Doctorate"):
        strengths.append(f"High education level ({data.education_of_employee})")
    elif data.education_of_employee == "Bachelor's":
        strengths.append("Bachelor's degree")
    else:
        weaknesses.append(f"Education level ({data.education_of_employee}) may be insufficient")
        suggestions.append("Consider pursuing higher education or certifications")

    if data.prevailing_wage >= 70000:
        strengths.append(f"Competitive wage (${data.prevailing_wage:,.0f})")
    elif data.prevailing_wage < 40000:
        weaknesses.append(f"Low prevailing wage (${data.prevailing_wage:,.0f})")
        suggestions.append("Negotiate a higher wage offer")

    if data.full_time_position == "Y":
        strengths.append("Full-time position")
    else:
        weaknesses.append("Not a full-time position")
        suggestions.append("Consider applying for full-time roles")

    if data.no_of_employees >= 1000:
        strengths.append(f"Large company ({data.no_of_employees} employees)")
    elif data.no_of_employees < 50:
        weaknesses.append(f"Small company ({data.no_of_employees} employees)")

    if data.company_age >= 10:
        strengths.append(f"Established company ({data.company_age} years)")
    elif data.company_age < 3:
        weaknesses.append(f"Very new company ({data.company_age} years old)")
        suggestions.append("Provide extra documentation of company stability")

    if data.requires_job_training == "Y":
        weaknesses.append("Position requires job training")
    else:
        strengths.append("No additional job training required")

    return {"strengths": strengths, "weaknesses": weaknesses, "suggestions": suggestions}


@app.post("/predict")
async def predict_json(data: VisaPredictRequest):
    try:
        visa_data = visaData(
            continent=data.continent,
            education_of_employee=data.education_of_employee,
            has_job_experience=data.has_job_experience,
            requires_job_training=data.requires_job_training,
            no_of_employees=data.no_of_employees,
            company_age=data.company_age,
            region_of_employment=data.region_of_employment,
            prevailing_wage=data.prevailing_wage,
            unit_of_wage=data.unit_of_wage,
            full_time_position=data.full_time_position,
        )

        visa_df = visa_data.get_visa_input_data_frame()
        model_predictor = visaClassifier()
        value = model_predictor.predict(dataframe=visa_df)[0]

        result = "approved" if value == 0 else "denied"
        confidence = 85 if value == 0 else 78

        input_data = data.dict()
        explainer = get_shap_explainer()
        if explainer is not None:
            insights = explainer.explain(visa_df, input_data, result)
        else:
            insights = _build_insights(data, result)

        return JSONResponse(content={
            "result": result,
            "confidence": confidence,
            "insights": insights,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)

