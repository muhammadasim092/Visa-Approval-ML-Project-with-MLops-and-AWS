# Visa Approval Prediction - ML Project

An end-to-end machine learning project that predicts US visa approval outcomes. Features a full ML pipeline (data ingestion, validation, transformation, model training, evaluation, and model registry) with a FastAPI web interface and SHAP-based explainability.

## Features

- **ML Pipeline**: Automated training pipeline with data validation, transformation, and model selection
- **Local-First**: Runs entirely locally — no cloud dependencies (MongoDB, S3) required
- **SHAP Explainability**: Model predictions include SHAP-based insights showing which factors influenced the decision
- **Web Interface**: Clean FastAPI frontend for submitting visa applications and viewing predictions
- **Modal Training**: Optional cloud training via [Modal](https://modal.com) for faster iteration

## Tech Stack

- **ML**: scikit-learn, XGBoost, CatBoost, SHAP
- **Backend**: FastAPI, Uvicorn
- **Data**: pandas, numpy
- **Frontend**: Jinja2 templates, HTML/CSS/JS

## Project Structure

```
├── visa_approval_prediction/
│   ├── components/          # Pipeline stages (ingestion, validation, training, etc.)
│   ├── entity/              # Config, artifacts, estimator, SHAP explainer
│   ├── pipline/             # Training and prediction pipelines
│   ├── data_access/         # Local CSV data access
│   ├── constants/           # Project constants
│   ├── utils/               # Utility functions
│   ├── logger/              # Logging setup
│   └── exception/           # Custom exceptions
├── config/                  # Model and schema configs
├── notebook/                # Dataset (Visa_Predection_Dataset.csv)
├── templates/               # Frontend HTML
├── static/                  # Static assets
├── model_registry/          # Trained model storage (gitignored)
├── app.py                   # FastAPI application
├── demo.py                  # Training script
└── train_modal.py           # Modal cloud training script
```

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/muhammadasim092/Visa-Approval-ML-Project-with-MLops-and-AWS.git
cd Visa-Approval-ML-Project-with-MLops-and-AWS
pip install -r requirements.txt
```

### Train the Model

**Locally:**
```bash
python demo.py
```

**On Modal (cloud):**
```bash
pip install modal
modal setup
modal run train_modal.py
```

### Run the App

```bash
python app.py
```

Open **http://localhost:8080** in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web form for visa prediction |
| POST | `/` | Form-based prediction (HTML response) |
| POST | `/predict` | JSON prediction with SHAP insights |
| GET | `/train` | Trigger model training |

### Example `/predict` Request

```json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": 5000,
  "region_of_employment": "Northeast",
  "prevailing_wage": 80000,
  "unit_of_wage": "Year",
  "full_time_position": "Y",
  "company_age": 20
}
```

### Example Response

```json
{
  "result": "approved",
  "confidence": 85,
  "insights": {
    "strengths": [
      "Master's education moderately favors approval",
      "Having job experience slightly favors approval"
    ],
    "weaknesses": [],
    "suggestions": []
  }
}
```

## Dataset

The dataset contains **25,480 records** of US visa applications with features including education level, job experience, wage, company size, and more. Source: `notebook/Visa_Predection_Dataset.csv`
