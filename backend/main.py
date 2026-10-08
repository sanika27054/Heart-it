"""FastAPI backend for the Heart Disease Predictor."""

import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import httpx
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 25

app = FastAPI(title="Heart-it API", version="1.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Authorization", "Content-Type", "apikey"],
)

_model = None
_metrics = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Run `python train_model.py` first to train and save it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as metrics_file:
            _metrics = json.load(metrics_file)
    return _metrics


def supabase_settings() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not anon_key:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.",
        )
    return url, anon_key


def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")

    url, anon_key = supabase_settings()
    try:
        response = httpx.get(
            f"{url}/auth/v1/user",
            headers={"apikey": anon_key, "Authorization": authorization},
            timeout=10,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, detail="Could not reach Supabase Auth."
        ) from exc

    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.")
    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail=f"Supabase Auth returned HTTP {response.status_code}.",
        )

    user = response.json()
    if not isinstance(user, dict) or not user.get("id"):
        raise HTTPException(status_code=502, detail="Supabase Auth returned an invalid user.")
    return user


def supabase_data_request(
    method: str,
    path: str,
    token: str,
    *,
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    prefer: str | None = None,
) -> httpx.Response:
    url, anon_key = supabase_settings()
    headers = {
        "apikey": anon_key,
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    try:
        response = httpx.request(
            method,
            f"{url}/rest/v1/{path}",
            headers=headers,
            params=params,
            json=json_body,
            timeout=10,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503, detail="Could not reach Supabase Database."
        ) from exc

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=401, detail="Your session expired or cannot access this entry."
        )
    if response.is_error:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Supabase Database returned HTTP {response.status_code}. "
                "Check that the latest_entries table and row-level security policies are set up."
            ),
        )
    return response


class PatientData(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: str = Field(..., description="M or F")
    ChestPainType: str = Field(..., description="TA, ATA, NAP, or ASY")
    RestingBP: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    Cholesterol: int = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    FastingBS: int = Field(..., ge=0, le=1, description="1 if fasting blood sugar > 120 mg/dl, else 0")
    RestingECG: str = Field(..., description="Normal, ST, or LVH")
    MaxHR: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    ExerciseAngina: str = Field(..., description="Y or N")
    Oldpeak: float = Field(..., ge=-5, le=10, description="ST depression induced by exercise")
    ST_Slope: str = Field(..., description="Up, Flat, or Down")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 54,
                "Sex": "M",
                "ChestPainType": "ASY",
                "RestingBP": 140,
                "Cholesterol": 239,
                "FastingBS": 0,
                "RestingECG": "Normal",
                "MaxHR": 138,
                "ExerciseAngina": "Y",
                "Oldpeak": 1.2,
                "ST_Slope": "Flat",
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float


class LatestEntryRequest(BaseModel):
    patient: PatientData
    result: PredictionResponse


class ExtractedTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=150_000)


def _labelled_match(
    text: str,
    labels: tuple[str, ...],
    value_pattern: str,
) -> tuple[str, str] | None:
    alternatives = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    match = re.search(
        rf"(?i)(?<![\w])({alternatives})[ \t]*(?::|=|[ \t]+)[ \t]*({value_pattern})",
        text,
    )
    if match is None:
        return None
    return match.group(2).strip(), match.group(0).strip()


def _extract_patient_fields(
    text: str,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    fields: dict[str, Any] = {}
    sources: dict[str, str] = {}

    numeric_specs: dict[str, tuple[tuple[str, ...], float, float, bool]] = {
        "Age": (("Age", "Patient age"), 1, 120, True),
        "RestingBP": (("Resting blood pressure", "RestingBP", "Systolic blood pressure"), 1, 300, True),
        "Cholesterol": (("Total cholesterol", "Cholesterol total", "Serum cholesterol", "Cholesterol"), 1, 700, True),
        "MaxHR": (("Maximum heart rate", "MaxHR", "Max HR"), 60, 220, True),
        "Oldpeak": (("Oldpeak", "ST depression"), -5, 10, False),
    }
    for field, (labels, minimum, maximum, integer) in numeric_specs.items():
        found = _labelled_match(
            text,
            labels,
            r"-?\d+(?:\.\d+)?(?![\d.])",
        )
        if found is None:
            continue
        raw, evidence = found
        number = float(raw)
        is_integer = not integer or number.is_integer()
        if is_integer and minimum <= number <= maximum:
            fields[field] = int(number) if integer else number
            sources[field] = evidence

    categorical_specs: dict[
        str, tuple[tuple[str, ...], tuple[tuple[str, str], ...]]
    ] = {
        "Sex": (
            ("Sex", "Gender"),
            ((r"male|m\b", "M"), (r"female|f\b", "F")),
        ),
        "ChestPainType": (
            ("Chest pain type", "ChestPainType", "Chest pain"),
            (
                (r"atypical angina|ata\b", "ATA"),
                (r"non[- ]anginal pain|nap\b", "NAP"),
                (r"asymptomatic|asy\b", "ASY"),
                (r"typical angina|ta\b", "TA"),
            ),
        ),
        "RestingECG": (
            ("Resting ECG", "RestingECG"),
            (
                (r"normal(?: ecg)?\b", "Normal"),
                (r"st[- ]?t(?: wave)? abnormality|\bst\b", "ST"),
                (r"left ventricular hypertrophy|\blvh\b", "LVH"),
            ),
        ),
        "ExerciseAngina": (
            ("Exercise-induced angina", "Exercise angina", "ExerciseAngina"),
            ((r"yes|y\b|true", "Y"), (r"no|n\b|false", "N")),
        ),
        "ST_Slope": (
            ("ST slope", "ST segment slope", "ST_Slope"),
            ((r"upsloping|\bup\b", "Up"), (r"flat\b", "Flat"), (r"downsloping|\bdown\b", "Down")),
        ),
    }
    for field, (labels, values) in categorical_specs.items():
        alternatives = "|".join(
            re.escape(label) for label in sorted(labels, key=len, reverse=True)
        )
        value_pattern = "|".join(
            pattern for pattern, _ in sorted(values, key=lambda entry: len(entry[0]), reverse=True)
        )
        match = re.search(
            rf"(?i)(?<![\w])({alternatives})[ \t]*(?::|=|[ \t]+)[ \t]*({value_pattern})(?![\w])",
            text,
        )
        if match is None:
            continue
        normalized_value = match.group(2).lower()
        for pattern, normalized in values:
            if re.fullmatch(pattern, normalized_value, re.IGNORECASE):
                fields[field] = normalized
                sources[field] = match.group(0).strip()
                break

    fasting = _labelled_match(
        text,
        ("Fasting blood sugar", "Fasting blood glucose", "Fasting glucose", "FastingBS"),
        r"yes|no|y\b|n\b|true|false|-?\d+(?:\.\d+)?(?![\d.])",
    )
    if fasting is not None:
        raw, evidence = fasting
        normalized = raw.strip().lower()
        if normalized in {"yes", "y", "true"}:
            fields["FastingBS"] = 1
            sources["FastingBS"] = evidence
        elif normalized in {"no", "n", "false"}:
            fields["FastingBS"] = 0
            sources["FastingBS"] = evidence
        else:
            number = float(raw)
            if number in (0, 1):
                fields["FastingBS"] = int(number)
                sources["FastingBS"] = evidence
            elif 0 < number <= 700:
                fields["FastingBS"] = int(number > 120)
                sources["FastingBS"] = f"{evidence} (converted using >120 mg/dL)"

    missing = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
        "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
    ]
    missing = [field for field in missing if field not in fields]
    return fields, sources, missing


def _extraction_response(text: str) -> dict[str, Any]:
    fields, sources, unmatched = _extract_patient_fields(text)
    return {
        "fields": fields,
        "sources": sources,
        "unmatched_fields": unmatched,
        "message": "Only values explicitly found in the document are filled. Verify each value against the PDF before checking risk.",
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "Heart-it API is running."}


@app.get("/model-info")
def model_info():
    metrics = get_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="No metrics found. Run train_model.py first.")
    return {
        "accuracy": metrics["accuracy"],
        "test_samples": metrics["test_samples"],
        "dataset_samples": metrics["dataset_samples"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    model = get_model()
    row = pd.DataFrame([patient.model_dump()])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    return PredictionResponse(
        prediction=pred,
        label="Heart disease likely" if pred == 1 else "Heart disease unlikely",
        probability=round(prob, 4),
    )


@app.put("/account/latest-entry")
def save_latest_entry(
    entry: LatestEntryRequest,
    authorization: str | None = Header(default=None),
    user: dict[str, Any] = Depends(current_user),
):
    token = authorization.removeprefix("Bearer ") if authorization else ""
    saved_entry = {
        "patient": entry.patient.model_dump(mode="json"),
        "result": entry.result.model_dump(mode="json"),
    }
    supabase_data_request(
        "POST",
        "latest_entries",
        token,
        params={"on_conflict": "user_id"},
        json_body={
            "user_id": user["id"],
            "latest_entry": saved_entry,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        prefer="resolution=merge-duplicates,return=minimal",
    )
    return {"saved": True}


@app.get("/account/latest-entry")
def get_latest_entry(
    authorization: str | None = Header(default=None),
    user: dict[str, Any] = Depends(current_user),
):
    token = authorization.removeprefix("Bearer ") if authorization else ""
    response = supabase_data_request(
        "GET",
        "latest_entries",
        token,
        params={
            "select": "latest_entry,updated_at",
            "user_id": f"eq.{user['id']}",
            "limit": "1",
        },
    )
    rows = response.json()
    return {"entry": rows[0] if rows else None}


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")
    contents = await file.read(MAX_PDF_BYTES + 1)
    if len(contents) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF must be 10 MB or smaller.")
    if not contents:
        raise HTTPException(status_code=422, detail="The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(contents))
    except (PdfReadError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The PDF could not be read.") from exc
    if reader.is_encrypted:
        raise HTTPException(status_code=422, detail="Password-protected PDFs are not supported.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise HTTPException(status_code=413, detail="PDF must contain 25 pages or fewer.")

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No selectable text was found. Image-only scanned PDFs need OCR before upload.",
        )

    return _extraction_response(text)


@app.post("/extract-text")
def extract_text(request: ExtractedTextRequest):
    """Map locally extracted PDF/OCR text to intake fields without uploading the PDF."""
    return _extraction_response(request.text)
