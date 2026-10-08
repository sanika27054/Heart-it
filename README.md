# Heart Disease Risk Screening

A full-stack educational heart-disease screening project with a FastAPI API,
a static browser frontend, PDF field extraction, and Supabase sign-in with
per-account storage of the most recent intake and result.

The bundled data is the 918-record
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

> This is not a medical device and must not be used to diagnose, prevent, or
> treat disease. The score is a model estimate, not a person's measured
> probability. Do not deploy it for clinical use without appropriate clinical,
> privacy, and regulatory review.

## Project structure

```text
heart-disease-predictor/
├── backend/
│   ├── heart.csv
│   ├── train_model.py
│   ├── main.py
│   ├── requirements.txt
│   ├── supabase_schema.sql
│   └── model/
│       ├── pipeline.joblib
│       └── metrics.json
├── frontend/
│   ├── index.html
│   └── config.js
└── README.md
```

## Local setup

### 1. Create the Supabase project

1. Create a project at [supabase.com](https://supabase.com/).
2. In the Supabase SQL Editor, run `backend/supabase_schema.sql`. It creates
   the table for each user's latest entry and row-level security policies.
3. Enable email/password sign-in under **Authentication → Providers → Email**.
   Configure email confirmation and the allowed redirect URLs to include your
   local frontend URL and, later, your production URL.
4. Copy the project URL and the **publishable key** (or legacy anon key) from
   **Project Settings → API**. These are public client settings; never put a
   service-role key in the frontend.

### 2. Configure the frontend

Create your local `frontend/config.js` by copying
`frontend/config.example.js`:

```js
export const APP_CONFIG = {
  apiBaseUrl: "http://localhost:8000",
  supabaseUrl: "https://YOUR_PROJECT.supabase.co",
  supabaseAnonKey: "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY",
};
```

Replace the Supabase placeholders. The URL and publishable/anon key are used
by Supabase's browser authentication client and are not database admin
credentials. `frontend/config.js` is ignored by Git so your local project
settings are not committed; only the example template is tracked.

### 3. Install and run the backend

In PowerShell:

```powershell
cd "C:\path\to\heart-disease-predictor\backend"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_ANON_KEY = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"
$env:CORS_ORIGINS = "http://localhost:5500"

# Only needed if you changed the data or training code:
python train_model.py

python -m uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`; interactive API docs are at
`http://localhost:8000/docs`.

### 4. Run the frontend

In a second PowerShell window:

```powershell
cd "C:\path\to\heart-disease-predictor\frontend"
python -m http.server 5500
```

Open `http://localhost:5500`. Serve the frontend over HTTP rather than opening
the HTML file directly so browser modules and Supabase authentication work as
expected.

The app works in guest mode without Supabase sign-in; guest results are not
saved. Select **Sign in to save** to create an account or sign in. Once signed
in, the app restores the account's latest saved entry. New predictions replace
that one entry. Supabase row-level security ensures users can access only
their own row. PDFs are
processed in the browser: text is extracted locally, and image-only PDFs use
on-device OCR. The PDF itself is not uploaded or saved; extracted text is sent
through a local label filter, and only short snippets matching screening-field
labels are sent to the API. Extracted values are suggestions: check every
field before submitting. The browser needs internet access to load the PDF.js
and Tesseract.js libraries and OCR language data from their CDNs.

PDF import only fills values explicitly labeled in the report and shows the
matched label/value as review evidence. It does not turn a reference range or
an unrelated lab result into a model input. Missing or unrecognized values
stay blank; the screening model requires all 11 intake fields, including
symptoms and exercise/ECG measurements that routine lab reports often do not
contain. Enter missing values from an appropriate clinical source rather than
letting the importer guess.

## Deploying

- Host `frontend/` on a static HTTPS host and set `apiBaseUrl` in
  `frontend/config.js` to the deployed API origin.
- Deploy the FastAPI backend and set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and
  `CORS_ORIGINS` as deployment environment variables. Restrict `CORS_ORIGINS`
  to the exact frontend origin(s). Never use a Supabase service-role key for
  this app.
- Set the production site URL and redirect allow-list in Supabase Auth. Verify
  email confirmation, password recovery, and session behavior on the deployed
  domains.
- The bundled dataset has 918 records. Do not append repeated rows to make a
  larger-looking dataset: duplicated patient records can inflate test
  accuracy. Use a larger cohort only after checking its provenance, license,
  feature definitions, and overlap with the current data.
- The public [1,190-row package](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset)
  with similar fields is assembled from the same five historical cohorts as
  the bundled data, and its provenance indicates
  the current 918-row release already removed duplicate records. It is not
  included as a larger independent cohort; use it only after a row-level
  overlap audit and appropriate attribution.
- Health data is sensitive. A public deployment needs an appropriate privacy
  notice, retention/deletion process, secure hosting, and review of the
  regulations that apply to its users and location. This student project is
  not certified for handling real patient data.

## Model evaluation

The trainer selects a candidate using a validation subset, retrains it on the
development data, and evaluates accuracy once on a separate held-out test
split. The website does not display algorithm names or a model comparison. It
displays only test accuracy and states that a machine-learning model generated
the estimate. The current bundled artifact reports 89.1% accuracy on 184
held-out records from the 918-record dataset. Re-training updates
`backend/model/pipeline.joblib` and `backend/model/metrics.json`.

`GET /model-info` returns only the accuracy and the dataset/test sample counts.
`POST /predict` accepts the 11 intake fields. `POST /extract-pdf` extracts
recognized labeled values from a directly uploaded PDF; the frontend instead
uses `POST /extract-text` after reading PDF text or running browser OCR.
`GET /account/latest-entry` and `PUT /account/latest-entry` require a Supabase
access token and read or replace the signed-in user's latest saved entry.
