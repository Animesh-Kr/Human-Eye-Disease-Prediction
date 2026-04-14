# Cloud Run Deployment Guide
## OCT Retinal AI — FastAPI Edge Inference API

### Prerequisites
- Google account (Gmail works)
- Google Cloud SDK installed
- Docker Desktop installed and running

---

## Step 1 — Install Google Cloud SDK

Download from: https://cloud.google.com/sdk/docs/install

After install, run:
```bash
gcloud init
gcloud auth login
```

---

## Step 2 — Create a Google Cloud Project

Go to: https://console.cloud.google.com
Click "New Project" → name it `oct-retinal-api` → Create

Then set it as active:
```bash
gcloud config set project oct-retinal-api
```

Enable required APIs:
```bash
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

---

## Step 3 — Copy Files to Your Project Folder

Copy these three files into your project:
```
C:\Users\adim\Desktop\oct_scan\deployment_cloud\
    ├── main.py
    ├── Dockerfile
    └── requirements.txt
```

---

## Step 4 — Build and Push Docker Image

Open PowerShell in the `deployment_cloud/` folder:

```powershell
cd C:\Users\adim\Desktop\oct_scan\deployment_cloud

# Build the image using Google Cloud Build (no local Docker needed)
gcloud builds submit --tag gcr.io/oct-retinal-api/oct-inference:v1
```

This uploads your code to Google, builds the Docker image in the cloud,
and pushes it to Google Container Registry. Takes 3-5 minutes.

---

## Step 5 — Deploy to Cloud Run

```powershell
gcloud run deploy oct-retinal-api \
  --image gcr.io/oct-retinal-api/oct-inference:v1 \
  --platform managed \
  --region europe-west2 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120 \
  --max-instances 3
```

`europe-west2` = London region — closest to Newcastle, lowest latency.
`--allow-unauthenticated` = anyone can call the API without auth tokens.

After deploy you get a URL like:
```
https://oct-retinal-api-XXXX-nw.a.run.app
```

---

## Step 6 — Test the API

### Test health endpoint
```bash
curl https://oct-retinal-api-XXXX-nw.a.run.app/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true}
```

### Test prediction endpoint (PowerShell)
```powershell
curl -X POST "https://oct-retinal-api-XXXX-nw.a.run.app/predict" `
  -H "accept: application/json" `
  -F "file=@C:\Users\adim\Desktop\oct_scan\test_scan.jpg"
```

Expected response:
```json
{
  "prediction": "CNV",
  "confidence": 91.46,
  "all_probabilities": {"CNV": 91.46, "DME": 4.12, "DRUSEN": 3.87, "NORMAL": 0.55},
  "ood_flag": false,
  "ood_score": 12.4,
  "uncertainty_flag": false,
  "entropy": 0.18,
  "clinical_note": "Prediction within normal confidence range.",
  "latency_ms": 64.31,
  "model": "human_eye_fp32.onnx (237MB edge node)",
  "temperature": 1.05
}
```

### Test via browser (Swagger UI)
Open: `https://oct-retinal-api-XXXX-nw.a.run.app/docs`

FastAPI auto-generates an interactive test interface. Upload any OCT scan
and see the full JSON response.

---

## Free Tier Limits

Google Cloud Run free tier (per month, per project):
- 2 million requests
- 360,000 GB-seconds of memory
- 180,000 vCPU-seconds

Your API uses ~64ms per request. The free tier covers roughly
**2 million OCT scans per month** — more than enough for a portfolio project.

---

## What to Add to Your CV

After deployment, add this line under Experience or Projects:

> "Deployed containerised retinal OCT inference API on Google Cloud Run;
> FastAPI + ONNX Runtime serving 4-class predictions at 62.9ms with OOD
> detection and uncertainty flagging; 237MB edge model (88% compression
> from 2.07GB master via FP32 ONNX export)"

Add the live URL to:
- GitHub README (Links section)
- HuggingFace Space description
- MIDL paper (Data Availability section)
- Portfolio akumar-tech.me

---

## Troubleshooting

**"Permission denied" on gcloud commands:**
```bash
gcloud auth application-default login
```

**Build fails with memory error:**
```bash
gcloud builds submit --tag gcr.io/oct-retinal-api/oct-inference:v1 --machine-type=E2_HIGHCPU_8
```

**Model download slow on first request:**
The first request after cold start downloads the ONNX model from HuggingFace (~243MB).
This takes 30-60 seconds. Subsequent requests are instant.
To avoid cold starts, set `--min-instances 1` in the deploy command
(note: this costs ~$5/month, no longer free).
