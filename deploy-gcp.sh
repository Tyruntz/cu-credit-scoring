#!/bin/bash
# ============================================================
# deploy-gcp.sh — Deploy cu-credit-scoring ke GCP Cloud Run
# Usage: chmod +x deploy-gcp.sh && ./deploy-gcp.sh
# ============================================================

set -e  # Exit on any error

# ─── CONFIG — SESUAIKAN INI ─────────────────────────────────
PROJECT_ID="project-97da1482-7227-4601-95b"       # ganti dengan GCP Project ID lo
SERVICE_NAME="cu-credit-scoring"
REGION="asia-southeast2"               # Jakarta region
IMAGE_NAME="asia-southeast2-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/app"
# ─────────────────────────────────────────────────────────────

echo "🚀 Starting deploy ke GCP Cloud Run..."
echo "   Project  : $PROJECT_ID"
echo "   Service  : $SERVICE_NAME"
echo "   Region   : $REGION"
echo ""

# 1. Set active project
echo "📌 Step 1/4: Set GCP project..."
gcloud config set project $PROJECT_ID

# 2. Enable APIs yang dibutuhkan (idempotent, aman dijalanin berkali-kali)
echo "⚙️  Step 2/4: Enable required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com --quiet

# 3. Build & push image ke Container Registry via Cloud Build
# (gak perlu Docker di local — build langsung di GCP)
echo "🔨 Step 3/4: Build image via Cloud Build & push ke GCR..."
gcloud builds submit --tag $IMAGE_NAME .

# 4. Deploy ke Cloud Run
echo "☁️  Step 4/4: Deploy ke Cloud Run ($REGION)..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --port 8080 \
    --timeout 120 \
    --quiet

echo ""
echo "✅ Deploy sukses!"
echo ""

# Ambil URL service yang baru di-deploy
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)')

echo "🌐 Backend URL: $SERVICE_URL"
echo ""
echo "🧪 Test health check:"
echo "   curl $SERVICE_URL/"
echo ""
echo "📝 Update frontend index.html:"
echo "   Ganti URL fetch di index.html dari kuberns.cloud ke:"
echo "   $SERVICE_URL/api/predict"
