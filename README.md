# AI-Powered ATS Resume Parser & Job Description Matcher

A production-grade, 100% pure Python ATS Resume Parser and Job Description (JD) Semantic Matching System. Runs completely on standard CPU with **$0 cloud/GPU costs** and sub-50ms matching latency.

---

## 🚀 Key Features

* **Sub-Second Resume Parsing**: Pure Python parser (0.37s / resume) extracting contact info, education, skills, work experience, projects, and certifications.
* **Dual Contacts Extraction**: Automatically extracts and preserves multiple phone numbers and multiple email addresses (joined with `" / "`).
* **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.doc` (via headless LibreOffice), and images (`.jpg`, `.png`, `.jfif`, `.jpeg` via Tesseract OCR).
* **Qdrant Vector Database**: Ingests dense embeddings with full JSON metadata payloads and payload indexes on `skills`, `total_experience`, and `degree`.
* **Zero GPU Dependencies**: Uses `FastEmbed` with `BAAI/bge-small-en-v1.5` running on `ONNX Runtime` with AVX2 CPU acceleration.
* **Interactive FastAPI Web UI**: Recruiter dashboard with drag-and-drop JD upload, paste text mode, intelligent experience auto-detection, must-have skills filter, and candidate cards.
* **Two-Stage Re-ranking Pipeline**: Combines Stage 1 fast Bi-Encoder vector retrieval in Qdrant with Stage 2 Cross-Encoder re-ranking (`FlashRank` ONNX) on CPU for deep word-for-word semantic alignment.
* **Smart Experience Auto-Detection**: Automatically extracts experience requirements directly from the Job Description (e.g. `3-5 years`, `2–6 Years`, `2+ yrs`, `freshers`), or allows free-form text input (`0-1`, `2+`, `5-10yr`, `min 4 yrs`). Zero rigid sliders needed.

---

## 🛠️ System Prerequisites

### 1. System Packages (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libreoffice tesseract-ocr libtesseract-dev
```

### 2. Qdrant Vector Database (Docker)
```bash
docker run -d --name qdrant-server \
  --restart always \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

---

## 📦 Installation & Setup

1. **Clone & Create Virtual Environment**:
   ```bash
   cd jd-matcher
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download SpaCy Language Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Customize port or database URL if needed:
   ```env
   HOST=0.0.0.0
   PORT=8000
   QDRANT_URL=http://localhost:6333
   COLLECTION_NAME=resumes
   EMBED_MODEL_NAME=BAAI/bge-small-en-v1.5
   ```

---

## 🏃 Running the Pipeline

### Step 1: Batch Parse Resumes (Layer 0 to Layer 4)
Place all candidate resumes inside `resumes/` folder, then run:
```bash
python run_pipeline.py --all
```
Outputs will be saved in:
- `extracted_text/`: Raw extracted text and OCR outputs
- `segmented_text/`: Section segmentation blocks
- `flat_json/`: Normalized structured candidate JSON records

### Step 2: Ingest into Qdrant Vector Database
Embed and index all parsed JSONs into Qdrant:
```bash
python scripts/ingest_to_qdrant.py
```

### Step 3: Start the FastAPI Web Server
```bash
python main.py
```
Or with Uvicorn CLI:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at: **`http://localhost:8000`**

---

## 🌐 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **`GET`** | `/` | Serves the interactive Recruiter Web UI |
| **`GET`** | `/api/stats` | Telemetry & Qdrant collection status |
| **`POST`** | `/api/match` | Matches a Job Description against candidate pool |
| **`GET`** | `/api/candidate/{id}` | Returns full parsed JSON payload for a candidate |
| **`GET`** | `/api/resume/{file_name}` | Downloads/streams the original resume file |

---

## 💡 How to Query `/api/match`

### Option A: JSON Body (Application to Application)
```bash
curl -X POST "http://localhost:8000/api/match" \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "Looking for Senior Data Scientist with 3-5 years experience in Python and PyTorch",
    "experience": "3-5",
    "limit": 10,
    "mandatory_skills": "Python"
  }'
```
> **Note on Experience**: The `"experience"` field is optional. If omitted, the API automatically parses experience (e.g. `3-5 years`, `2+ yrs`, `freshers`) directly from `"jd_text"`!

### Option B: Multipart / File Upload (PDF / DOCX)
```bash
curl -X POST "http://localhost:8000/api/match" \
  -F "jd_file=@/path/to/job_description.pdf" \
  -F "experience=2+" \
  -F "limit=10"
```

### Response Format:
```json
{
  "success": true,
  "total_matches": 10,
  "experience_filter": {
    "source": "jd_auto_detected",
    "raw_input": "3.0 - 5.0 years",
    "min_experience": 3.0,
    "max_experience": 5.0,
    "label": "3.0 - 5.0 years"
  },
  "total_time_ms": 32.4,
  "candidates": [
    {
      "rank": 1,
      "id": 142,
      "name": "Alok Kumar Sharma",
      "match_score": 80.2,
      "total_experience": 3.0,
      "file_name": "Naukri_AlokKumarSharma[2y_1m].pdf",
      "email": "alok.sharma@example.com",
      "mobile_number": "+91 9876543210",
      "matched_skills": ["Python", "PyTorch", "Machine Learning"],
      "other_skills": ["FastAPI", "Docker", "Git"]
    }
  ]
}
```

---

## 📥 Adding New Resumes (Incremental Ingestion)

When new resumes arrive, you do **not** need to re-index the entire database. You can add 1, 3, or 50 new resumes incrementally without touching existing records:

### Method 1: Web UI (Easiest)
1. Open the dashboard at `http://localhost:8000`.
2. Click the **"+ Add Resumes"** button in the top navigation bar.
3. Select or drag & drop 1 or more resume files (`.pdf`, `.docx`, `.doc`, `.png`, `.jpg`).
4. Click **"Start Ingestion"**. The resumes are parsed, numbered (e.g. 591, 592...), saved to `flat_json/`, embedded, and upserted directly into Qdrant in real-time.

### Method 2: REST API Endpoint (`POST /api/resumes/upload`)
```bash
curl -X POST "http://localhost:8000/api/resumes/upload" \
  -F "files=@/path/to/resume1.pdf" \
  -F "files=@/path/to/resume2.docx" \
  -F "files=@/path/to/resume3.png"
```

### Method 3: Python CLI Script
```bash
source venv/bin/activate

# Ingest specific resume files:
python scripts/add_resumes.py resume1.pdf resume2.docx

# Ingest all resumes from a folder:
python scripts/add_resumes.py --dir /path/to/new_resumes_folder/
```
