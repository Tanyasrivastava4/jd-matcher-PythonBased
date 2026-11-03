"""
main.py
=======
FastAPI Application for ATS Job Description (JD) to Resume Matching.
Powered by Qdrant Vector Database, FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime),
and pure Python ATS extraction engine.
"""

import os
import io
import re
import time
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

from run_pipeline import process_resume
from scripts.ingest_to_qdrant import build_candidate_search_text

# Load environment variables from .env
load_dotenv()

# Directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
RESUMES_DIR = BASE_DIR / "resumes"
FLAT_JSON_DIR = BASE_DIR / "flat_json"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2")

app = FastAPI(
    title="AI Job Description & Resume Matcher",
    description="Production-grade ATS Matcher using Qdrant Vector DB, FastEmbed & FlashRank (ONNX Runtime CPU)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global models initialized on startup
qdrant_client: Optional[QdrantClient] = None
embed_model: Optional[TextEmbedding] = None
ranker: Optional[Ranker] = None


@app.on_event("startup")
def startup_event():
    global qdrant_client, embed_model, ranker
    print("Connecting to Qdrant at", QDRANT_URL)
    qdrant_client = QdrantClient(url=QDRANT_URL)

    print(f"Loading FastEmbed model '{EMBED_MODEL_NAME}' (ONNX Runtime CPU)...")
    t0 = time.perf_counter()
    embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    print(f"✓ Embed model loaded in {time.perf_counter() - t0:.2f}s")

    print(f"Loading FlashRank Cross-Encoder '{RERANK_MODEL_NAME}' (ONNX Runtime CPU)...")
    t1 = time.perf_counter()
    ranker = Ranker(model_name=RERANK_MODEL_NAME, cache_dir=str(MODELS_DIR))
    print(f"✓ Cross-Encoder Re-ranker loaded in {time.perf_counter() - t1:.2f}s")


def extract_text_from_upload(filename: str, content: bytes) -> Tuple[str, str]:
    """
    Extracts raw text from an uploaded JD file.
    Supports:
      - Digital PDF (PyMuPDF native text extraction)
      - Scanned / Image PDF (PyMuPDF rendering + Tesseract OCR fallback)
      - Direct images (.png, .jpg, .jpeg, .jfif, .webp, .tiff) via Tesseract OCR
      - Word documents (.docx, .doc via python-docx / LibreOffice)
      - Plain text (.txt, .md)
    Returns:
      (extracted_text, extraction_method)
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        import fitz
        from PIL import Image
        import pytesseract

        doc = fitz.open(stream=content, filetype="pdf")
        text_parts = []

        # 1. Try digital text extraction first
        for page in doc:
            t = page.get_text().strip()
            if t:
                text_parts.append(t)

        full_text = "\n\n".join(text_parts).strip()

        # 2. If scanned / image PDF (less than 40 chars digital text), run OCR on all pages
        if len(full_text) < 40 and len(doc) > 0:
            ocr_parts = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    ocr_parts.append(ocr_text)
            full_text = "\n\n".join(ocr_parts).strip()
            return full_text, "Scanned PDF (Tesseract OCR Engine)"

        return full_text, "Digital PDF (Native Vector Text)"

    elif ext in [".png", ".jpg", ".jpeg", ".jfif", ".webp", ".tiff"]:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img).strip()
        return text, "Image Document (Tesseract OCR Engine)"

    elif ext in [".docx"]:
        import docx
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip(), "Word Document (.docx)"

    elif ext in [".doc"]:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_doc = Path(tmpdir) / "temp.doc"
            tmp_doc.write_bytes(content)
            try:
                subprocess.run(
                    ["soffice", "--headless", "--convert-to", "docx", str(tmp_doc), "--outdir", tmpdir],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15
                )
                converted_docx = Path(tmpdir) / "temp.docx"
                if converted_docx.exists():
                    import docx
                    d = docx.Document(str(converted_docx))
                    text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
                    return text.strip(), "Word Document (.doc Converted)"
            except Exception:
                pass
        return content.decode("utf-8", errors="ignore").strip(), "Word Document (.doc Raw)"

    elif ext in [".txt", ".md"]:
        return content.decode("utf-8", errors="ignore").strip(), "Plain Text Document"

    else:
        raise ValueError(
            f"Unsupported JD file format: {ext}. "
            "Supported formats: PDF (Digital & Scanned), DOCX, DOC, TXT, PNG, JPG, JPEG, JFIF, WEBP."
        )


def extract_experience_from_text(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Parses natural language experience strings typed by recruiters:
    e.g. '0-1', '2+', '5-10yr', '2-6', '3 yrs', 'min 4 years', 'freshers', 'up to 5 yrs'.
    Handles standard hyphens (-), en-dashes (–), em-dashes (—), and 'to'.
    """
    if not text:
        return None, None, ""
    s = text.strip().lower()

    if s in ["any", "all", "none", "na", "n/a", "no filter"]:
        return None, None, "Any Experience"

    # Fresher check
    if re.search(r"\b(freshers?|entry\s*level|trainee|intern)\b", s):
        return 0.0, 1.0, "0-1 years (Fresher/Entry)"

    dash_re = r"(?:[\-–—‒―−]|to)"

    # Range: '0-1', '2-5', '2–6', '5-10yr', '5-10 yrs', '5 to 10 years'
    m = re.search(r"(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
    if m:
        mn, mx = float(m.group(1)), float(m.group(2))
        return mn, mx, f"{mn} - {mx} years"

    # Plus: '2+', '2 +', '3+ yr', '5+ yrs'
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)?", s)
    if m:
        mn = float(m.group(1))
        return mn, None, f"{mn}+ years"

    # Min / At least: 'min 3', 'at least 4 yrs'
    m = re.search(r"(?:min|minimum|at\s*least|more\s*than|over|>=|>)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
    if m:
        mn = float(m.group(1))
        return mn, None, f"{mn}+ years"

    # Max / Up to: 'max 4', 'up to 3 yrs', '<=5', '<5'
    m = re.search(r"(?:max|maximum|up\s*to|upto|<=|<)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
    if m:
        mx = float(m.group(1))
        return 0.0, mx, f"Up to {mx} years"

    # Single number: '3', '3yr', '3 yrs', '3 years'
    m = re.search(r"^(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?$", s)
    if m:
        mn = float(m.group(1))
        return mn, None, f"{mn}+ years"

    return None, None, ""


def extract_experience_from_jd(jd_text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Auto-detects experience requirements from full Job Description text.
    Handles ranges ('3-5 years', '2–6 Years'), plus ('2+ years'), min ('at least 4 years'), and freshers.
    Supports all Unicode dash variations (hyphen, en-dash, em-dash) and multi-line headers ('Experience \n 2–6 Years').
    Avoids false positives from version numbers (Python 3.10) or shift hours (24/7).
    """
    if not jd_text:
        return None, None, ""
    text = jd_text.lower()
    dash_re = r"(?:[\-–—‒―−]|to)"

    # Priority 1: Explicit total/overall/relevant experience
    p1 = re.search(r"(?:total|overall|relevant)\s+experience\s*(?:of|is|:|-)?\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", text)
    if p1:
        mn, mx = float(p1.group(1)), float(p1.group(2))
        return mn, mx, f"{mn} - {mx} years"

    p1_plus = re.search(r"(?:total|overall|relevant)\s+experience\s*(?:of|is|:|-)?\s*(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)?", text)
    if p1_plus:
        mn = float(p1_plus.group(1))
        return mn, None, f"{mn}+ years"

    # Priority 2: Experience field: 'experience: 3-5 yrs', 'Experience \n 2–6 Years', 'Experience - 2 to 4'
    p2 = re.search(r"\bexperience\s*(?:[:\-–—\n]|\s+)\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", text)
    if p2:
        mn, mx = float(p2.group(1)), float(p2.group(2))
        return mn, mx, f"{mn} - {mx} years"

    p2_plus = re.search(r"\bexperience\s*(?:[:\-–—\n]|\s+)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)?", text)
    if p2_plus:
        mn = float(p2_plus.group(1))
        if mn > 0:
            return mn, None, f"{mn}+ years"

    # Priority 3: 'X to Y years of experience' or 'X-Y years experience'
    p3 = re.search(r"(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text)
    if p3:
        mn, mx = float(p3.group(1)), float(p3.group(2))
        return mn, mx, f"{mn} - {mx} years"

    # Priority 4: 'X+ years of experience' or 'at least X years of experience'
    p4 = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text)
    if p4:
        mn = float(p4.group(1))
        return mn, None, f"{mn}+ years"

    p4_min = re.search(r"(?:minimum|min|at\s*least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text)
    if p4_min:
        mn = float(p4_min.group(1))
        return mn, None, f"{mn}+ years"

    # Priority 5: Fresher
    if re.search(r"\b(freshers?\s*(?:welcome|can\s*apply|only)?|entry\s*level(?:\s*role|\s*position|\s*candidate)?)\b", text):
        return 0.0, 1.0, "0 - 1 years (Fresher/Entry)"

    # Priority 6: Fallback pattern: 'X-Y years' or 'X+ years'
    p6 = re.search(r"\b(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text)
    if p6:
        mn, mx = float(p6.group(1)), float(p6.group(2))
        return mn, mx, f"{mn} - {mx} years"

    p6_plus = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\b", text)
    if p6_plus:
        mn = float(p6_plus.group(1))
        return mn, None, f"{mn}+ years"

    return None, None, ""


def extract_rerank_query(jd_text: str, max_chars: int = 1500) -> str:
    """
    Extracts a high-signal search query from the JD for the Cross-Encoder.
    Prioritizes Role, Technical Skills, Key Responsibilities, and Qualifications,
    bypassing repetitive corporate 'About Us' boilerplate if present.
    """
    if not jd_text:
        return ""
    if len(jd_text) <= max_chars:
        return jd_text

    # Extract title/role line
    title_line = ""
    for line in jd_text.splitlines()[:10]:
        l_str = line.strip()
        if re.search(r"^(?:job\s+title|job\s+description|position(?:\s+title)?|designation|role)\s*[:\-–—]\s*(.+)$", l_str, re.I):
            title_line = l_str
            break
        elif re.search(r"\b(?:head\s*[-–—]|senior|lead|manager|engineer|developer|architect|director|specialist)\b", l_str, re.I) and len(l_str.split()) <= 10:
            title_line = l_str
            break
    if not title_line:
        title_line = jd_text.splitlines()[0][:150].strip()

    # Priority 1: Key Responsibilities / Required Technical Skills
    match = re.search(
        r"\b(?:Key\s+Responsibilities|What\s+We\s+Are\s+Looking\s+For|Required\s+Technical\s+Skills|Must\s+Have\s+Skills|Technical\s+Skills|Key\s+Requirements|Required\s+Experience)\b",
        jd_text,
        re.IGNORECASE
    )
    if not match:
        match = re.search(
            r"\b(?:Job\s+Summary|Role\s+Summary|Role\s+overview|About\s+the\s+role)\b",
            jd_text,
            re.IGNORECASE
        )

    if match:
        start_pos = match.start()
        tech_chunk = jd_text[start_pos:start_pos + (max_chars - len(title_line) - 10)].strip()
        return f"{title_line}\n\n{tech_chunk}"

    return f"{title_line}\n\n{jd_text[:max_chars]}"


def resolve_experience_filter(
    experience_text: Optional[str] = None,
    jd_text: Optional[str] = None,
    legacy_min_exp: Optional[float] = None
) -> Dict[str, Any]:
    """
    Resolves the active experience filter with precedence:
      1. Explicit recruiter text input (e.g. '0-1', '2+', '5-10yr')
      2. Auto-detection from JD text/file
      3. Legacy min_experience parameter
      4. None (open to all candidates)
    """
    # 1. Check explicit experience_text
    if experience_text and experience_text.strip():
        s = experience_text.strip().lower()
        if s in ["any", "all", "none", "no filter"]:
            return {
                "source": "manual_override",
                "raw_input": experience_text,
                "min_experience": None,
                "max_experience": None,
                "label": "Any Experience (All candidates)"
            }
        mn, mx, lbl = extract_experience_from_text(experience_text)
        if mn is not None or mx is not None:
            return {
                "source": "text_input",
                "raw_input": experience_text,
                "min_experience": mn,
                "max_experience": mx,
                "label": lbl
            }

    # 2. Check JD auto-detect
    if jd_text:
        mn, mx, lbl = extract_experience_from_jd(jd_text)
        if mn is not None or mx is not None:
            return {
                "source": "jd_auto_detected",
                "raw_input": lbl,
                "min_experience": mn,
                "max_experience": mx,
                "label": lbl
            }

    # 3. Check legacy min_experience
    if legacy_min_exp is not None and legacy_min_exp > 0.0:
        return {
            "source": "legacy_param",
            "raw_input": str(legacy_min_exp),
            "min_experience": float(legacy_min_exp),
            "max_experience": None,
            "label": f"{legacy_min_exp}+ years"
        }

    return {
        "source": "none",
        "raw_input": "",
        "min_experience": None,
        "max_experience": None,
        "label": "Any Experience"
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the main interactive recruiter dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/stats")
async def get_stats():
    """Returns database, collection and two-stage pipeline telemetry."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Database not ready")

    try:
        col_info = qdrant_client.get_collection(COLLECTION_NAME)
        points_count = col_info.points_count
        status = col_info.status.value if hasattr(col_info.status, "value") else str(col_info.status)
    except Exception as e:
        points_count = 0
        status = f"Error: {e}"

    return {
        "collection": COLLECTION_NAME,
        "points_count": points_count,
        "status": status,
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dim": 384,
        "distance": "Cosine",
        "engine": "FastEmbed + ONNX Runtime (CPU)",
        "reranker_model": RERANK_MODEL_NAME if ranker else None,
        "pipeline": "Two-Stage Retrieval (Qdrant Bi-Encoder + FlashRank Cross-Encoder)"
    }


@app.post("/api/match")
async def match_jd(
    request: Request,
    jd_text: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None),
    limit: Optional[int] = Form(None),
    min_experience: Optional[float] = Form(None),
    experience: Optional[str] = Form(None),
    experience_text: Optional[str] = Form(None),
    mandatory_skills: Optional[str] = Form(None),
    enable_rerank: Optional[bool] = Form(True)
):
    """
    Main endpoint for matching a Job Description against the 590 resumes.
    Two-Stage Architecture:
      Stage 1: Filtered Dense Vector Search via Qdrant (Sub-10ms)
      Stage 2: Cross-Encoder Re-Ranking via FlashRank ONNX (Sub-20ms)

    Supports BOTH:
      1. multipart/form-data (File upload or Form fields)
      2. application/json (Raw JSON body)
    """
    t_start = time.perf_counter()

    content_type = request.headers.get("content-type", "").lower()
    eff_limit = 10
    raw_exp_text = ""
    legacy_min_exp = None
    eff_mandatory_skills = ""
    query_text = ""
    source_filename = None
    extraction_method = "Pasted Text / JSON Body"
    eff_enable_rerank = True

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
        query_text = (body.get("jd_text") or "").strip()
        eff_limit = int(body.get("limit", 10))
        raw_exp_text = str(body.get("experience") or body.get("experience_text") or "").strip()
        if body.get("min_experience") is not None:
            try:
                legacy_min_exp = float(body.get("min_experience"))
            except (ValueError, TypeError):
                legacy_min_exp = None
        eff_mandatory_skills = str(body.get("mandatory_skills", "") or "")
        eff_enable_rerank = bool(body.get("enable_rerank", True))
    else:
        eff_limit = int(limit) if limit is not None else 10
        raw_exp_text = (experience_text or experience or "").strip()
        if min_experience is not None:
            try:
                legacy_min_exp = float(min_experience)
            except (ValueError, TypeError):
                legacy_min_exp = None
        eff_mandatory_skills = mandatory_skills or ""
        eff_enable_rerank = bool(enable_rerank) if enable_rerank is not None else True

        if jd_file and jd_file.filename:
            source_filename = jd_file.filename
            content = await jd_file.read()
            try:
                query_text, extraction_method = extract_text_from_upload(jd_file.filename, content)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read uploaded JD: {e}")
        elif jd_text:
            query_text = jd_text.strip()
            extraction_method = "Pasted Text"

    if not query_text or len(query_text) < 15:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid Job Description with at least 15 characters, or upload a JD document."
        )

    # 1. Resolve experience requirement (user text input vs JD auto-detection)
    exp_filter = resolve_experience_filter(
        experience_text=raw_exp_text,
        jd_text=query_text,
        legacy_min_exp=legacy_min_exp
    )

    # 2. Extract keywords from JD to highlight matching skills
    jd_lower = query_text.lower()

    # 3. Embed JD via FastEmbed ONNX
    t_embed_start = time.perf_counter()
    query_vector = list(embed_model.embed([query_text]))[0].tolist()
    t_embed = time.perf_counter() - t_embed_start

    # 4. Build Qdrant Filters
    filter_conditions = []

    # Apply experience bounds if any
    min_exp = exp_filter.get("min_experience")
    max_exp = exp_filter.get("max_experience")
    if min_exp is not None or max_exp is not None:
        range_args = {}
        if min_exp is not None:
            range_args["gte"] = min_exp
        if max_exp is not None:
            range_args["lte"] = max_exp
        filter_conditions.append(
            models.FieldCondition(
                key="total_experience",
                range=models.Range(**range_args)
            )
        )

    # Mandatory skills filter
    req_skills_list = [s.strip() for s in eff_mandatory_skills.split(",") if s.strip()]
    for skill in req_skills_list:
        filter_conditions.append(
            models.FieldCondition(
                key="skills",
                match=models.MatchValue(value=skill)
            )
        )

    query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

    # 5. Stage 1: Fast Candidate Retrieval via Qdrant Bi-Encoder
    t_search_start = time.perf_counter()
    initial_fetch_limit = max(eff_limit * 3, 25) if (eff_enable_rerank and ranker) else eff_limit
    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=initial_fetch_limit
    )
    t_search = time.perf_counter() - t_search_start

    # 6. Stage 2: Cross-Encoder Re-Ranking (FlashRank CPU)
    t_rerank = 0.0
    candidate_items = []

    if eff_enable_rerank and ranker and search_response.points:
        passages = []
        for pt in search_response.points:
            p = pt.payload or {}
            desigs = ", ".join(p.get("designation", [])[:4])
            skills = ", ".join(p.get("skills", [])[:25])
            exp = p.get("total_experience", 0.0)
            summary = (p.get("summary") or "")[:300]
            comp_lines = [c for c in p.get("company_details", []) if not c.startswith("EXPERIENCE") and not c.startswith("Total experience")]
            companies = " ".join(comp_lines[:4])
            text_block = f"Role: {desigs}. Experience: {exp} years. Skills: {skills}. Jobs: {companies}. Summary: {summary}"
            passages.append({"id": pt.id, "text": text_block, "pt": pt})

        t_rerank_start = time.perf_counter()
        rerank_query = extract_rerank_query(query_text, max_chars=1500)
        req = RerankRequest(query=rerank_query, passages=passages)
        reranked_results = ranker.rerank(req)
        t_rerank = time.perf_counter() - t_rerank_start

        # Stage 2: Hybrid re-ranking
        # Combine Dense Bi-Encoder Cosine (domain match) with Log-Scaled Cross-Encoder (fine-grained precision)
        ce_scores = [float(r["score"]) for r in reranked_results]
        cos_scores = [float(r["pt"].score) for r in reranked_results]

        import math
        log_ces = [math.log(1.0 + max(0.0, s) * 500.0) for s in ce_scores]
        min_lce, max_lce = min(log_ces), max(log_ces)
        spread_lce = max_lce - min_lce if max_lce > min_lce else 1.0

        min_cos, max_cos = min(cos_scores), max(cos_scores)
        spread_cos = max_cos - min_cos if max_cos > min_cos else 1.0

        for r, lce in zip(reranked_results, log_ces):
            norm_ce = (lce - min_lce) / spread_lce
            norm_cos = (float(r["pt"].score) - min_cos) / spread_cos
            r["hybrid_score"] = (0.60 * norm_cos) + (0.40 * norm_ce)

        reranked_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        candidate_items = [
            (r["pt"], float(r["score"]))
            for r in reranked_results[:eff_limit]
        ]
    else:
        candidate_items = [
            (pt, None)
            for pt in search_response.points[:eff_limit]
        ]

    # 7. Format Candidate Cards with Two-Stage Metrics
    raw_cosines = [float(pt.score) for pt, _ in candidate_items]
    cross_scores = [cs for _, cs in candidate_items if cs is not None]

    calibrated_scores = []
    if eff_enable_rerank and ranker and cross_scores and len(candidate_items) > 0:
        import math
        log_ces = [math.log(1.0 + max(0.0, cs) * 500.0) for cs in cross_scores]
        min_lce, max_lce = min(log_ces), max(log_ces)
        spread_lce = max_lce - min_lce if max_lce > min_lce else 1.0

        last_score = 99.0
        for idx, (pt, cs) in enumerate(candidate_items):
            cos = float(pt.score)
            lce = math.log(1.0 + max(0.0, cs) * 500.0) if cs is not None else min_lce
            norm_ce = (lce - min_lce) / spread_lce

            target = (cos * 100.0) + (norm_ce * 8.0) - 2.0
            if target >= last_score:
                target = last_score - 0.4
            target = max(40.0, min(98.0, round(target, 1)))
            calibrated_scores.append(target)
            last_score = target
    else:
        for pt, _ in candidate_items:
            calibrated_scores.append(min(100.0, max(0.0, round(float(pt.score) * 100.0, 1))))

    candidates = []
    for rank, ((pt, cross_score), match_percentage) in enumerate(zip(candidate_items, calibrated_scores), 1):
        payload = pt.payload or {}
        candidate_skills = payload.get("skills", []) or []

        matched_skills = [
            s for s in candidate_skills
            if s.lower() in jd_lower
        ]
        other_skills = [
            s for s in candidate_skills
            if s not in matched_skills
        ]

        raw_cosine = float(pt.score)

        candidates.append({
            "rank": rank,
            "id": pt.id,
            "match_score": match_percentage,
            "raw_score": round(raw_cosine, 4),
            "cross_encoder_score": round(cross_score, 4) if cross_score is not None else None,
            "name": payload.get("name") or "Candidate",
            "file_name": payload.get("file_name", ""),
            "email": payload.get("email"),
            "mobile_number": payload.get("mobile_number"),
            "linkedin": payload.get("linkedin"),
            "github": payload.get("github"),
            "total_experience": payload.get("total_experience", 0.0),
            "designations": payload.get("designation", []) or [],
            "company_details": payload.get("company_details", []) or [],
            "education": payload.get("EDUCATION", []) or [],
            "degrees": payload.get("degree", []) or [],
            "matched_skills": matched_skills,
            "other_skills": other_skills[:15],
            "total_skills_count": len(candidate_skills),
            "summary": payload.get("summary", ""),
            "projects": payload.get("projects", []) or [],
            "certification": payload.get("certification", []) or [],
        })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 1)

    try:
        col_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_db_points = col_info.points_count
    except Exception:
        total_db_points = len(list(FLAT_JSON_DIR.glob("*.json")))

    return {
        "success": True,
        "total_database_points": total_db_points,
        "total_matches": len(candidates),
        "experience_filter": exp_filter,
        "total_time_ms": total_time_ms,
        "embed_time_ms": round(t_embed * 1000, 1),
        "search_time_ms": round(t_search * 1000, 1),
        "rerank_time_ms": round(t_rerank * 1000, 1),
        "two_stage_rerank": bool(eff_enable_rerank and ranker),
        "reranker_model": RERANK_MODEL_NAME if (eff_enable_rerank and ranker) else None,
        "source_file": source_filename,
        "extraction_method": extraction_method,
        "char_count": len(query_text),
        "word_count": len(query_text.split()),
        "query_preview": query_text[:200] + ("..." if len(query_text) > 200 else ""),
        "extracted_text": query_text,
        "candidates": candidates
    }


@app.post("/api/extract-jd")
async def extract_jd_preview(jd_file: UploadFile = File(...)):
    """
    Extracts text and auto-detects experience requirements from any uploaded JD
    document or image (Digital PDF, Scanned PDF via OCR, PNG, JPG, DOCX, TXT)
    WITHOUT executing a candidate search. Allows users to inspect and evaluate the extracted data.
    """
    if not jd_file or not jd_file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await jd_file.read()
    try:
        extracted_text, method = extract_text_from_upload(jd_file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {e}")

    exp_info = resolve_experience_filter(jd_text=extracted_text)

    return {
        "success": True,
        "filename": jd_file.filename,
        "extraction_method": method,
        "char_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
        "detected_experience": exp_info,
        "extracted_text": extracted_text
    }


@app.get("/api/candidate/{candidate_id}")
async def get_candidate_details(candidate_id: int):
    """Retrieves full candidate payload and raw json by sequential counting ID."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Database not ready")

    records = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[candidate_id],
        with_payload=True
    )
    if not records:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return records[0].payload


@app.get("/api/resume/{file_name}")
async def download_original_resume(file_name: str):
    """Serves the original resume file (PDF/DOCX/DOC) for browser preview or download."""
    resume_path = RESUMES_DIR / file_name
    if not resume_path.exists():
        raise HTTPException(status_code=404, detail=f"Resume file not found: {file_name}")

    # Determine media type
    ext = resume_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jfif": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(
        path=str(resume_path),
        filename=file_name,
        media_type=media_type
    )


def get_next_resume_id() -> int:
    """Finds the next available sequential counting ID from flat_json/."""
    max_id = 0
    if FLAT_JSON_DIR.exists():
        for p in FLAT_JSON_DIR.glob("*.json"):
            prefix = p.name.split("_")[0]
            if prefix.isdigit():
                max_id = max(max_id, int(prefix))
    return max_id + 1


@app.post("/api/resumes/upload")
async def upload_and_ingest_resumes(files: List[UploadFile] = File(...)):
    """
    Upload and incrementally ingest 1 or more new resumes (PDF, DOCX, DOC, PNG, JPG, TXT):
      1. Determines the next sequential counting ID (e.g. 591, 592...).
      2. Saves original file into resumes/ directory.
      3. Parses resume through ATS pipeline into extracted_text/, segmented_text/, flat_json/.
      4. Generates 384D embedding via FastEmbed ONNX on CPU.
      5. Incrementally upserts to Qdrant vector database (no re-indexing of old resumes).
      6. Returns the parsed candidate profile and updated collection count.
    """
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Vector database not ready")
    if not embed_model:
        raise HTTPException(status_code=503, detail="Embedding model not ready")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    current_id = get_next_resume_id()
    ingested_resumes = []
    errors = []

    for upload_file in files:
        if not upload_file.filename:
            continue

        filename = upload_file.filename
        assigned_id = current_id
        t_start = time.perf_counter()

        try:
            # 1. Save original file to resumes/ directory
            dest_file = RESUMES_DIR / filename
            content = await upload_file.read()
            with open(dest_file, "wb") as f:
                f.write(content)

            # 2. Run ATS Parser (Layer 0 Ingestion/OCR, Layer 1 Segmentation, Layers 2-4 Assembly & Flat JSON)
            flat_record = process_resume(
                file_path=str(dest_file),
                resume_id=assigned_id,
                output_dir=str(FLAT_JSON_DIR),
                verbose=False
            )

            # 3. Build search document & embed with FastEmbed ONNX
            search_doc = build_candidate_search_text(flat_record)
            vector = list(embed_model.embed([search_doc]))[0].tolist()

            # 4. Incremental Upsert to Qdrant
            point = models.PointStruct(
                id=int(assigned_id),
                vector=vector,
                payload=flat_record
            )
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )

            t_total = time.perf_counter() - t_start
            name = flat_record.get("name") or "Candidate"
            exp = flat_record.get("total_experience", 0.0)
            skills = flat_record.get("skills", []) or []

            ingested_resumes.append({
                "id": assigned_id,
                "name": name,
                "file_name": filename,
                "total_experience": exp,
                "designations": flat_record.get("designation", []) or [],
                "skills_count": len(skills),
                "skills_sample": skills[:8],
                "email": flat_record.get("email"),
                "mobile_number": flat_record.get("mobile_number"),
                "parsing_time_seconds": round(t_total, 2)
            })

            current_id += 1

        except Exception as e:
            errors.append({"file_name": filename, "error": str(e)})

    # Fetch updated collection stats
    try:
        col_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_count = col_info.points_count
    except Exception:
        total_count = len(list(FLAT_JSON_DIR.glob("*.json")))

    return {
        "success": len(ingested_resumes) > 0,
        "ingested_count": len(ingested_resumes),
        "total_database_points": total_count,
        "ingested_resumes": ingested_resumes,
        "errors": errors
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on http://{HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
