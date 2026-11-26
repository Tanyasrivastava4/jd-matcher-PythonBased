"""
main.py
=======
FastAPI Application for ATS Job Description (JD) to Resume Matching.
Powered by Qdrant Vector Database, FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime),
and pure Python ATS extraction engine.
"""

import os
import sys
import io
import re
import time
import json
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set

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

# Load environment variables from .env
load_dotenv()

# Import ATS Resume Parser & Real-Time Qdrant Sync Engine
sys.path.insert(0, "/home/intileo/Desktop/python-based-parser/ats_parser")
from run_pipeline import process_resume
from qdrant_sync import sync_resume_to_qdrant

def parse_demographic_criteria(text: Optional[str]) -> Dict[str, Any]:
    """
    Parses recruiter notes or JD text for explicit gender and age filtering requirements.
    Avoids false positives from Equal Opportunity Employer (EEO) disclaimers.
    E.g.:
      - 'female candidates only, age below 35' -> {'gender': 'female', 'max_age': 35.0}
      - 'Gender: Female, Age: 25-32' -> {'gender': 'female', 'min_age': 25.0, 'max_age': 32.0}
      - 'Age Limit: Not exceeding 35' -> {'max_age': 35.0}
    """
    if not text or not text.strip():
        return {}
    res = {}

    # Strip EEO disclaimers to prevent false triggers on statements like 'equal opportunity employer welcoming women...'
    cleaned = re.sub(r"equal\s+opportunity\s+employer.*?(\n\n|$)", "", text, flags=re.I | re.DOTALL)
    cleaned = re.sub(r"affirmative\s+action.*?(\n\n|$)", "", cleaned, flags=re.I | re.DOTALL)

    # 1. Gender detection (explicit label or preference/requirement)
    m_g_label = re.search(r"\b(?:preferred\s+)?gender\s*[:\-–—]\s*(female|male|women|men)\b", cleaned, re.I)
    if m_g_label:
        g = m_g_label.group(1).lower()
        res["gender"] = "female" if ("fem" in g or "wom" in g) else "male"
    else:
        m_g_req = re.search(
            r"\b(?:only\s+(female|male)\s+(?:candidates?|applicants?)|"
            r"(female|male)\s+(?:candidates?|applicants?)\s+(?:only|preferred)|"
            r"(?:prefer|preference\s+for)\s+(?:female|women|male|men)|"
            r"female\s+only|women\s+only|male\s+only|men\s+only)\b",
            cleaned, re.I
        )
        if m_g_req:
            raw = m_g_req.group(0).lower()
            res["gender"] = "female" if ("fem" in raw or "wom" in raw) else "male"

    # 2. Age Range: e.g. 'age: 25-35', 'age 25 to 35', 'age between 25 and 35'
    m_range = re.search(
        r"\b(?:age\s*(?:criteria|limit|requirement)?\s*[:\-–—]?\s*|age\s+between\s+|between\s+)(\d{2})\s*(?:-|to|and)\s*(\d{2})\s*(?:years?|yrs|age)?\b",
        cleaned, re.I
    )
    if m_range and any(k in cleaned.lower() for k in ["age", "years", "yrs", "criteria", "limit"]):
        v1, v2 = float(m_range.group(1)), float(m_range.group(2))
        if 18 <= v1 <= 75 and 18 <= v2 <= 75:
            res["min_age"] = min(v1, v2)
            res["max_age"] = max(v1, v2)

    # 3. Age bounds (below / above / max / up to)
    if "min_age" not in res and "max_age" not in res:
        m_below = re.search(
            r"\b(?:age\s*(?:limit)?\s*[:\-–—]?\s*(?:not\s+exceeding|below|<|under|less\s+than|<=|max|up\s+to)\s*|max\s*age\s*[:\-–—]?\s*)(\d{2})\b",
            cleaned, re.I
        )
        if m_below:
            v = float(m_below.group(1))
            if 18 <= v <= 75:
                res["max_age"] = v

        m_above = re.search(
            r"\b(?:age\s*(?:limit)?\s*[:\-–—]?\s*(?:above|>|greater\s+than|>=|min|at\s+least)\s*|min\s*age\s*[:\-–—]?\s*)(\d{2})\b",
            cleaned, re.I
        )
        if m_above:
            v = float(m_above.group(1))
            if 18 <= v <= 75:
                res["min_age"] = v

    return res

# Directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
RESUMES_DIR = BASE_DIR / "resumes"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jinaai/jina-embeddings-v2-small-en")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "ms-marco-TinyBERT-L-2-v2")

# Generic soft skills & buzzwords excluded from technical scoring and matched-skill highlights
GENERIC_SOFT_SKILLS = {
    # Communication
    "communication", "verbal communication", "written communication", "communication skills",
    "effective communication", "interpersonal skills", "presentation skills", "public speaking",
    "active listening", "storytelling",

    # Leadership & Teamwork
    "leadership", "team leadership", "leadership skills", "teamwork", "team collaboration",
    "cross-functional collaboration", "collaboration", "collaborative", "team player",
    "team building", "team management", "people management", "mentoring", "coaching",
    "supervision", "delegation",

    # General Attributes & Work Traits
    "problem solving", "analytical thinking", "critical thinking", "creative problem solving",
    "adaptability", "flexibility", "agility", "resilience", "time management", "multitasking",
    "prioritization", "work ethic", "fast learner", "quick learner", "self motivated",
    "detail oriented", "attention to detail", "decision making", "conflict resolution",
    "stress management", "emotional intelligence", "hardworking", "punctuality",
    "organizational skills", "interpersonal", "positive attitude", "negotiation skills",
    "relationship building"
}

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
    embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME, max_length=2048)
    embed_model.model.tokenizer.enable_truncation(max_length=2048)
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

    # Range: '0-1', '2-5', '2–6', '5-10yr', '5-10 yrs', '5 to 10 years', '(4-6) years'
    for m in re.finditer(r"[\(\[]?\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*[\)\]]?\s*(?:years?|yrs?)?", s):
        mn, mx = float(m.group(1)), float(m.group(2))
        # Guard against calendar years (e.g. 2015-16, 2021-2024)
        if mn <= 35.0 and mx <= 35.0 and mn <= mx:
            return mn, mx, f"{mn} - {mx} years"

    # Plus: '2+', '2 +', '3+ yr', '5+ yrs'
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)?", s)
    if m:
        mn = float(m.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    # Min / At least: 'min 3', 'at least 4 yrs'
    m = re.search(r"(?:min|minimum|at\s*least|more\s*than|over|>=|>)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
    if m:
        mn = float(m.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    # Max / Up to: 'max 4', 'up to 3 yrs', '<=5', '<5'
    m = re.search(r"(?:max|maximum|up\s*to|upto|<=|<)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
    if m:
        mx = float(m.group(1))
        if mx <= 35.0:
            return 0.0, mx, f"Up to {mx} years"

    # Single number: '3', '3yr', '3 yrs', '3 years'
    m = re.search(r"^(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?$", s)
    if m:
        mn = float(m.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    return None, None, ""


def extract_experience_from_jd(jd_text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Auto-detects experience requirements from full Job Description text.
    Handles ranges ('3-5 years', '2–6 Years', '(4 – 6) years'), plus ('2+ years'), min ('at least 4 years'), and freshers.
    Supports all Unicode dash variations (hyphen, en-dash, em-dash) and multi-line headers ('Experience \n 2–6 Years').
    Avoids false positives from calendar years (2015-16), version numbers (Python 3.10), or shift hours (24/7).
    """
    if not jd_text:
        return None, None, ""
    text = jd_text.lower()
    dash_re = r"(?:[\-–—‒―−]|to)"

    # Priority 1: Explicit total/overall/relevant experience
    for p1 in re.finditer(r"(?:total|overall|relevant)\s+experience\s*(?:of|is|:|-)?\s*[\(\[]?\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*[\)\]]?\s*(?:years?|yrs?)?", text):
        mn, mx = float(p1.group(1)), float(p1.group(2))
        if mn <= 35.0 and mx <= 35.0 and mn <= mx:
            return mn, mx, f"{mn} - {mx} years"

    for p1_plus in re.finditer(r"(?:total|overall|relevant)\s+experience\s*(?:of|is|:|-)?\s*(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)?", text):
        mn = float(p1_plus.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    # Priority 2: Experience field: 'experience: 3-5 yrs', 'Experience \n 2–6 Years', 'Experience - 2 to 4'
    for p2 in re.finditer(r"\bexperience\s*(?:[:\-–—\n]|\s+)\s*[\(\[]?\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*[\)\]]?\s*(?:years?|yrs?)?", text):
        mn, mx = float(p2.group(1)), float(p2.group(2))
        if mn <= 35.0 and mx <= 35.0 and mn <= mx:
            return mn, mx, f"{mn} - {mx} years"

    for p2_plus in re.finditer(r"\bexperience\s*(?:[:\-–—\n]|\s+)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)?", text):
        mn = float(p2_plus.group(1))
        if 0 < mn <= 35.0:
            return mn, None, f"{mn}+ years"

    # Priority 3: 'X to Y years of experience' or 'X-Y years experience' or '(4 – 6) years of experience'
    for p3 in re.finditer(r"[\(\[]?\s*(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*[\)\]]?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text):
        mn, mx = float(p3.group(1)), float(p3.group(2))
        if mn <= 35.0 and mx <= 35.0 and mn <= mx:
            return mn, mx, f"{mn} - {mx} years"

    # Priority 4: 'X+ years of experience' or 'at least X years of experience'
    for p4 in re.finditer(r"(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text):
        mn = float(p4.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    for p4_min in re.finditer(r"(?:minimum|min|at\s*least)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|hands-on|industry|work)?\s*experience", text):
        mn = float(p4_min.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    # Priority 5: Fresher
    if re.search(r"\b(freshers?\s*(?:welcome|can\s*apply|only)?|entry\s*level(?:\s*role|\s*position|\s*candidate)?)\b", text):
        return 0.0, 1.0, "0 - 1 years (Fresher/Entry)"

    # Priority 6: Fallback pattern: 'X-Y years' or 'X+ years'
    for p6 in re.finditer(r"\b(\d+(?:\.\d+)?)\s*" + dash_re + r"\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", text):
        mn, mx = float(p6.group(1)), float(p6.group(2))
        if mn <= 35.0 and mx <= 35.0 and mn <= mx:
            return mn, mx, f"{mn} - {mx} years"

    for p6_plus in re.finditer(r"\b(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)\b", text):
        mn = float(p6_plus.group(1))
        if mn <= 35.0:
            return mn, None, f"{mn}+ years"

    return None, None, ""


def clean_jd_text(text: str) -> str:
    """
    Production Pre-Processor / Boilerplate Stripper:
    Strips leading corporate 'About Us', awards, and brand marketing boilerplate from the JD,
    preserving the job title, profile header, responsibilities, and desired skills.
    Ensures embeddings and cross-encoders evaluate actual job requirements rather than brand marketing.
    """
    if not text or len(text) < 150:
        return text

    # Check for title/profile line at the very top (e.g. '(Profile: Sales & Services)', 'Position: Consultant')
    top_header = ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:10]:
        if re.search(r"^\(?(?:profile|position|job\s*title|designation|role)\s*[:\-–—]\s*(.+)\)?$", l, re.I):
            top_header += l + "\n"
        elif re.search(r"\b(?:consultant|analyst|engineer|manager|developer|executive|specialist|officer|lead|director)\b", l, re.I) and len(l.split()) <= 8:
            top_header += l + "\n"

    # Search for job start anchors that indicate the actual job specification begins
    job_anchor = re.search(
        r"(?:^|\n)\s*(?:(?:job\s+description|designation|position|role|about\s+the\s+role|role\s+overview|job\s+summary|responsibilities|key\s+responsibilities|what\s+you(?:'\''|’)?ll\s+do|desired\s+skills?|desired\s+skill\s+set|required\s+skills?|key\s+requirements|qualifications?)\s*[:\-–—\n])",
        text,
        re.IGNORECASE
    )
    if job_anchor and job_anchor.start() > 100:
        clean_body = text[job_anchor.start():].strip()
        if top_header and not clean_body.lower().startswith(top_header.lower().strip()):
            return f"{top_header.strip()}\n\n{clean_body}"
        return clean_body

    return text


def clean_role_candidate(t: str) -> str:
    """
    Cleans raw extracted role text by stripping conversational prefixes,
    adjectival buzzwords, parenthetical locations, and trailing punctuation.
    """
    if not t:
        return ""
    t = t.strip()
    # Strip leading prefixes
    t = re.sub(r"^(?:for\s+the\s+(?:position|post|role)\s+of\s+|the\s+(?:position|role|post)\s+of\s+|post\s+of\s+|role\s+of\s+)", "", t, flags=re.I).strip()
    # Strip conversational buzzwords at start
    buzzwords = r"\b(?:a|an|the|and|passionate|highly\s+motivated|energetic|dynamic|seasoned|experienced|talented|results?[-–\s]+driven|results?[-–\s]+oriented|motivated|proven|exceptional|driven)\b"
    prev = None
    while prev != t:
        prev = t
        t = re.sub(rf"^{buzzwords}\s*", "", t, flags=re.I).strip()
    # Strip trailing location or details in parentheses like (Delhi NCR), (Remote)
    t = re.sub(r"\s*\([^\)]*\)", "", t).strip()
    # Strip trailing participle/adjective descriptors like 'proficient in...', 'experienced in...'
    t = re.sub(r"\s+(?:proficient|experienced|skilled|specializing|responsible|expert)\b.*$", "", t, flags=re.I).strip()
    # Strip trailing punctuation, colons, hyphens
    t = re.sub(r"\s*[:\-–—]\s*$", "", t).strip()
    t = re.sub(r"^[^\w]+|[^\w\s\-\&]+$", "", t).strip()
    return t


# ==============================================================================
# STANDARD INDUSTRY ROLE CATALOG (150+ Canonical Market Titles & Executive Archetypes)
# ==============================================================================
INDUSTRY_ROLE_CATALOG_PATTERNS = [
    # C-Level & Executives
    r"Chief Executive Officer|CEO",
    r"Chief Technology Officer|CTO",
    r"Chief Operating Officer|COO",
    r"Chief Financial Officer|CFO",
    r"Chief Product Officer|CPO",
    r"Chief Revenue Officer|CRO",
    r"Chief Commercial Officer|CCO",
    r"Chief Information Officer|CIO",
    r"Chief Information Security Officer|CISO",
    r"Chief Legal Officer|CLO|General Counsel",
    r"Chief People Officer|Chief Human Resources Officer|CHRO",
    r"Chief Data Officer|CDO",

    # Sales, Business Development & Partnerships
    r"Vice President[-–\s]+Business Development|VP[-–\s]+Business Development|VP[-–\s]+BD|AVP[-–\s]+Business Development",
    r"Vice President[-–\s]+Sales|VP[-–\s]+Sales|AVP[-–\s]+Sales|President[-–\s]+Sales",
    r"Head[-–\s]+Business Development|Director[-–\s]+Business Development|Business Development Director",
    r"Head[-–\s]+Sales|Director[-–\s]+Sales|Sales Director|Commercial Director",
    r"General Manager[-–\s]+Sales|General Manager[-–\s]+Business Development|GM[-–\s]+Sales",
    r"Business Development Manager|BD Manager|Business Development Lead|Lead Business Development",
    r"Regional Sales Manager|Regional Sales Director|Area Sales Manager|Zonal Sales Manager|Territory Sales Manager",
    r"National Sales Manager|Country Sales Manager|Corporate Sales Manager",
    r"Enterprise Sales Manager|Enterprise Account Executive|Senior Account Executive|Account Executive",
    r"Strategic Account Manager|Key Account Manager|Global Account Manager|Account Manager",
    r"Client Relationship Manager|Client Acquisition Manager|Client Success Manager",
    r"Channel Sales Manager|Partner Sales Manager|Alliance Manager|Partnership Manager",
    r"B2B Sales Manager|Direct Sales Manager|Institutional Sales Manager",
    r"Sales Operations Manager|Revenue Operations Manager|RevOps Manager",
    r"Inside Sales Manager|Inside Sales Representative|Sales Development Representative|SDR|BDR",
    r"Sales Manager|Senior Sales Executive|Sales Executive|Sales Specialist",

    # Data, Analytics & AI
    r"Head of Data|Director of Data Science|Director of Analytics|Director of Data Engineering",
    r"Chief Data Scientist|Staff Data Scientist|Principal Data Scientist",
    r"Lead Data Scientist|Senior Data Scientist|Data Scientist",
    r"Staff Machine Learning Engineer|Principal Machine Learning Engineer",
    r"Lead Machine Learning Engineer|Senior Machine Learning Engineer|Machine Learning Engineer|ML Engineer",
    r"MLOps Engineer|AI Engineer|Artificial Intelligence Engineer|Deep Learning Engineer|NLP Engineer|Computer Vision Engineer",
    r"Data Architect|Big Data Architect|Enterprise Data Architect",
    r"Lead Data Engineer|Senior Data Engineer|Data Engineer|Big Data Engineer|Data Platform Engineer",
    r"Analytics Engineer|Lead Analytics Engineer|Senior Analytics Engineer",
    r"Lead Data Analyst|Senior Data Analyst|Data Analyst",
    r"Lead BI Analyst|Senior BI Analyst|BI Analyst|Business Intelligence Analyst",
    r"Lead BI Developer|Senior BI Developer|BI Developer|Business Intelligence Developer",
    r"Power BI Developer|Power BI Specialist|Tableau Developer|Tableau Specialist",
    r"Quantitative Analyst|Quant Analyst|Data Governance Specialist|Data Steward",

    # Software Engineering, Cloud & DevOps
    r"Vice President[-–\s]+Engineering|VP[-–\s]+Engineering|AVP[-–\s]+Engineering",
    r"Head of Engineering|Director of Engineering|Engineering Director",
    r"Software Engineering Manager|Engineering Manager|Lead Engineering Manager",
    r"Enterprise Architect|Solutions Architect|Cloud Solutions Architect|Principal Solutions Architect",
    r"Software Architect|Principal Software Architect|Chief Software Architect",
    r"Technical Architect|Principal Architect|Lead Architect",
    r"Principal Software Engineer|Staff Software Engineer|Lead Software Engineer",
    r"Senior Software Engineer|Software Engineer|Senior Software Developer|Software Developer",
    r"Lead Full Stack Developer|Senior Full Stack Developer|Full Stack Developer",
    r"Lead Full Stack Engineer|Senior Full Stack Engineer|Full Stack Engineer",
    r"Lead Backend Developer|Senior Backend Developer|Backend Developer",
    r"Lead Backend Engineer|Senior Backend Engineer|Backend Engineer",
    r"Lead Frontend Developer|Senior Frontend Developer|Frontend Developer",
    r"Lead Frontend Engineer|Senior Frontend Engineer|Frontend Engineer",
    r"Python Developer|Senior Python Developer|Lead Python Developer",
    r"Java Developer|Senior Java Developer|Lead Java Developer",
    r"React Developer|Angular Developer|Node\.js Developer|Go Developer|Golang Developer",
    r"Lead Mobile Engineer|Senior Mobile Engineer|iOS Developer|Android Developer|Mobile App Developer",
    r"Lead DevOps Engineer|Senior DevOps Engineer|DevOps Engineer|DevOps Specialist",
    r"Site Reliability Engineer|SRE|Lead SRE|Senior SRE|Platform Engineer|Cloud Platform Engineer",
    r"Cloud Engineer|Cloud Infrastructure Engineer|Infrastructure Engineer|Systems Engineer",
    r"Cyber Security Engineer|Information Security Specialist|Security Engineer|SOC Analyst",
    r"Lead QA Engineer|Senior QA Engineer|QA Engineer|Quality Assurance Engineer",
    r"Software Development Engineer in Test|SDET|Lead SDET|Automation Test Engineer",

    # Legal, Secretarial & Corporate Governance
    r"General Manager[-–\s]+Legal and Company Secretary|GM[-–\s]+Legal and Company Secretary",
    r"Company Secretary and Legal Head|Legal Head and Company Secretary",
    r"Company Secretary|Qualified Company Secretary|Associate Company Secretary|CS",
    r"Head Legal|Head of Legal|Director Legal|VP Legal",
    r"General Legal Counsel|Legal Counsel|Senior Legal Counsel|Corporate Legal Counsel",
    r"Legal and Secretarial|Secretarial Officer|Secretarial Executive",
    r"Chief Compliance Officer|Compliance Officer|Head of Compliance|Compliance Manager",
    r"Legal Manager|Senior Legal Manager|Senior Legal Associate|Legal Associate|Corporate Lawyer|Advocate",

    # Finance, Accounting & Audit
    r"VP Finance|Director Finance|Head of Finance|Finance Director",
    r"Finance Manager|Senior Finance Manager|Financial Controller",
    r"Chartered Accountant|Qualified Chartered Accountant|CA",
    r"Senior Internal Auditor|Lead Internal Auditor|Internal Audit Manager|Head of Internal Audit",
    r"Internal Auditor|Audit Manager|Senior Audit Manager|Senior Auditor|Statutory Auditor",
    r"Tax Manager|Senior Tax Specialist|Treasury Manager|Accounts Manager",
    r"Senior Financial Analyst|Financial Analyst|FP&A Manager|FP&A Analyst",

    # Product & Program Management
    r"VP Product Management|VP Product|Director of Product Management|Director of Product",
    r"Principal Product Manager|Staff Product Manager|Lead Product Manager",
    r"Senior Product Manager|Product Manager|Associate Product Manager|Technical Product Manager",
    r"Director of Program Management|Program Manager|Lead Program Manager|Technical Program Manager",
    r"Senior Project Manager|Project Manager|IT Project Manager|Technical Project Manager",
    r"Scrum Master|Agile Coach|Delivery Manager|Release Manager",

    # Human Resources & Talent Acquisition
    r"VP Human Resources|VP HR|Director Human Resources|Head of HR|HR Director",
    r"Senior HR Business Partner|HR Business Partner|HRBP|Corporate HRBP",
    r"Talent Acquisition Lead|Head of Talent Acquisition|Director of Talent Acquisition",
    r"Talent Acquisition Specialist|Senior Technical Recruiter|Technical Recruiter|Recruiter",
    r"HR Manager|Senior HR Manager|Assistant Manager HR|HR Generalist|HR Operations Manager",
    r"People Operations Manager|Compensation and Benefits Manager|Learning and Development Manager|L&D Manager",

    # Supply Chain, Operations & Marketing
    r"VP Operations|Director of Operations|Operations Director|Head of Operations|Operations Manager",
    r"Supply Chain Director|Supply Chain Manager|Head of Supply Chain|Logistics Manager",
    r"Procurement Manager|Head of Procurement|Purchasing Manager|Vendor Manager",
    r"VP Marketing|Chief Marketing Officer|CMO|Head of Marketing|Marketing Director",
    r"Digital Marketing Manager|Brand Manager|Growth Marketing Manager|Performance Marketing Manager"
]
INDUSTRY_ROLE_CATALOG_RE = re.compile(r"\b(" + "|".join(INDUSTRY_ROLE_CATALOG_PATTERNS) + r")\b", re.I)


def extract_jd_title(text: str) -> str:
    """
    Production Multi-Strategy Target Role Extraction:
    1. Explicit Metadata Labels (Job Title, Position, Role, Target Role, Profile, Designation, Vacancy, Opening)
    2. Natural Language Intent Sentences ('Looking for', 'Hiring', 'Seeking', 'Recruiting for', 'Urgent opening for')
    3. Contextual Role Introductions ('As a(n) [Role], you will...')
    4. Standard Industry Role Catalog (150+ Market Titles via compiled regex)
    5. Document Layout / Header Lines (Top 15 lines with role nouns)
    """
    if not text:
        return "Target Role"

    # Strategy 1: Explicit metadata labels
    m_lbl = re.search(
        r"(?:job\s*(?:title|profile|role|name)|position(?:\s*title)?|role(?:\s*(?:title|purpose|name))?|"
        r"target(?:\s*role)?|designation|profile(?:\s*name)?|vacancy(?:\s*for)?|opening(?:\s*for)?|"
        r"requisition(?:\s*title)?|post)\s*[:\-–—]\s*([^\n\r]+)", text, re.I
    )
    if m_lbl:
        cand = clean_role_candidate(m_lbl.group(1))
        # If header captured a sentence like 'As a Senior Product Manager, you will...', extract role
        m_as_inside = re.search(r"as\s+(?:a|an)?\s*([^\n\r,\.;]{3,50}?)(?:,\s*you\b|$)", cand, re.I)
        if m_as_inside:
            cand = clean_role_candidate(m_as_inside.group(1))
        if 3 < len(cand) < 65:
            return cand

    # Strategy 2: Intent / Action Sentence Matcher
    intent_verbs = r"(?:we\s+are\s+|company\s+is\s+|is\s+)?(?:looking\s+(?:to\s+hire|for)|hiring|seeking|recruiting\s+(?:for)?|urgent\s+opening\s+for|inviting\s+applications\s+for|need|searching\s+for)"
    m_intent = re.search(
        rf"{intent_verbs}\s+([^\n\r,\.;]{{3,80}}?)(?:\s+(?:to\s+(?:join|lead|drive|manage|build|oversee|be)|with\b|in\b|who\b|at\b|\([^\)]*\)|$))",
        text, re.I
    )
    if m_intent:
        cand = clean_role_candidate(m_intent.group(1))
        if 3 < len(cand) < 65 and not any(k in cand.lower() for k in ["candidate", "individual", "professional", "person", "someone", "team"]):
            return cand

    # Strategy 3: Contextual Role Opening (e.g. 'As a(n) [Role], you will...')
    m_as = re.search(r"\bAs\s+(?:a|an|our)?\s*([^\n\r,\.;:\(]{3,50}?)(?:\s*\([^\)]*\))?,\s*you\s+will\b", text, re.I)
    if m_as:
        cand = clean_role_candidate(m_as.group(1))
        if 3 < len(cand) < 50:
            return cand

    # Strategy 4: Standard Industry Role Catalog (150+ Market Titles)
    m_phrase = INDUSTRY_ROLE_CATALOG_RE.search(text)
    if m_phrase:
        return m_phrase.group(0).strip()

    # Strategy 5: Scanning top lines / header banner
    for line in text.splitlines()[:15]:
        l = line.strip()
        if 4 < len(l) < 75 and not any(k in l.lower() for k in ["about us", "http", "www", "confidential", "page ", "overview", "stonex", "awards", "location"]):
            if any(k in l.lower() for k in ["manager", "lead", "head", "engineer", "secretary", "counsel", "specialist", "executive", "director", "architect", "developer", "consultant", "analyst", "president", "officer"]):
                cand = re.sub(r"^(?:WE ARE\s+HIRING|WE\'RE HIRING|HIRING NOW|NOW HIRING|URGENT OPENING|INVITING APPLICATIONS FOR THE POST OF)\s*[:\-–]?\s*", "", l, flags=re.I).strip()
                cand = clean_role_candidate(cand)
                if 3 < len(cand) < 65:
                    return cand

    return "Target Role"


def extract_jd_qualifications(jd_text: str) -> Set[str]:
    """
    Extracts required degree / professional credential archetypes from JD text.
    """
    text_u = jd_text.upper()
    reqs = set()
    if re.search(r"\b(?:LLB|LLM|LAW|LEGAL DEGREE|BACHELOR OF LAWS?)\b", text_u):
        reqs.add("LAW")
    if re.search(r"\b(?:COMPANY SECRETARY|QUALIFIED CS|CS|ICSI|SECRETARIAL PRACTICE)\b", text_u):
        reqs.add("CS")
    if re.search(r"\b(?:CA|CHARTERED ACCOUNTANT|ICAI)\b", text_u):
        reqs.add("CA")
    if re.search(r"\b(?:B\.TECH|BTECH|B\.E\.|BE|M\.TECH|MTECH|MCA|BCA|COMPUTER SCIENCE)\b", text_u):
        reqs.add("TECH")
    if re.search(r"\b(?:MBA|PGDM)\b", text_u):
        reqs.add("MGMT")
    if re.search(r"\b(?:MBBS|MD|MS)\b", text_u):
        reqs.add("MED")
    return reqs


def extract_rerank_query(jd_text: str, max_chars: int = 1500) -> str:
    """
    Extracts a high-signal search query from the JD for the Cross-Encoder.
    Prioritizes Role, Technical Skills, Key Responsibilities, and Qualifications,
    bypassing repetitive corporate 'About Us' boilerplate if present.
    """
    if not jd_text:
        return ""
    
    # Pre-clean boilerplate if present
    cleaned = clean_jd_text(jd_text)
    jd_title = extract_jd_title(jd_text)
    jd_reqs = extract_jd_qualifications(jd_text)
    reqs_str = f" Required Qualifications: {', '.join(sorted(jd_reqs))}." if jd_reqs else ""

    header_line = f"Target Role: {jd_title}.{reqs_str}"
    if len(cleaned) <= max_chars:
        return f"{header_line}\n\n{cleaned}"

    # Priority 1: Key Responsibilities / Required Technical Skills / Job Description
    match = re.search(
        r"\b(?:Job\s+Description|Designation|Desired\s+Skills?|Desired\s+Skill\s+Set|Key\s+Responsibilities|Responsibilities|What\s+We\s+Are\s+Looking\s+For|Required\s+Technical\s+Skills|Must\s+Have\s+Skills|Technical\s+Skills|Key\s+Requirements|Required\s+Experience)\b",
        cleaned,
        re.IGNORECASE
    )
    if not match:
        match = re.search(
            r"\b(?:Job\s+Summary|Role\s+Summary|Role\s+overview|About\s+the\s+role)\b",
            cleaned,
            re.IGNORECASE
        )

    if match:
        start_pos = match.start()
        tech_chunk = cleaned[start_pos:start_pos + (max_chars - len(header_line) - 10)].strip()
        return f"{header_line}\n\n{tech_chunk}"

    return f"{header_line}\n\n{cleaned[:max_chars]}"


def resolve_experience_filter(
    experience_text: Optional[str] = None,
    additional_info: Optional[str] = None,
    jd_text: Optional[str] = None,
    legacy_min_exp: Optional[float] = None
) -> Dict[str, Any]:
    """
    Resolves the active experience filter with strict precedence:
      1. Explicit recruiter experience text input (e.g. '0-1', '2+', '5-10yr')
      2. Experience criteria in additional_info (e.g. 'experience: 4-8 years', '4-8 yrs')
      3. Auto-detection from JD text/file
      4. Legacy min_experience parameter
      5. None (open to all candidates)
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

    # 2. Check additional_info for an experience override
    if additional_info and additional_info.strip():
        mn, mx, lbl = extract_experience_from_text(additional_info.strip())
        if mn is None and mx is None:
            mn, mx, lbl = extract_experience_from_jd(additional_info.strip())
        if mn is not None or mx is not None:
            return {
                "source": "additional_info_override",
                "raw_input": additional_info.strip(),
                "min_experience": mn,
                "max_experience": mx,
                "label": f"{lbl} (Recruiter Override)"
            }

    # 3. Check JD auto-detect
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

    # 4. Check legacy min_experience
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


ROLE_FAMILY_MAP = {
    "analyst": ["analyst", "analysts", "analytics", "analysis", "bi developer", "power bi", "business intelligence"],
    "analytics": ["analyst", "analysts", "analytics", "analysis", "data science", "data scientist"],
    "auditor": ["auditor", "auditors", "audit", "auditing"],
    "audit": ["auditor", "auditors", "audit", "auditing"],
    "secretary": ["secretary", "secretaries", "secretarial", "company secretary", "company secretaries", "cs"],
    "counsel": ["counsel", "legal", "advocate", "lawyer", "law"],
    "developer": ["developer", "developers", "engineer", "engineers", "programmer", "programmers"],
    "engineer": ["engineer", "engineers", "developer", "developers", "architect", "architects"],
    "sales": ["sales", "business development", "bd", "selling", "account director", "account manager", "client acquisition", "client relationship", "commercial", "revenue", "regional sales", "zonal sales", "sales manager", "sales leader", "channel sales"],
    "development": ["business development", "bd", "client acquisition", "partnerships", "sales development", "strategic partnerships"],
    "bd": ["business development", "bd", "sales", "client acquisition", "partnerships"],
    "product": ["product manager", "product management", "technical product manager", "program manager", "scrum master"],
    "hr": ["hr", "human resources", "talent acquisition", "recruiter", "hrbp", "people operations", "talent"],
    "finance": ["finance", "financial", "accounting", "accountant", "chartered accountant", "ca", "taxation"],
    "marketing": ["marketing", "brand", "growth", "digital marketing", "seo", "content"],
    "operations": ["operations", "supply chain", "procurement", "logistics"]
}


def eval_role_alignment(jd_title: str, cand_desigs: List[str], cand_summary: str = "") -> float:
    """
    Evaluates role & designation alignment (out of 25.0 points).
    Prioritizes actual candidate designations over freeform summary text.
    Uses multi-word phrases and role family stems to avoid false substring matches.
    """
    if not jd_title or jd_title == "Target Role":
        return 18.0

    desig_str = " ".join(cand_desigs).lower()
    sum_str = (cand_summary or "").lower()
    full_str = f"{desig_str} {sum_str}"
    jd_lower = jd_title.lower()

    # Pre-clean false compound phrases
    cleaned_desig = desig_str
    for false_compound in ["data structure", "data structures", "data centre", "data center", "audit trail", "audit log"]:
        cleaned_desig = cleaned_desig.replace(false_compound, "")

    # Check for cross-domain clashes
    is_clashing = False
    if any(k in jd_lower for k in ["legal", "secretary", "counsel"]):
        if any(w in full_str for w in ["sales specialist", "sales manager", "it auditor", "boomi developer", "oracle financial", "architect"]):
            is_clashing = True
    elif "audit" in jd_lower:
        if any(w in full_str for w in ["sales specialist", "boomi developer", "software developer", "legal counsel"]):
            is_clashing = True
    elif any(k in jd_lower for k in ["engineer", "developer", "data scientist"]):
        if any(w in full_str for w in ["sales specialist", "company secretary", "legal head"]):
            is_clashing = True
    elif any(k in jd_lower for k in ["pilot", "captain", "aviation", "flight"]):
        if any(w in full_str for w in ["developer", "programmer", "secretary", "accountant", "auditor", "sales"]):
            is_clashing = True
    elif "analyst" in jd_lower:
        if any(w in full_str for w in ["audit manager", "article assistant", "company secretary", "legal head", "legal manager", "radar system"]):
            is_clashing = True

    if is_clashing:
        return 3.5

    # High-precision evaluation for Business Development & Sales roles
    if any(k in jd_lower for k in ["business development", "sales", "vp bd", "commercial director", "revenue officer", "client acquisition"]):
        sales_kw = [
            "sales", "business development", "bd", "account manager", "key account",
            "client relationship", "client acquisition", "commercial", "revenue",
            "channel management", "channel sales", "zonal sales", "regional sales",
            "store operations", "retail operations", "expansion"
        ]
        cand_has_sales_desig = any(
            bool(re.search(r"\b" + re.escape(w) + r"\b", cleaned_desig))
            for w in sales_kw
        )
        cand_has_sales_sum = any(
            bool(re.search(r"\b" + re.escape(w) + r"\b", sum_str))
            for w in ["sales", "business development", "b2b sales", "revenue growth", "client acquisition", "luxury sales", "channel sales", "sales leader", "retail professional"]
        )
        clash_terms = [
            "software", "developer", "architect", "engineer", "devops", "cloud", "data analyst", "bi developer",
            "hr business partner", "human resources", "talent acquisition", "recruiter", "hr manager", "hr head", "agm hr",
            "legal counsel", "company secretary", "advocate", "lawyer",
            "internal audit", "audit manager", "auditor", "article assistant"
        ]
        if not cand_has_sales_desig and not cand_has_sales_sum:
            if any(w in cleaned_desig for w in clash_terms):
                return 3.5
            return 4.0
        if cand_has_sales_desig:
            has_lead = any(k in desig_str for k in ["vice president", "vp", "director", "head", "general manager", "lead", "manager", "regional", "zonal"])
            return 24.0 if has_lead else 21.0
        if cand_has_sales_sum:
            return 12.0

    # Core title phrase check (e.g. "data analyst" directly in designations)
    core_title = re.sub(r"\b(senior|sr|lead|head|junior|jr|associate|assistant|manager|director|general|chief|officer|executive)\b", "", jd_lower).strip()
    core_title = re.sub(r"\s+", " ", core_title)
    if core_title and len(core_title) >= 4 and core_title in cleaned_desig:
        base = 24.0 if any(k in desig_str for k in ["head", "manager", "senior", "sr.", "sr ", "lead", "director"]) else 21.0
        return round(base, 1)

    # Exclude non-role words and generic descriptors
    non_role_words = {
        "general", "manager", "senior", "lead", "head", "associate", "officer", 
        "executive", "role", "title", "and", "the", "for", "with", "target",
        "commercial", "international", "national", "global", "corporate", "business",
        "specialist", "assistant", "junior", "level", "group", "team", "services",
        "system", "systems", "solutions", "operations", "practice", "center"
    }
    jd_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", jd_lower) if w not in non_role_words]
    if not jd_words:
        return 16.0

    # Expand JD words with role family synonyms
    expanded_target_terms = set(jd_words)
    for w in jd_words:
        if w in ROLE_FAMILY_MAP:
            expanded_target_terms.update(ROLE_FAMILY_MAP[w])

    desig_family_match = any(bool(re.search(r"\b" + re.escape(term) + r"\b", cleaned_desig)) for term in expanded_target_terms)
    sum_family_match = any(bool(re.search(r"\b" + re.escape(term) + r"\b", sum_str)) for term in expanded_target_terms)

    # Check match ratio on raw words
    desig_matches = [w for w in jd_words if re.search(r"\b" + re.escape(w) + r"\b", cleaned_desig)]
    desig_ratio = len(desig_matches) / len(jd_words)

    # Discipline noun enforcement (e.g., if JD specifies 'analyst', an engineer or developer cannot get top role score)
    discipline_nouns = {"analyst", "analytics", "auditor", "audit", "secretary", "counsel", "lawyer", "developer", "engineer", "scientist", "architect"}
    target_disciplines = [w for w in jd_words if w in discipline_nouns]
    if target_disciplines:
        holds_in_desig = any(
            any(bool(re.search(r"\b" + re.escape(term) + r"\b", cleaned_desig)) for term in ROLE_FAMILY_MAP.get(disc, [disc]))
            for disc in target_disciplines
        )
        holds_in_sum = any(
            any(bool(re.search(r"\b" + re.escape(term) + r"\b", sum_str)) for term in ROLE_FAMILY_MAP.get(disc, [disc]))
            for disc in target_disciplines
        )
        if not holds_in_desig:
            if holds_in_sum:
                return 18.0  # Certified or held role mentioned in summary/headline
            elif desig_ratio > 0 or desig_family_match or sum_family_match:
                return 10.0
            return 4.0

    if desig_ratio >= 0.5 or (len(jd_words) == 2 and desig_family_match and any(w in cleaned_desig for w in ["data", "senior", "lead", "bi", "business"])):
        base = 24.0 if any(k in desig_str for k in ["head", "manager", "senior", "sr.", "lead", "director"]) else 21.0
        return round(base, 1)
    elif desig_family_match:
        return 20.0
    elif desig_ratio > 0:
        return 16.0
    elif sum_family_match:
        return 12.0
    else:
        return 4.0


def eval_work_alignment(norm_ce: float) -> float:
    """
    Evaluates cross-encoder work experience & responsibilities alignment (out of 25.0 points).
    """
    norm_val = max(0.0, min(1.0, norm_ce))
    return round(4.0 + (norm_val * 21.0), 1)


def eval_skills_alignment(matched_domain_skills_count: int) -> float:
    """
    Evaluates technical & domain skills matched (out of 20.0 points).
    """
    if matched_domain_skills_count >= 4:
        return 20.0
    elif matched_domain_skills_count == 3:
        return 17.0
    elif matched_domain_skills_count == 2:
        return 14.0
    elif matched_domain_skills_count == 1:
        return 10.0
    else:
        return 3.0


def eval_education_alignment(jd_reqs: Set[str], cand_degrees: List[str], cand_edu: List[Any]) -> float:
    """
    Evaluates education & professional credentials (out of 15.0 points).
    """
    if not jd_reqs:
        return 13.5

    cand_str = " ".join([str(d).upper() for d in cand_degrees] + [str(e).upper() for e in cand_edu])
    cand_has = set()
    if re.search(r"\b(?:LLB|LLM|LAW|ADVOCATE|BAR COUNCIL)\b", cand_str): cand_has.add("LAW")
    if re.search(r"\b(?:CS|COMPANY SECRETARY|ICSI|ACS|FCS)\b", cand_str): cand_has.add("CS")
    if re.search(r"\b(?:CA|CHARTERED ACCOUNTANT|ICAI|ACA|FCA)\b", cand_str): cand_has.add("CA")
    if re.search(r"\b(?:B\.TECH|BTECH|B\.E\.|BE|M\.TECH|MTECH|MCA|BCA|COMPUTER SCIENCE)\b", cand_str): cand_has.add("TECH")
    if re.search(r"\b(?:MBA|PGDM)\b", cand_str): cand_has.add("MGMT")
    if re.search(r"\b(?:MBBS|MD|MS)\b", cand_str): cand_has.add("MED")

    matched = cand_has.intersection(jd_reqs)
    if len(matched) == len(jd_reqs):
        return 15.0
    elif len(matched) > 0:
        return 12.0
    elif cand_has:
        return 4.0
    else:
        return 2.0


def eval_experience_fit(cand_exp: float, min_exp: Optional[float], max_exp: Optional[float]) -> float:
    """
    Evaluates experience range fit additively (out of 15.0 points).
    """
    if min_exp is None and max_exp is None:
        return 15.0

    cand_exp = float(cand_exp or 0.0)
    lower_bound = float(min_exp) if min_exp is not None else 0.0
    upper_bound = float(max_exp) if max_exp is not None else 999.0

    if lower_bound <= cand_exp <= upper_bound:
        return 15.0
    elif cand_exp > upper_bound:
        gap = cand_exp - upper_bound
        return round(max(8.0, 15.0 - (0.40 * gap)), 1)
    else:
        gap = lower_bound - cand_exp
        return round(max(3.0, 14.0 - (1.0 * gap)), 1)


def compute_composite_match_score(
    payload: dict,
    norm_ce: float,
    jd_title: str,
    jd_reqs: Set[str],
    query_text: str,
    min_exp: Optional[float],
    max_exp: Optional[float],
    demo_criteria: Optional[Dict[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Production Tiered Multi-Factor Scoring Engine:
    1. Evaluates Domain Qualifications (Role 25% + Work 25% + Skills 20% + Education 15% = 85%).
    2. Implements Two-Tier Production Experience Separation:
       - Tier 1: Matching Background + In-Range Experience -> Ranks at Top (82% - 98%).
       - Tier 2A: Matching Background + Senior Over-Target Experience -> Ranks right below Tier 1 (65% - 78%).
       - Tier 2B: Matching Background + Junior Under-Target Experience -> Ranks below Senior (42% - 64%).
       - Irrelevant Domain: Disqualified from Top (score < 35%).
    """
    cand_desigs = payload.get("designation", []) or []
    cand_summary = payload.get("summary") or ""
    cand_degrees = payload.get("degree", []) or []
    cand_edu = payload.get("EDUCATION", []) or []
    cand_exp = float(payload.get("total_experience", 0.0))
    all_skills = payload.get("skills", []) or []
    dom_skills = [s for s in all_skills if s.strip().lower() not in GENERIC_SOFT_SKILLS]
    matched_domain_skills = [
        s for s in dom_skills 
        if re.search(r"\b" + re.escape(s.strip().lower()) + r"\b", query_text.lower())
    ]

    role_pts = eval_role_alignment(jd_title, cand_desigs, cand_summary)
    work_pts = eval_work_alignment(norm_ce)
    skills_pts = eval_skills_alignment(len(matched_domain_skills))
    edu_pts = eval_education_alignment(jd_reqs, cand_degrees, cand_edu)

    domain_raw = role_pts + work_pts + skills_pts + edu_pts  # out of 85
    domain_pct = (domain_raw / 85.0) * 100.0

    lower_bound = float(min_exp) if min_exp is not None else 0.0
    upper_bound = float(max_exp) if max_exp is not None else 999.0

    # Demographic criteria evaluation
    has_demo_criteria = bool(
        demo_criteria and (demo_criteria.get("gender") or demo_criteria.get("min_age") is not None or demo_criteria.get("max_age") is not None)
    )
    is_pure_demographic_query = bool(
        has_demo_criteria
        and (not jd_title or jd_title in ["Target Role", "Job Description"])
        and not jd_reqs
        and len(matched_domain_skills) == 0
    )

    # Determine domain relevance
    has_degree_match = bool(jd_reqs and edu_pts >= 12.0)
    has_skills_match = bool(len(matched_domain_skills) >= 2)
    has_role_match = bool(role_pts >= 20.0)

    is_domain_match = has_role_match or (role_pts >= 15.0 and (has_skills_match or has_degree_match))

    demo_status = "N/A"
    is_demo_verified = False

    if is_pure_demographic_query:
        # PURE DEMOGRAPHIC QUERY: Recruiter specified only age and/or gender (no specific role or skill constraint)
        cand_gender = (payload.get("gender") or "").strip().lower()
        cand_age = payload.get("age")

        req_gender = (demo_criteria.get("gender") or "").strip().lower()
        req_min_age = demo_criteria.get("min_age")
        req_max_age = demo_criteria.get("max_age")

        is_gender_verified = (cand_gender == req_gender) if req_gender else True
        is_age_verified = True
        if req_min_age is not None or req_max_age is not None:
            if cand_age is not None:
                try:
                    c_age = float(cand_age)
                    if req_min_age is not None and c_age < float(req_min_age):
                        is_age_verified = False
                    if req_max_age is not None and c_age > float(req_max_age):
                        is_age_verified = False
                except (ValueError, TypeError):
                    is_age_verified = False
            else:
                is_age_verified = False

        if is_gender_verified and is_age_verified:
            demo_status = "Verified Match"
            # TIER 1: Resumes with verified demographic match rank at TOP (88% - 97%)
            final_score = round(min(97.0, max(88.0, 88.0 + (norm_ce * 9.0))), 1)
            exp_pts = 15.0
            tier_label = "Tier 1: Verified Demographics"
        else:
            demo_status = "Unspecified on Resume"
            # TIER 2: Resumes without demographic data on CV rank right BELOW Tier 1 (70% - 78%)
            final_score = round(min(78.0, max(70.0, 70.0 + (norm_ce * 8.0))), 1)
            exp_pts = 10.0
            tier_label = "Tier 2: Demographics Unspecified on CV"
    elif not is_domain_match:
        # Irrelevant Domain (Sales, IT, Audit for Legal role, or vice-versa)
        final_score = round(min(35.0, domain_pct * 0.38), 1)
        exp_pts = 0.0
        tier_label = "Irrelevant Domain"
    elif min_exp is None and max_exp is None:
        # Open experience range
        final_score = round(max(35.0, min(98.0, 15.0 + (domain_pct * 0.83))), 1)
        exp_pts = 15.0
        tier_label = "Open Range"
    elif lower_bound <= cand_exp <= upper_bound:
        is_exact_fit = (role_pts >= 20.0) and (work_pts >= 8.0 or has_skills_match)
        if is_exact_fit:
            # TIER 1: Exact Background + Exact In-Range Experience -> GUARANTEED TOP LIST (82% - 98%)
            s = 82.0 + ((domain_pct - 50.0) / 50.0) * 16.0
            final_score = round(min(98.0, max(82.0, s)), 1)
            exp_pts = 15.0
            tier_label = "Tier 1: Exact Match (In-Range)"
        else:
            # TIER 2C: Adjacent Background (In-Range Exp, but different role discipline or low work match) -> (48% - 64%)
            s = 48.0 + ((domain_pct - 35.0) / 50.0) * 16.0
            final_score = round(min(64.0, max(45.0, s)), 1)
            exp_pts = 10.0
            tier_label = "Tier 2C: Adjacent Role / Partial Fit"
    elif cand_exp > upper_bound:
        # TIER 2A: Exact Background + Senior Over-Range -> Guaranteed BELOW Tier 1 (65% - 78%)
        gap = cand_exp - upper_bound
        exp_factor = max(0.85, 1.0 - (0.012 * gap))
        s = (66.0 + ((domain_pct - 50.0) / 50.0) * 12.0) * exp_factor
        final_score = round(min(78.0, max(64.0, s)), 1)
        exp_pts = round(max(6.0, 15.0 - (0.8 * gap)), 1)
        tier_label = f"Tier 2A: Senior Over-Range (+{round(gap, 1)}y)"
    else:
        # TIER 2B: Exact Background + Junior Under-Range -> Guaranteed BELOW Senior (42% - 64%)
        gap = lower_bound - cand_exp
        exp_factor = max(0.70, 1.0 - (0.03 * gap))
        s = (48.0 + ((domain_pct - 50.0) / 50.0) * 16.0) * exp_factor
        final_score = round(min(64.0, max(42.0, s)), 1)
        exp_pts = round(max(2.0, 14.0 - (1.2 * gap)), 1)
        tier_label = f"Tier 2B: Junior Under-Range (-{round(gap, 1)}y)"

    # Demographic adjustment for role/domain queries
    if has_demo_criteria and not is_pure_demographic_query:
        cand_gender = (payload.get("gender") or "").strip().lower()
        cand_age = payload.get("age")

        req_gender = (demo_criteria.get("gender") or "").strip().lower()
        req_min_age = demo_criteria.get("min_age")
        req_max_age = demo_criteria.get("max_age")

        is_gender_verified = (cand_gender == req_gender) if req_gender else True
        is_age_verified = True
        if req_min_age is not None or req_max_age is not None:
            if cand_age is not None:
                try:
                    c_age = float(cand_age)
                    if req_min_age is not None and c_age < float(req_min_age):
                        is_age_verified = False
                    if req_max_age is not None and c_age > float(req_max_age):
                        is_age_verified = False
                except (ValueError, TypeError):
                    is_age_verified = False
            else:
                is_age_verified = False

        if is_gender_verified and is_age_verified:
            demo_status = "Verified Match"
        else:
            demo_status = "Unspecified on Resume"
            # Soft demographic ranking: resumes that match qualifications but omit demographics
            # are ranked directly below verified profiles rather than being discarded
            if is_domain_match:
                final_score = round(max(35.0, final_score - 15.0), 1)
            tier_label = f"{tier_label} (Demographics Unspecified)"

    breakdown = {
        "role": role_pts,
        "work": work_pts,
        "skills": skills_pts,
        "education": edu_pts,
        "experience": exp_pts,
        "tier": tier_label,
        "demographics": demo_status
    }
    return final_score, breakdown


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
        "embedding_dim": 512,
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
    additional_info: Optional[str] = Form(None),
    target_role: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
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
    eff_additional_info = ""
    eff_target_role = ""
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
        eff_additional_info = str(body.get("additional_info") or "").strip()
        eff_target_role = str(body.get("target_role") or body.get("role") or "").strip()
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
        eff_additional_info = str(additional_info or "").strip()
        eff_target_role = str(target_role or role or "").strip()
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

    # 1. Resolve experience requirement (user text input vs additional_info vs JD auto-detection)
    exp_filter = resolve_experience_filter(
        experience_text=raw_exp_text,
        additional_info=eff_additional_info,
        jd_text=query_text,
        legacy_min_exp=legacy_min_exp
    )

    # 2. Extract keywords from JD to highlight matching skills
    jd_lower = query_text.lower()

    # 3. Embed JD via FastEmbed ONNX (pre-processed to isolate job specification from marketing boilerplate)
    t_embed_start = time.perf_counter()
    clean_job_query = clean_jd_text(query_text)
    query_vector = list(embed_model.embed([clean_job_query]))[0].tolist()
    t_embed = time.perf_counter() - t_embed_start

    # 4. Build Qdrant Filters
    filter_conditions = []

    # Soft Experience Scoring: do NOT hard-discard candidates at Stage 1.
    # In-range profiles get prioritized, while out-of-range candidates with matching backgrounds
    # are gracefully penalized in Stage 2 rather than rejected.
    min_exp = exp_filter.get("min_experience")
    max_exp = exp_filter.get("max_experience")
    exp_filter["mode"] = "soft_experience_scoring"
    exp_filter["applied_search_range"] = {
        "min": min_exp,
        "max": max_exp,
        "strategy": "Soft Experience Scoring (in-range prioritized, near-range scored right below)"
    }

    # Mandatory skills filter
    req_skills_list = [s.strip() for s in eff_mandatory_skills.split(",") if s.strip()]
    for skill in req_skills_list:
        filter_conditions.append(
            models.FieldCondition(
                key="skills",
                match=models.MatchValue(value=skill)
            )
        )

    # Demographic filters (Gender, Age) parsed from recruiter notes OR JD text
    demo_criteria = parse_demographic_criteria(eff_additional_info)
    if not demo_criteria:
        demo_criteria = parse_demographic_criteria(query_text)

    must_conditions = list(filter_conditions)
    must_not_conditions = []

    if demo_criteria.get("gender"):
        req_g = demo_criteria["gender"].lower()
        opp_gender = "male" if req_g == "female" else ("female" if req_g == "male" else None)
        if opp_gender:
            must_not_conditions.append(
                models.FieldCondition(
                    key="gender",
                    match=models.MatchValue(value=opp_gender)
                )
            )

    if demo_criteria.get("min_age") is not None:
        must_not_conditions.append(
            models.FieldCondition(
                key="age",
                range=models.Range(lt=float(demo_criteria["min_age"]))
            )
        )

    if demo_criteria.get("max_age") is not None:
        must_not_conditions.append(
            models.FieldCondition(
                key="age",
                range=models.Range(gt=float(demo_criteria["max_age"]))
            )
        )

    query_filter = None
    if must_conditions or must_not_conditions:
        query_filter = models.Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )

    # 5. Stage 1: Fast Candidate Retrieval via Qdrant Bi-Encoder
    t_search_start = time.perf_counter()
    # Broaden Stage 1 pool so FlashRank has access to in-range, near-range, and demographic-tiered talent
    base_pool = 100 if demo_criteria else 60
    initial_fetch_limit = max(eff_limit * 6, base_pool) if (eff_enable_rerank and ranker) else max(eff_limit * 3, 30)
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
            desigs = ", ".join(p.get("designation", [])[:5])
            all_skills = p.get("skills", []) or []
            # Exclude generic soft skills from cross-encoder passage to evaluate domain competencies only
            domain_skills = [s for s in all_skills if s.strip().lower() not in GENERIC_SOFT_SKILLS]
            skills_str = ", ".join(domain_skills[:25])
            exp = p.get("total_experience", 0.0)
            summary = p.get("summary") or ""

            # Structured work experience responsibilities for cross-encoder
            we_list = p.get("work_experience") or []
            exp_resp = []
            if we_list:
                for idx, we in enumerate(we_list[:5]):
                    r = we.get("role") or ""
                    c = we.get("company") or ""
                    resp = we.get("responsibilities") or []
                    if idx == 0:
                        sel = resp[:3]
                    elif idx == 1:
                        sel = resp[:2]
                    else:
                        sel = resp[:1]
                    b_text = " ".join([b.strip() for b in sel])
                    if b_text:
                        exp_resp.append(f"{r} @ {c}: {b_text}")
                    elif r or c:
                        exp_resp.append(f"{r} @ {c}")
            elif p.get("company_details"):
                comp_lines = [c for c in p.get("company_details", []) if not c.startswith("EXPERIENCE") and not c.startswith("Total experience")]
                exp_resp = comp_lines[:5]

            exp_block = " | ".join(exp_resp)
            proj_list = p.get("projects") or []
            proj_str = f" Projects: {'; '.join([str(x).strip() for x in proj_list[:3]])}" if proj_list else ""
            deg_str = ", ".join(p.get("degree", [])[:4])
            text_block = f"Role: {desigs}. Degrees: {deg_str}. Experience: {exp} years. Work: {exp_block}. Domain Skills: {skills_str}. Summary: {summary}.{proj_str}"
            passages.append({"id": pt.id, "text": text_block, "pt": pt})

        t_rerank_start = time.perf_counter()
        jd_title = eff_target_role if eff_target_role else extract_jd_title(query_text)
        jd_reqs = extract_jd_qualifications(query_text)
        reqs_str = f" Required Qualifications: {', '.join(sorted(jd_reqs))}." if jd_reqs else ""
        rerank_query = f"Target Role: {jd_title}.{reqs_str} {extract_rerank_query(query_text, max_chars=1200)}"
        if eff_additional_info:
            rerank_query = f"{rerank_query}\n\nAdditional Recruiter Requirements & Notes:\n{eff_additional_info}"
        req = RerankRequest(query=rerank_query, passages=passages)
        reranked_results = ranker.rerank(req)
        t_rerank = time.perf_counter() - t_rerank_start

        # Stage 2: Multi-Factor Composite Scoring
        ce_scores = [float(r["score"]) for r in reranked_results]
        log_ces = [math.log(1.0 + max(0.0, s) * 500.0) for s in ce_scores]
        min_lce, max_lce = min(log_ces), max(log_ces)
        spread_lce = max_lce - min_lce if max_lce > min_lce else 1.0

        # Deduplicate candidates by identity (email/phone/name) so duplicate uploads never repeat
        seen_identities = set()
        deduped_results = []
        for r, lce in zip(reranked_results, log_ces):
            p = r["pt"].payload or {}
            c_name = (p.get("name") or "").strip().lower()
            c_email = (p.get("email") or "").strip().lower()
            c_phone = (p.get("mobile_number") or "").strip()
            ident = c_email if (c_email and c_email not in ["none", "null"]) else (c_phone if len(c_phone) >= 8 else c_name)
            if ident and ident in seen_identities:
                continue
            seen_identities.add(ident)
            r["log_ce"] = lce
            deduped_results.append(r)

        # Compute Production Multi-Factor Composite Match Percentage
        # Evaluates Role (25%), Work Responsibilities (25%), Domain Skills (20%), Education (15%), and Experience Range (15%)
        evaluated_items = []
        for r in deduped_results:
            pt = r["pt"]
            p = pt.payload or {}
            lce = r["log_ce"]
            norm_ce = (lce - min_lce) / spread_lce
            cs = float(r["score"])

            composite_score, score_bd = compute_composite_match_score(
                payload=p,
                norm_ce=norm_ce,
                jd_title=jd_title,
                jd_reqs=jd_reqs,
                query_text=query_text,
                min_exp=min_exp,
                max_exp=max_exp,
                demo_criteria=demo_criteria
            )
            evaluated_items.append((pt, cs, composite_score, score_bd))

        evaluated_items.sort(key=lambda x: x[2], reverse=True)
        candidate_items = evaluated_items[:eff_limit]
    else:
        jd_title = eff_target_role if eff_target_role else extract_jd_title(query_text)
        jd_reqs = extract_jd_qualifications(query_text)

        seen_identities = set()
        deduped_pts = []
        for pt in search_response.points:
            p = pt.payload or {}
            c_name = (p.get("name") or "").strip().lower()
            c_email = (p.get("email") or "").strip().lower()
            c_phone = (p.get("mobile_number") or "").strip()
            ident = c_email if (c_email and c_email not in ["none", "null"]) else (c_phone if len(c_phone) >= 8 else c_name)
            if ident and ident in seen_identities:
                continue
            seen_identities.add(ident)
            deduped_pts.append(pt)

        scored_pts = []
        for pt in deduped_pts:
            p = pt.payload or {}
            cos = float(pt.score)
            norm_cos = max(0.0, min(1.0, (cos - 0.70) / 0.25))
            composite_score, score_bd = compute_composite_match_score(
                payload=p,
                norm_ce=norm_cos,
                jd_title=jd_title,
                jd_reqs=jd_reqs,
                query_text=query_text,
                min_exp=min_exp,
                max_exp=max_exp,
                demo_criteria=demo_criteria
            )
            scored_pts.append((pt, None, composite_score, score_bd))
        scored_pts.sort(key=lambda x: x[2], reverse=True)

        candidate_items = scored_pts[:eff_limit]

    # 7. Format Candidate Cards with Multi-Factor Metrics
    candidates = []
    for rank, (pt, cross_score, match_percentage, score_bd) in enumerate(candidate_items, 1):
        payload = pt.payload or {}
        candidate_skills = payload.get("skills", []) or []

        matched_skills = [
            s for s in candidate_skills
            if re.search(r"\b" + re.escape(s.strip().lower()) + r"\b", jd_lower)
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
            "score_breakdown": score_bd,
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
            "work_experience": payload.get("work_experience", []) or [],
            "education": payload.get("EDUCATION", []) or [],
            "degrees": payload.get("degree", []) or [],
            "matched_skills": matched_skills,
            "other_skills": other_skills[:15],
            "total_skills_count": len(candidate_skills),
            "summary": payload.get("summary", ""),
            "gender": payload.get("gender"),
            "age": payload.get("age"),
            "dob": payload.get("dob"),
            "projects": payload.get("projects", []) or [],
            "certification": payload.get("certification", []) or [],
        })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 1)

    try:
        col_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_db_points = col_info.points_count
    except Exception:
        total_db_points = 0

    return {
        "success": True,
        "total_database_points": total_db_points,
        "total_matches": len(candidates),
        "jd_title": jd_title,
        "role_source": "manual_override" if eff_target_role else ("auto_detected" if (jd_title and jd_title != "Target Role") else "none"),
        "jd_qualifications": sorted(list(jd_reqs)) if jd_reqs else [],
        "experience_filter": exp_filter,
        "demographics_filter": demo_criteria if demo_criteria else None,
        "additional_info": eff_additional_info if eff_additional_info else None,
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
    detected_role = extract_jd_title(extracted_text)
    detected_quals = extract_jd_qualifications(extracted_text)
    detected_demo = parse_demographic_criteria(extracted_text)

    return {
        "success": True,
        "filename": jd_file.filename,
        "extraction_method": method,
        "char_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
        "detected_role": detected_role,
        "detected_qualifications": sorted(list(detected_quals)),
        "detected_experience": exp_info,
        "detected_demographics": detected_demo if detected_demo else None,
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


@app.post("/api/resumes/upload")
async def upload_and_ingest_resumes(files: List[UploadFile] = File(...)):
    """
    Upload and incrementally ingest 1 or more new resumes (PDF, DOCX, DOC, PNG, JPG, TXT):
      1. Saves original file into resumes/ directory.
      2. Parses resume through ATS pipeline into flat_json/.
      3. Uses sync_resume_to_qdrant() to check identity, run Smart Diff & Merge,
         and upsert into Qdrant collection 'resumes' (in-place if existing candidate).
      4. Returns ingested / updated candidate records and live collection stats.
    """
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Vector database not ready")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    ingested_resumes = []
    errors = []

    for upload_file in files:
        if not upload_file.filename:
            continue

        filename = upload_file.filename
        t_start = time.perf_counter()

        try:
            # 1. Save original file to resumes/ directory
            dest_file = RESUMES_DIR / filename
            content = await upload_file.read()
            with open(dest_file, "wb") as f:
                f.write(content)

            # 2. Parse resume with pure Python ATS parser
            flat_record = process_resume(
                file_path=str(dest_file),
                resume_id=0,
                output_dir=str(BASE_DIR / "flat_json"),
                verbose=False
            )

            # 3. Smart Diff, Identity Resolution & In-Place Qdrant Sync
            point_id = sync_resume_to_qdrant(
                parsed_record=flat_record,
                filename=filename,
                file_bytes=content
            )

            t_total = time.perf_counter() - t_start
            name = flat_record.get("name") or "Candidate"
            exp = flat_record.get("total_experience", 0.0)
            skills = flat_record.get("skills", []) or []

            ingested_resumes.append({
                "id": point_id,
                "name": name,
                "file_name": filename,
                "total_experience": exp,
                "designations": flat_record.get("designation", []) or [],
                "skills_count": len(skills),
                "skills_sample": skills[:8],
                "email": flat_record.get("email"),
                "mobile_number": flat_record.get("mobile_number"),
                "update_reasons": flat_record.get("update_reasons", []),
                "parsing_time_seconds": round(t_total, 2)
            })

        except Exception as e:
            errors.append({"file_name": filename, "error": str(e)})

    # Fetch updated collection stats
    try:
        col_info = qdrant_client.get_collection(COLLECTION_NAME)
        total_count = col_info.points_count
    except Exception:
        total_count = len(list((BASE_DIR / "flat_json").glob("*.json")))

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
