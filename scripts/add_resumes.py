"""
scripts/add_resumes.py
======================
Incremental Ingestion Tool for New Resumes.
Adds 1 or more new resumes to the existing database without wiping previous records.

Usage:
    # Add a single resume:
    python scripts/add_resumes.py path/to/new_resume.pdf

    # Add multiple resumes:
    python scripts/add_resumes.py resume1.pdf resume2.docx resume3.png

    # Add all files from a directory:
    python scripts/add_resumes.py --dir /path/to/new_folder
"""

import sys
import os
import glob
import json
import shutil
import time
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# Add workspace to path
WORKSPACE = Path("/home/intileo/Desktop/jd-matcher")
sys.path.insert(0, str(WORKSPACE))

from run_pipeline import process_resume
from scripts.ingest_to_qdrant import build_candidate_search_text

FLAT_JSON_DIR = WORKSPACE / "flat_json"
RESUMES_DIR = WORKSPACE / "resumes"
COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"


def get_next_resume_id() -> int:
    """Finds the next available counting ID from flat_json/."""
    max_id = 0
    if FLAT_JSON_DIR.exists():
        for p in FLAT_JSON_DIR.glob("*.json"):
            prefix = p.name.split("_")[0]
            if prefix.isdigit():
                max_id = max(max_id, int(prefix))
    return max_id + 1


def add_single_resume(
    file_path: Path,
    qdrant_client: QdrantClient,
    embed_model: TextEmbedding,
    assigned_id: int
) -> dict:
    """
    Parses a single resume, assigns the next ID, saves flat_json,
    embeds it, and upserts it directly into Qdrant.
    """
    t_start = time.perf_counter()
    print(f"\n[{assigned_id}] Ingesting: {file_path.name}")

    # 1. Ensure file is copied into resumes/ directory if not already there
    dest_path = RESUMES_DIR / file_path.name
    if file_path.resolve() != dest_path.resolve():
        RESUMES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        actual_source = dest_path
    else:
        actual_source = file_path

    # 2. Run ATS Parser (Layer 0 Ingestion/OCR, Layer 1 Segment, Layer 2-4 Extraction/Flat JSON)
    flat_record = process_resume(
        file_path=str(actual_source),
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
    skills = flat_record.get("skills", [])
    exp = flat_record.get("total_experience", 0.0)

    print(f"  ✓ Parsed & Upserted in {t_total:.2f}s:")
    print(f"    - ID        : {assigned_id}")
    print(f"    - Name      : {name}")
    print(f"    - Experience: {exp} years")
    print(f"    - Skills ({len(skills)}): {', '.join(skills[:6])}...")

    return {
        "id": assigned_id,
        "name": name,
        "file_name": actual_source.name,
        "experience": exp,
        "skills_count": len(skills),
        "time_seconds": round(t_total, 2)
    }


def ingest_files(file_paths: list[str]) -> list[dict]:
    """Processes a list of resume file paths sequentially into the system."""
    valid_paths = [Path(p) for p in file_paths if Path(p).exists() and Path(p).is_file()]
    if not valid_paths:
        print("No valid files found to ingest.")
        return []

    print(f"\n=======================================================")
    print(f" 📥 INCREMENTAL RESUME INGESTION ({len(valid_paths)} files)")
    print(f"=======================================================")

    print("Connecting to Qdrant and loading embedding model...")
    client = QdrantClient(url=QDRANT_URL)
    embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)

    current_id = get_next_resume_id()
    print(f"Next available ID in database: {current_id}")

    results = []
    for p in valid_paths:
        try:
            res = add_single_resume(p, client, embed_model, current_id)
            results.append(res)
            current_id += 1
        except Exception as e:
            print(f"  ✗ Failed to process {p.name}: {e}")

    # Check updated total count
    col_info = client.get_collection(COLLECTION_NAME)
    print(f"\n🎉 Ingestion finished! Total points in Qdrant: {col_info.points_count}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Add new resumes to Qdrant and flat_json.")
    parser.add_argument("files", nargs="*", help="Paths to individual resume files (.pdf, .docx, .png, etc.)")
    parser.add_argument("--dir", help="Directory containing new resume files to ingest")
    args = parser.parse_args()

    paths_to_process = []
    if args.files:
        paths_to_process.extend(args.files)

    if args.dir:
        folder = Path(args.dir)
        if folder.exists() and folder.is_dir():
            for ext in ["*.pdf", "*.docx", "*.doc", "*.png", "*.jpg", "*.jpeg", "*.txt"]:
                paths_to_process.extend(glob.glob(str(folder / ext)))
                paths_to_process.extend(glob.glob(str(folder / ext.upper())))

    if not paths_to_process:
        parser.print_help()
        sys.exit(1)

    ingest_files(paths_to_process)


if __name__ == "__main__":
    main()
