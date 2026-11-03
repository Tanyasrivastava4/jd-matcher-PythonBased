import sys
from pathlib import Path
import json

WORKSPACE = Path("/home/intileo/Desktop/jd-matcher")
sys.path.insert(0, str(WORKSPACE))

from run_pipeline import process_resume
from scripts.ingest_to_qdrant import build_candidate_search_text
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"

client = QdrantClient(url=QDRANT_URL)
embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)

targets = [
    (595, "Shubham Singhal_Head - Production (SFP Transceiver Manufacturing)_Intileo.pdf"),
    (597, "Nitin Arora_Head - Production (SFP Transceiver Manufacturing)_Intileo.pdf"),
]

for resume_id, filename in targets:
    file_path = WORKSPACE / "resumes" / filename
    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    print(f"\nReparsing ID {resume_id}: {filename}...")
    flat_record = process_resume(
        file_path=str(file_path),
        resume_id=resume_id,
        output_dir=str(WORKSPACE / "flat_json"),
        verbose=False
    )

    exp = flat_record.get("total_experience", 0.0)
    name = flat_record.get("name")
    print(f"  ✓ Name: {name}")
    print(f"  ✓ Calculated Experience: {exp} years")
    for comp in flat_record.get("company_details", [])[:6]:
        print(f"    - {comp}")

    # Embed and update Qdrant
    search_doc = build_candidate_search_text(flat_record)
    vector = list(embed_model.embed([search_doc]))[0].tolist()

    point = models.PointStruct(
        id=int(resume_id),
        vector=vector,
        payload=flat_record
    )
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print(f"  ✓ Updated Qdrant point {resume_id} successfully!")

print("\nAll done!")
