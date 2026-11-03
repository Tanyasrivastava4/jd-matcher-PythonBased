"""
scripts/ingest_to_qdrant.py
===========================
Ingests all parsed resume JSON records from flat_json/ into Qdrant Vector Database.

Key features:
  - 100% Local CPU Embedding via FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime).
  - Dimension: 384, Distance: Cosine.
  - High-signal semantic search document for each candidate.
  - Complete flat JSON stored in Qdrant Payload for instant display & filtering.
  - Point ID = sequential counting number (1, 2, ... 590).
  - Payload indexes created on skills, total_experience, and degree for instant hybrid filtering.
"""

import os
import glob
import json
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

WORKSPACE = Path("/home/intileo/Desktop/jd-matcher")
FLAT_JSON_DIR = WORKSPACE / "flat_json"
COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 64


def build_candidate_search_text(data: dict) -> str:
    """
    Synthesizes a high-signal search document for a candidate from their parsed fields.
    Optimized for matching against recruiter Job Descriptions (JDs).
    """
    parts = []

    name = data.get("name")
    if name and name.lower() not in ["unknown", "curriculum vitae", "resume", "profile"]:
        parts.append(f"Candidate: {name}")

    desigs = data.get("designation")
    if desigs and isinstance(desigs, list):
        clean_desigs = [d.replace("Designation |", "").strip() for d in desigs if d]
        if clean_desigs:
            parts.append(f"Designations: {', '.join(clean_desigs[:5])}")

    skills = data.get("skills")
    if skills and isinstance(skills, list):
        parts.append(f"Skills: {', '.join(skills)}")

    tot_exp = data.get("total_experience")
    if tot_exp is not None and tot_exp > 0:
        parts.append(f"Total Experience: {tot_exp} years")

    summary = data.get("summary")
    if summary and isinstance(summary, str):
        # Truncate summary to avoid dilution
        clean_summary = " ".join(summary.split())[:500]
        parts.append(f"Summary: {clean_summary}")

    degrees = data.get("degree")
    if degrees and isinstance(degrees, list):
        parts.append(f"Education: {', '.join(degrees)}")

    projects = data.get("projects")
    if projects and isinstance(projects, list):
        clean_projects = " ".join([str(p).strip() for p in projects[:3]])
        if clean_projects:
            parts.append(f"Projects: {clean_projects[:400]}")

    certs = data.get("certification")
    if certs and isinstance(certs, list):
        parts.append(f"Certifications: {', '.join(certs[:4])}")

    return "\n".join(parts)


def main():
    print("\n" + "=" * 65)
    print(" 🚀 INGESTING RESUMES INTO QDRANT VECTOR DATABASE")
    print(f" Qdrant URL        : {QDRANT_URL}")
    print(f" Collection Name   : {COLLECTION_NAME}")
    print(f" Embedding Model   : {EMBED_MODEL_NAME} (FastEmbed / ONNX Runtime)")
    print(f" Vector Dimension  : 384 (Cosine Distance)")
    print(f" Flat JSON Source  : {FLAT_JSON_DIR}/")
    print("=" * 65 + "\n")

    # 1. Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    # 2. Re-create collection
    print(f"Recreating collection '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

    # 3. Load FastEmbed model
    print(f"Loading FastEmbed model '{EMBED_MODEL_NAME}' on CPU...")
    t_load = time.perf_counter()
    embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    print(f"✓ Model loaded in {time.perf_counter() - t_load:.2f}s")

    # 4. Read all flat JSON records
    json_paths = sorted(
        FLAT_JSON_DIR.glob("*.json"),
        key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 999999
    )
    total_files = len(json_paths)
    print(f"Found {total_files} resume records to embed and ingest.\n")

    # Prepare data
    records_to_process = []
    texts_to_embed = []
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_text = build_candidate_search_text(data)
            records_to_process.append(data)
            texts_to_embed.append(doc_text)
        except Exception as e:
            print(f"Warning: Failed to read {p.name}: {e}")

    # 5. Batch Embed & Upsert
    t_start = time.perf_counter()
    total_upserted = 0

    print(f"Embedding and uploading in batches of {BATCH_SIZE}...")
    for i in range(0, len(records_to_process), BATCH_SIZE):
        batch_records = records_to_process[i:i + BATCH_SIZE]
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]

        # Generate vectors using FastEmbed ONNX runtime
        vectors = list(embed_model.embed(batch_texts))

        points = []
        for rec, vec in zip(batch_records, vectors):
            point_id = rec.get("id")
            if point_id is None:
                continue

            points.append(
                models.PointStruct(
                    id=int(point_id),
                    vector=vec.tolist(),
                    payload=rec
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_upserted += len(points)
        print(f"  ✓ Uploaded {total_upserted}/{total_files} points...")

    elapsed = time.perf_counter() - t_start
    print(f"\n✓ Successfully ingested {total_upserted} resumes into Qdrant in {elapsed:.2f}s ({total_upserted/elapsed:.1f} resumes/sec)!")

    # 6. Create Payload Indexes for Instant Filtering
    print("\nCreating payload indexes for fast hybrid filtering...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="skills",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="total_experience",
        field_schema=models.PayloadSchemaType.FLOAT
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="degree",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    print("✓ Payload indexes created on 'skills', 'total_experience', and 'degree'")

    # 7. Verification Test Query
    print("\n" + "=" * 65)
    print(" 🔍 RUNNING VERIFICATION SEARCH TEST")
    print("=" * 65)
    test_jd = """
    We are looking for a Senior Python / Data Scientist with experience in
    Machine Learning, Generative AI, RAG pipelines, Vector Databases (Qdrant),
    and building scalable backend APIs.
    """
    print(f"Query JD:\n{test_jd.strip()}\n")

    query_vector = list(embed_model.embed([test_jd]))[0].tolist()
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    print("Top 3 Matching Candidates Returned by Qdrant:")
    print("-" * 65)
    for rank, pt in enumerate(search_response.points, 1):
        payload = pt.payload
        print(f"#{rank} [Score: {pt.score:.4f}] - ID {pt.id}: {payload.get('name')}")
        print(f"    File       : {payload.get('file_name')}")
        print(f"    Email      : {payload.get('email')}")
        print(f"    Phone      : {payload.get('mobile_number')}")
        print(f"    Experience : {payload.get('total_experience')} yrs")
        skills_sample = payload.get('skills', [])[:8]
        print(f"    Top Skills : {', '.join(skills_sample)}")
        print("-" * 65)

    print("\n🎉 QDRANT INGESTION & TEST SEARCH COMPLETE!\n")


if __name__ == "__main__":
    main()
