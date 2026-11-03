"""
scripts/audit_resumes.py
========================
Performs a rigorous, line-by-line and field-by-field audit across all 590 resumes.
Compares original raw text (extracted_text/) against parsed output (flat_json/)
to determine whether each resume received complete, accurate, and faithful extraction.
"""

import os
import glob
import json
import re
from pathlib import Path
from collections import Counter

WORKSPACE = Path("/home/intileo/Desktop/jd-matcher")
FLAT_JSON_DIR = WORKSPACE / "flat_json"
RAW_TEXT_DIR = WORKSPACE / "extracted_text"

# Regex for email detection in raw text
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Regex for Indian & International phone numbers
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{4,6}")

# Common false-positive / boilerplate emails to exclude from candidate audit
SYSTEM_EMAILS = {"support@naukri.com", "feedback@naukri.com", "services@naukri.com", "info@intileo.com"}

DEGREE_KEYWORDS = [
    r"\bb\.?tech\b", r"\bb\.?e\b", r"\bm\.?tech\b", r"\bbca\b", r"\bmca\b",
    r"\bmba\b", r"\bb\.?sc\b", r"\bm\.?sc\b", r"\bb\.?com\b", r"\bm\.?com\b",
    r"\bbachelor\b", r"\bmaster\b", r"\bdiploma\b", r"\bph\.?d\b", r"\b10th\b", r"\b12th\b",
    r"\bhsc\b", r"\bssc\b", r"\bmatriculation\b", r"\bintermediate\b", r"\bdegree\b"
]
DEGREE_PATTERN = re.compile("|".join(DEGREE_KEYWORDS), re.IGNORECASE)

NON_RESUME_KEYWORDS = ["offer letter", "monthly report", "joining date", "compensation package", "basic salary", "sign in", "microsoftonline"]


def clean_phone(p: str) -> str:
    """Normalize phone to digits only."""
    return re.sub(r"\D", "", p)


def audit_single_resume(json_path: Path) -> dict:
    stem = json_path.stem.replace("_flat", "")
    with open(json_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    resume_id = parsed.get("id")
    file_name = parsed.get("file_name", "")

    # Find matching raw text
    raw_path = RAW_TEXT_DIR / f"{stem}.txt"
    if not raw_path.exists():
        # Fallback search by ID
        matches = list(RAW_TEXT_DIR.glob(f"{resume_id}_*.txt"))
        if matches:
            raw_path = matches[0]

    raw_text = ""
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    raw_len = len(raw_text.strip())
    raw_lower = raw_text.lower()

    # Document Type Classification
    is_non_resume = False
    non_resume_reason = ""
    if raw_len < 120:
        is_non_resume = True
        non_resume_reason = "Extremely short text (< 120 chars, image/screenshot)"
    elif "offer letter" in file_name.lower() or "offer of employment" in raw_lower:
        is_non_resume = True
        non_resume_reason = "Employment Offer Letter document"
    elif "monthly report" in file_name.lower():
        is_non_resume = True
        non_resume_reason = "Spreadsheet / Monthly Report"
    elif "login.microsoftonline.com" in raw_lower:
        is_non_resume = True
        non_resume_reason = "Login / screenshot file"

    # --- 1. EMAIL AUDIT ---
    raw_emails_raw = EMAIL_REGEX.findall(raw_text)
    raw_emails = []
    for em in raw_emails_raw:
        em_clean = em.strip(".,;:()<>[]'\"").lower()
        if em_clean not in SYSTEM_EMAILS and len(em_clean) > 5 and "@" in em_clean:
            if em_clean not in raw_emails:
                raw_emails.append(em_clean)

    parsed_email_str = str(parsed.get("email") or "").strip().lower()
    parsed_emails = [e.strip() for e in re.split(r"[\s/,;]+", parsed_email_str) if "@" in e]

    email_status = "OK"
    missing_emails = []
    if raw_emails and not parsed_emails:
        email_status = "MISSED_EMAIL"
        missing_emails = raw_emails
    elif not raw_emails and not parsed_emails:
        email_status = "NOT_IN_ORIGINAL"
    elif len(raw_emails) > len(parsed_emails):
        # Check if candidate had multiple and we missed one
        for re_em in raw_emails:
            if not any(re_em in pe for pe in parsed_emails):
                missing_emails.append(re_em)
        if missing_emails:
            email_status = "PARTIAL_EMAIL"

    # --- 2. PHONE AUDIT ---
    raw_phones_raw = PHONE_REGEX.findall(raw_text)
    raw_valid_phones = []
    for ph in raw_phones_raw:
        digits = clean_phone(ph)
        # Indian phone numbers typically have 10 digits (or 11 with leading 0, or 12 with 91)
        if 10 <= len(digits) <= 13:
            # Check not a year like 20202021 or postal code
            if not (digits.startswith("19") or digits.startswith("20")) or len(digits) > 8:
                if digits not in [clean_phone(x) for x in raw_valid_phones]:
                    raw_valid_phones.append(ph.strip())

    parsed_phone_str = str(parsed.get("mobile_number") or "").strip()
    parsed_phone_digits = [clean_phone(p) for p in re.split(r"[/,;]", parsed_phone_str) if len(clean_phone(p)) >= 10]

    phone_status = "OK"
    if raw_valid_phones and not parsed_phone_digits:
        phone_status = "MISSED_PHONE"
    elif not raw_valid_phones and not parsed_phone_digits:
        phone_status = "NOT_IN_ORIGINAL"

    # --- 3. NAME AUDIT ---
    name = str(parsed.get("name") or "").strip()
    name_status = "OK"
    if not name or name.lower() in ["unknown", "curriculum vitae", "resume", "profile", "cv", "null"]:
        name_status = "MISSING_NAME"
    elif len(name.split()) > 5:
        name_status = "NAME_TOO_LONG"

    # --- 4. SKILLS AUDIT ---
    skills = parsed.get("skills") or []
    skills_count = len(skills)
    skills_status = "OK"
    if skills_count == 0:
        if is_non_resume:
            skills_status = "NON_RESUME"
        else:
            skills_status = "NO_SKILLS_FOUND"
    elif skills_count < 3:
        skills_status = "FEW_SKILLS"

    # --- 5. EDUCATION AUDIT ---
    education = parsed.get("EDUCATION") or []
    degrees = parsed.get("degree") or []
    has_parsed_edu = bool(education or degrees)
    has_raw_edu_keywords = bool(DEGREE_PATTERN.search(raw_text))

    edu_status = "OK"
    if has_raw_edu_keywords and not has_parsed_edu:
        edu_status = "MISSED_EDUCATION"
    elif not has_raw_edu_keywords and not has_parsed_edu:
        edu_status = "NOT_IN_ORIGINAL"

    # --- 6. EXPERIENCE AUDIT ---
    exp_details = parsed.get("company_details") or []
    designations = parsed.get("designation") or []
    total_exp = parsed.get("total_experience")
    has_parsed_exp = bool(exp_details or designations or (total_exp is not None and total_exp > 0))

    exp_status = "OK"
    if not has_parsed_exp:
        exp_status = "NO_EXPERIENCE_EXTRACTED"

    # --- 7. OVERALL VERDICT ---
    if is_non_resume:
        verdict = "NON_RESUME_DOC"
    elif (
        email_status in ["OK", "NOT_IN_ORIGINAL"]
        and phone_status in ["OK", "NOT_IN_ORIGINAL"]
        and name_status == "OK"
        and skills_count >= 3
        and edu_status in ["OK", "NOT_IN_ORIGINAL"]
        and exp_status == "OK"
    ):
        verdict = "COMPLETE_AND_CORRECT"
    elif email_status == "MISSED_EMAIL" or phone_status == "MISSED_PHONE" or edu_status == "MISSED_EDUCATION":
        verdict = "ACTIONABLE_DISCREPANCY"
    else:
        verdict = "ACCEPTABLE_WITH_NOTES"

    return {
        "id": resume_id,
        "file_name": file_name,
        "raw_char_len": raw_len,
        "verdict": verdict,
        "is_non_resume": is_non_resume,
        "non_resume_reason": non_resume_reason,
        "name": name,
        "name_status": name_status,
        "email_status": email_status,
        "parsed_email": parsed_email_str,
        "missing_emails": missing_emails,
        "phone_status": phone_status,
        "parsed_phone": parsed_phone_str,
        "skills_count": skills_count,
        "skills_status": skills_status,
        "edu_status": edu_status,
        "degree_count": len(degrees),
        "exp_status": exp_status,
        "total_experience": total_exp,
        "parsing_time": parsed.get("parsing_time_seconds", 0.0),
    }


def run_full_audit():
    json_files = sorted(FLAT_JSON_DIR.glob("*.json"), key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 999999)
    print(f"Auditing {len(json_files)} resumes against their raw extractions...")

    results = []
    verdict_counter = Counter()

    for idx, jf in enumerate(json_files, 1):
        res = audit_single_resume(jf)
        results.append(res)
        verdict_counter[res["verdict"]] += 1

    # Save detailed JSON report
    report_json_path = WORKSPACE / "audit_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("AUDIT RESULTS SUMMARY ACROSS ALL 590 RESUMES")
    print("=" * 60)
    for v, c in verdict_counter.most_common():
        pct = (c / len(results)) * 100
        print(f"  * {v:30s}: {c:3d} ({pct:5.1f}%)")

    # Detailed break-down
    missed_emails = [r for r in results if r["email_status"] in ["MISSED_EMAIL", "PARTIAL_EMAIL"]]
    missed_phones = [r for r in results if r["phone_status"] == "MISSED_PHONE"]
    missed_edu = [r for r in results if r["edu_status"] == "MISSED_EDUCATION"]
    no_skills = [r for r in results if r["skills_status"] == "NO_SKILLS_FOUND"]
    non_resumes = [r for r in results if r["is_non_resume"]]

    print("\n" + "-" * 60)
    print(f"Non-resume / Corrupted files detected: {len(non_resumes)}")
    for nr in non_resumes:
        print(f"  - ID #{nr['id']}: {nr['file_name']} -> {nr['non_resume_reason']}")

    print("\n" + "-" * 60)
    print(f"Email discrepancies (missed from raw text): {len(missed_emails)}")
    for me in missed_emails[:10]:
        print(f"  - ID #{me['id']}: {me['file_name']} -> Missed: {me['missing_emails']} (Parsed: {me['parsed_email']})")

    print("\n" + "-" * 60)
    print(f"Phone discrepancies (missed from raw text): {len(missed_phones)}")
    for mp in missed_phones[:10]:
        print(f"  - ID #{mp['id']}: {mp['file_name']} -> Raw had phone candidates, parsed: {mp['parsed_phone']}")

    print("\n" + "-" * 60)
    print(f"Education discrepancies (degree keyword in raw but not parsed): {len(missed_edu)}")
    for medu in missed_edu[:10]:
        print(f"  - ID #{medu['id']}: {medu['file_name']}")

    print("\n" + "-" * 60)
    print(f"Zero skills detected: {len(no_skills)}")
    for ns in no_skills:
        print(f"  - ID #{ns['id']}: {ns['file_name']}")

    return results


if __name__ == "__main__":
    run_full_audit()
