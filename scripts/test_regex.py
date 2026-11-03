import re

MONTH_NAMES = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|Apil|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_VALID_YEAR = r"(?:19[6-9]\d|20[0-3]\d)"

# Old regex:
_MONTH_YEAR_OLD = rf"(?:{MONTH_NAMES}[\s,/.-]*(?:{_VALID_YEAR}|['’‘`´]?\s*\d{{2}})|{_VALID_YEAR}[\s,/.-]*{MONTH_NAMES})"

# New regex:
_SEP = r"[\s,/.\-–—'’‘`´]*"
_MONTH_YEAR_NEW = rf"(?:{MONTH_NAMES}{_SEP}(?:{_VALID_YEAR}|['’‘`´]?\s*\d{{2}})|{_VALID_YEAR}{_SEP}{MONTH_NAMES})"

test_dates = ["Mar’2026", "Oct’2025", "March'2017", "Sept'2020", "Aug’2020"]

print("--- OLD REGEX ---")
for td in test_dates:
    m = re.search(_MONTH_YEAR_OLD, td)
    print(f"{td} -> {m.group(0) if m else None}")

print("\n--- NEW REGEX ---")
for td in test_dates:
    m = re.search(_MONTH_YEAR_NEW, td)
    print(f"{td} -> {m.group(0) if m else None}")
