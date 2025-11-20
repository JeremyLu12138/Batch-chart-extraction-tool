import os
import json
from pathlib import Path

import pdfplumber
import pandas as pd
from openai import OpenAI

# ============ BASIC CONFIG ============
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_DIR = "input-test"   # folder containing PDFs
OUTPUT_DIR = "output"      # folder for generated Excel files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ MODEL: classify whether a table is a rates/premium table ============

CLASSIFIER_SYSTEM_PROMPT = """
You are a table classifier for Australian insurance / superannuation PDFs.

You will receive a JSON payload:
{
  "page_index": X,
  "page_text": "...",
  "table_preview": [...]
}

Your task is to decide whether the table is a **rates / premium table** (e.g., cost / rate / weekly cost / TPD / IP / Income Protection).

A table MUST satisfy **at least 2** of the following to be considered a rates table:

1. The table contains substantial numeric data (0.034, 0.011, 1.20, 35, 70, etc.).
2. The table header contains any of the following keywords:
   - age, premium, rate, cost, fee, death, tpd, ip, income protection, waiting period, benefit period
3. The page_text contains any of the following:
   - cost, rate, premium, insurance, cover, benefit period, waiting period, IP, Income Protection, Death, TPD
4. The table structure is at least 3 rows + 3 columns.

The following MUST be excluded (never a rates table):
- contact, help, call us, beneficiary, claim, summary, introduction tables
- anything like "If you need to make a claim…"
- tables with almost no numeric content
- very small tables (1–2 rows or columns)
- FAQ, procedural, or descriptive tables

Output strict JSON only:

{
  "is_rate_table": true/false,
  "sheet_title": "xxx"   // empty if is_rate_table = false
}
"""


def classify_table_with_model(page_index: int, page_text: str, table_preview, model: str = "gpt-4o-mini") -> dict:
    """
    Robust rate table identification strategy:
    1. Structural check (avoid false positives)
    2. Numeric-row detection (core rule)
    3. Keyword detection
    4. Model-based classification
    """

    # ========== STRUCTURAL FILTER ==========
    num_rows = len(table_preview)
    num_cols = max([len(r) for r in table_preview]) if table_preview else 0

    if num_rows < 4 or num_cols < 4:
        return {"is_rate_table": False, "sheet_title": ""}

    # ========== NUMERIC ROW CHECK (strong indicator of rate tables) ==========
    def is_numeric_row(row):
        digit_cells = 0
        total = len(row)
        for c in row:
            if isinstance(c, str) and any(ch.isdigit() for ch in c):
                digit_cells += 1
        return (digit_cells / total) >= 0.5  # ≥50% digits means this is a numeric row

    numeric_row_count = sum(is_numeric_row(r) for r in table_preview)

    # A legit rates table should have at least 2 numeric rows
    if numeric_row_count < 2:
        return {"is_rate_table": False, "sheet_title": ""}

    # ========== KEYWORD CHECK (light, not strict) ==========
    KEYWORDS = [
        "cost", "rate", "premium", "weekly", "monthly",
        "death", "tpd", "income protection",
        "waiting period", "benefit period", "cover"
    ]

    text_lower = (page_text or "").lower()
    hit_keyword = any(k in text_lower for k in KEYWORDS)

    # Even if keyword missed → model will still decide
    # (so no early exit here)

    # ========== CALL MODEL FOR FINAL DECISION ==========
    payload = {
        "page_index": page_index,
        "page_text": text_lower[:1500],
        "table_preview": table_preview
    }

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0,
        max_tokens=800
    )

    raw = resp.choices[0].message.content
    return json.loads(raw)


def process_single_pdf(pdf_path: Path, model: str = "gpt-4o-mini"):
    print(f"\nProcessing PDF: {pdf_path.name}")
    rate_tables = []  # list of {sheet_title, data}

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []

            if not tables:
                continue

            for t_idx, tbl in enumerate(tables):
                if not tbl:
                    continue

                # extract preview: first 6 rows, first 8 columns
                preview = [row[:8] for row in tbl[:6]]

                try:
                    cls = classify_table_with_model(page_idx, page_text, preview, model=model)
                except Exception as e:
                    print(f"  [WARN] Model classification failed page={page_idx+1}, table={t_idx+1}: {e}")
                    continue

                if not cls.get("is_rate_table"):
                    continue  # skip non-rate tables

                raw_title = cls.get("sheet_title") or f"Rates page {page_idx+1} table {t_idx+1}"
                sheet_title = raw_title.strip() or f"Rates page {page_idx+1} table {t_idx+1}"

                rate_tables.append({
                    "sheet_title": sheet_title,
                    "data": tbl
                })

    if not rate_tables:
        print("  No rate tables detected. Skipping.")
        return

    # === WRITE TO EXCEL ===
    out_path = Path(OUTPUT_DIR) / f"{pdf_path.stem}_rates.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        used_names = set()

        for idx, tbl in enumerate(rate_tables):
            base_name = tbl["sheet_title"][:31]  # Excel sheet name limit = 31 chars
            name = base_name
            k = 1
            while name in used_names:
                suffix = f"_{k}"
                name = (base_name[: (31 - len(suffix))] + suffix)
                k += 1
            used_names.add(name)

            df = pd.DataFrame(tbl["data"])
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"  Excel generated → {out_path}")


def process_all_pdfs(model: str = "gpt-4o-mini"):
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {INPUT_DIR}")

    if not pdf_files:
        print("Please add PDF files to the input folder.")
        return

    for pdf_path in pdf_files:
        process_single_pdf(pdf_path, model=model)


if __name__ == "__main__":
    process_all_pdfs()
