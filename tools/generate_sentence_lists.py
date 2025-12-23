#!/usr/bin/env python3
"""Generate sentence list TXT files from an Excel workbook.

Usage:
  python tools/generate_sentence_lists.py \
      --excel "G:\\My Drive\\Mbú'ŋwɑ̀'nì\\Livres Nufi\\African_Languages_Phrasebook_GuideConversationExpressionsUsuelles_TabularPrototype_bon.xlsx" \
      --out-dir assets/Languages

The script reads all sheets (or a subset) and looks for columns matching
English, French and a local language column. It writes files named
`<language>_english_french_phrasebook_sentences_list.txt` with lines:

1) English text | Local text | French text

If a sheet lacks an English or French column the script will still try to
produce lines using whatever columns are available.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import pandas as pd


def find_column(columns, patterns):
    for pat in patterns:
        for c in columns:
            if re.search(pat, str(c), re.IGNORECASE):
                return c
    return None


def detect_columns(df: pd.DataFrame, sheet_name: str | None = None):
    cols = list(df.columns)
    id_col = find_column(cols, [r"^id$", r"\bID\b", r"^\s*#\s*$", r"^\s*no\.?\s*$", r"^\s*n°\s*$"])
    eng = find_column(cols, [r"^eng", r"anglais", r"english"])
    fr = find_column(cols, [r"^fr", r"franc", r"french"])

    # local: first try obvious names, then try a column that matches the sheet name,
    # then fall back to the first non-eng/non-fr column that isn't an ID/unnamed column.
    local = find_column(cols, [r"local", r"langue", r"native", r"lang"])
    if not local and sheet_name:
        try:
            sheet_pattern = re.escape(str(sheet_name).strip())
            local = find_column(cols, [sheet_pattern])
        except Exception:
            local = None

    if not local:
        for c in cols:
            if c == eng or c == fr:
                continue
            name = str(c)
            # skip ID-like, unnamed or empty columns
            if re.search(r"\bID\b", name, re.IGNORECASE) or re.search(r"unnamed", name, re.IGNORECASE):
                continue
            if re.match(r"^\s*$", name):
                continue
            local = c
            break

    return id_col, eng, local, fr


def normalize_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # replace newlines with spaces
    s = re.sub(r"\s+", " ", s)
    return s


def process_sheet(name: str, df: pd.DataFrame, out_root: Path):
    id_col, eng_col, local_col, fr_col = detect_columns(df, sheet_name=name)
    lang_key = re.sub(r"\s+", "_", name.strip()).lower()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{lang_key}_english_french_phrasebook_sentences_list.txt"

    print(
        f"Processing sheet '{name}': id={id_col}, eng={eng_col}, local={local_col}, fr={fr_col} -> {out_path}"
    )

    lines = []
    for _, row in df.iterrows():
        if id_col is not None:
            raw_id = row.get(id_col, "")
            try:
                idx = int(str(raw_id).strip())
            except Exception:
                idx = None
        else:
            idx = None

        eng = normalize_text(row.get(eng_col, "")) if eng_col is not None else ""
        local = normalize_text(row.get(local_col, "")) if local_col is not None else ""
        fr = normalize_text(row.get(fr_col, "")) if fr_col is not None else ""

        # Skip empty rows (all blank)
        if not (eng or local or fr):
            continue

        # Format: N) English | Local | French
        if idx is None:
            # Fallback if ID missing/invalid: use row index + 1
            idx = len(lines) + 1
        line = f"{idx}) {eng} | {local} | {fr}"
        lines.append(line)

    # Write file (UTF-8)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Wrote {len(lines)} lines to {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parent.parent

    use_private = os.getenv("USE_PRIVATE_ASSETS", "1").strip().lower()
    use_private_flag = use_private in ("1", "true", "yes")

    # Defaults depend on USE_PRIVATE_ASSETS: if true -> private_assets, else -> assets + G: workbook
    default_private_excel = project_root / "private_assets" / "African_Languages_Phrasebook_GuideConversationExpressionsUsuelles_TabularPrototype_bon.xlsx"
    default_public_excel = Path(r"G:\My Drive\Mbú'ŋwɑ̀'nì\Livres Nufi\African_Languages_Phrasebook_GuideConversationExpressionsUsuelles_TabularPrototype_bon.xlsx")

    if use_private_flag:
        default_excel_path = default_private_excel
        default_out_dir = project_root / "private_assets" / "Languages"
    else:
        default_excel_path = default_public_excel
        default_out_dir = project_root / "assets" / "Languages"

    p.add_argument("--excel", default=str(default_excel_path), help=f"Path to the Excel workbook (default: {default_excel_path})")
    p.add_argument("--out-dir", default=str(default_out_dir), help=f"Base output folder (default: {default_out_dir})")
    p.add_argument("--sheets", nargs="*", help="Optional list of sheet names to process")
    args = p.parse_args(argv)

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        sys.exit(2)

    try:
        xl = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        print(f"Error reading Excel: {e}")
        sys.exit(3)

    sheets = args.sheets if args.sheets else list(xl.keys())
    for sheet in sheets:
        if sheet not in xl:
            print(f"Sheet not found: {sheet}")
            continue
        df = xl[sheet]
        # Output folder per language: assets/Languages/<SheetTitle>Phrasebook/
        out_root = Path(args.out_dir) / f"{sheet}Phrasebook"
        process_sheet(sheet, df, out_root)


if __name__ == "__main__":
    main()
