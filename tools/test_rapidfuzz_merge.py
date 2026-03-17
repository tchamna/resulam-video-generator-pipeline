#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test phrase alignment using exact FR/EN matches first, then fuzzy matching.

Expected input:
    Source phrasebook .docx files loaded through guidedeconversationphrasebook_processing,
    or alternatively an existing Excel workbook / standalone CSVs.
"""

import argparse
import sys
import re
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.guidedeconversationphrasebook_processing import (
    DEFAULT_BASE_PATH,
    PhrasebookProcessor,
    normalize_merge_key_text,
)


def ensure_rapidfuzz():
    try:
        from rapidfuzz import fuzz, process
    except ImportError as exc:
        raise ImportError(
            "rapidfuzz is required for this script. "
            "Install it with: pip install rapidfuzz"
        ) from exc
    return fuzz, process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test RapidFuzz-based phrase merge")
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Base folder for source phrasebook .docx files.",
    )
    parser.add_argument(
        "--excel",
        help="Path to the Excel workbook containing language sheets, e.g. African_Languages_dataframes.xlsx",
    )
    parser.add_argument("--master", help="Path to master CSV, e.g. nufi.csv")
    parser.add_argument("--new", help="Path to new language CSV, e.g. ewondo.csv")
    parser.add_argument("--master-language", required=True, help="Master language column name, e.g. Nufi")
    parser.add_argument("--new-language", required=True, help="New language column name, e.g. Ewondo")
    parser.add_argument(
        "--text-column",
        choices=["English", "French", "both"],
        default="both",
        help="Text basis for fuzzy matching after exact FR/EN matching.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Minimum fuzzy score to accept a match.",
    )
    parser.add_argument(
        "--output",
        default="merged_dataset_rapidfuzz.csv",
        help="Output CSV path.",
    )
    return parser


def load_phrasebook_csv(path: str, language_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) < 3:
        raise ValueError(f"{path} must have at least 3 columns")

    df = df.iloc[:, :3].copy()
    df.columns = ["French", "English", language_column]
    df["_source_id"] = range(1, len(df) + 1)
    for col in ["French", "English", language_column]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df[["_source_id", "French", "English", language_column]]


def load_phrasebook_sheet(workbook_path: str, language_column: str) -> pd.DataFrame:
    df = pd.read_excel(workbook_path, sheet_name=language_column)
    needed = [language_column, "Francais", "Anglais"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{language_column}' is missing columns: {missing}")

    df = df[[language_column, "Francais", "Anglais"]].copy()
    df = df.rename(columns={"Francais": "French", "Anglais": "English"})
    # The authoritative source order is the physical sheet row order.
    df["_source_id"] = range(1, len(df) + 1)
    for col in [language_column, "French", "English"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df[["_source_id", "French", "English", language_column]]


def load_phrasebook_from_source(base_path: str, language_column: str) -> pd.DataFrame:
    processor = PhrasebookProcessor(base_path=base_path)
    processor.load_all_languages()
    if language_column not in processor.language_data:
        raise ValueError(f"Language '{language_column}' was not loaded from source files.")
    processor.language_names = [language_column]
    processor.language_data = {language_column: processor.language_data[language_column]}
    processor.process_all_languages()
    df = processor.dataframes[language_column][[language_column, "Francais", "Anglais", "LOCAL_ID"]].copy()
    df = df.rename(columns={"Francais": "French", "Anglais": "English", "LOCAL_ID": "_source_id"})
    for col in [language_column, "French", "English"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df[["_source_id", "French", "English", language_column]]


def build_match_text(df: pd.DataFrame, text_column: str) -> pd.Series:
    if text_column == "French":
        return df["French"]
    if text_column == "English":
        return df["English"]
    return "French: " + df["French"] + " | English: " + df["English"]


def is_informative_text(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if normalize_merge_key_text(text) == "":
        return False
    if not re.search(r"[A-Za-zÀ-ÿ0-9]", text):
        return False
    return True


def row_has_informative_anchor(row: pd.Series) -> bool:
    return is_informative_text(row["French"]) or is_informative_text(row["English"])


STOPWORDS = {
    "a", "an", "and", "at", "de", "des", "du", "et", "i", "il", "is", "je",
    "l", "la", "le", "les", "my", "of", "the", "to", "un", "une", "you",
}


def normalized_tokens(value: str) -> set[str]:
    normalized = normalize_merge_key_text(str(value or ""))
    return {
        token
        for token in normalized.split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def has_strong_anchor_overlap(master_row: pd.Series, new_row: pd.Series) -> bool:
    for column in ("French", "English"):
        master_tokens = normalized_tokens(master_row[column])
        new_tokens = normalized_tokens(new_row[column])
        if not master_tokens or not new_tokens:
            continue
        overlap = master_tokens & new_tokens
        if len(overlap) >= 2:
            return True
        if overlap and (
            len(overlap) / min(len(master_tokens), len(new_tokens))
        ) >= 0.6:
            return True
    return False


def smart_merge(
    df_master: pd.DataFrame,
    df_new: pd.DataFrame,
    master_language: str,
    new_language: str,
    text_column: str,
    threshold: float,
) -> pd.DataFrame:
    """
    1. Exact match on French and English.
    2. Template-normalized match on French and English.
    3. Fuzzy match the remaining rows against the selected text basis.
    """
    fuzz, process = ensure_rapidfuzz()
    master = df_master.copy()
    new_df = df_new.copy()
    counterpart_french_col = f"{new_language}__French"
    counterpart_english_col = f"{new_language}__English"

    if new_language not in master.columns:
        master[new_language] = pd.NA
    if counterpart_french_col not in master.columns:
        master[counterpart_french_col] = pd.NA
    if counterpart_english_col not in master.columns:
        master[counterpart_english_col] = pd.NA

    master["_norm_french"] = master["French"].map(normalize_merge_key_text)
    master["_norm_english"] = master["English"].map(normalize_merge_key_text)
    new_df["_norm_french"] = new_df["French"].map(normalize_merge_key_text)
    new_df["_norm_english"] = new_df["English"].map(normalize_merge_key_text)

    new_df = new_df.reset_index(drop=True)
    used_new_indices = set()
    matchable_new_indices = set()

    exact_queues = defaultdict(deque)
    norm_queues = defaultdict(deque)
    for new_idx, row in new_df.iterrows():
        if not row_has_informative_anchor(row):
            continue
        matchable_new_indices.add(new_idx)
        exact_queues[(row["French"], row["English"])].append(new_idx)
        norm_queues[(row["_norm_french"], row["_norm_english"])].append(new_idx)

    def pop_unused(queue: deque):
        while queue:
            new_idx = queue.popleft()
            if new_idx not in used_new_indices:
                return new_idx
        return None

    # Stage 1: exact FR+EN one-to-one matching.
    for master_idx, row in master.iterrows():
        if not row_has_informative_anchor(row):
            continue
        exact_key = (row["French"], row["English"])
        new_idx = pop_unused(exact_queues[exact_key])
        if new_idx is None:
            continue
        master.at[master_idx, new_language] = new_df.at[new_idx, new_language]
        master.at[master_idx, counterpart_french_col] = new_df.at[new_idx, "French"]
        master.at[master_idx, counterpart_english_col] = new_df.at[new_idx, "English"]
        used_new_indices.add(new_idx)

    # Stage 2: normalized/template FR+EN one-to-one matching.
    for master_idx, row in master.iterrows():
        if not row_has_informative_anchor(row):
            continue
        current_value = str(master.at[master_idx, new_language]).strip() if not pd.isna(master.at[master_idx, new_language]) else ""
        if current_value:
            continue
        norm_key = (row["_norm_french"], row["_norm_english"])
        new_idx = pop_unused(norm_queues[norm_key])
        if new_idx is None:
            continue
        master.at[master_idx, new_language] = new_df.at[new_idx, new_language]
        master.at[master_idx, counterpart_french_col] = new_df.at[new_idx, "French"]
        master.at[master_idx, counterpart_english_col] = new_df.at[new_idx, "English"]
        used_new_indices.add(new_idx)

    unmatched_new = new_df[
        new_df.index.isin(matchable_new_indices) & ~new_df.index.isin(used_new_indices)
    ].copy()

    if unmatched_new.empty:
        return master.drop(columns=["_norm_french", "_norm_english"])

    master_match_text = build_match_text(master, text_column)
    unmatched_match_text = build_match_text(unmatched_new, text_column)
    available_master = master[
        (
            master[new_language].isna()
            | (master[new_language].fillna("").astype(str).str.strip() == "")
        )
        & master.apply(row_has_informative_anchor, axis=1)
    ].copy()
    available_match_text = build_match_text(available_master, text_column)
    choices = available_match_text.tolist()
    choice_indices = available_master.index.tolist()

    for new_idx, candidate_text in unmatched_match_text.items():
        best_match = process.extractOne(candidate_text, choices, scorer=fuzz.WRatio)
        if not best_match:
            continue
        _, score, match_pos = best_match
        if float(score) < threshold:
            continue
        master_idx = choice_indices[match_pos]
        if not has_strong_anchor_overlap(master.loc[master_idx], unmatched_new.loc[new_idx]):
            continue
        if pd.isna(master.at[master_idx, new_language]) or str(master.at[master_idx, new_language]).strip() == "":
            master.at[master_idx, new_language] = unmatched_new.at[new_idx, new_language]
            master.at[master_idx, counterpart_french_col] = unmatched_new.at[new_idx, "French"]
            master.at[master_idx, counterpart_english_col] = unmatched_new.at[new_idx, "English"]
            choices[match_pos] = ""

    return master.drop(columns=["_norm_french", "_norm_english"])


def main() -> None:
    args = build_parser().parse_args()
    ensure_rapidfuzz()

    if args.excel:
        master_df = load_phrasebook_sheet(args.excel, args.master_language)
        new_df = load_phrasebook_sheet(args.excel, args.new_language)
    elif args.master and args.new:
        master_df = load_phrasebook_csv(args.master, args.master_language)
        new_df = load_phrasebook_csv(args.new, args.new_language)
    else:
        master_df = load_phrasebook_from_source(args.base_path, args.master_language)
        new_df = load_phrasebook_from_source(args.base_path, args.new_language)

    merged = smart_merge(
        df_master=master_df,
        df_new=new_df,
        master_language=args.master_language,
        new_language=args.new_language,
        text_column=args.text_column,
        threshold=args.threshold,
    )

    output_path = Path(args.output)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote merged output to {output_path}")


if __name__ == "__main__":
    main()
