#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge all phrasebook languages into one table using the current RapidFuzz pipeline.

Default behavior:
    - load source phrasebook .docx files through guidedeconversationphrasebook_processing
    - use Nufi as the master language
    - merge every other language into that master table
"""

import argparse
import gc
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from tools.guidedeconversationphrasebook_processing import DEFAULT_BASE_PATH, LANGUAGE_PATHS, PhrasebookProcessor
from tools.test_rapidfuzz_merge import load_phrasebook_sheet, smart_merge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge all phrasebook languages with RapidFuzz")
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Base folder for source phrasebook .docx files.",
    )
    parser.add_argument(
        "--excel",
        help="Path to the Excel workbook containing language sheets, e.g. African_Languages_dataframes.xlsx",
    )
    parser.add_argument(
        "--master-language",
        default="Nufi",
        help="Master language column name. Defaults to Nufi.",
    )
    parser.add_argument(
        "--text-column",
        choices=["English", "French", "both"],
        default="both",
        help="Text basis for fuzzy matching after exact and normalized matching.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Minimum fuzzy score to accept a match.",
    )
    parser.add_argument(
        "--output",
        default="African_Languages_dataframes_rapidfuzz_merged.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--excel-output",
        default="African_Languages_dataframes_rapidfuzz_merged.xlsx",
        help="Output Excel workbook path with one sheet per language.",
    )
    parser.add_argument(
        "--checkpoint-every-language",
        action="store_true",
        help="Write the current merged CSV after each language merge.",
    )
    parser.add_argument(
        "--language-names",
        help="Optional comma-separated subset of language names to merge.",
    )
    return parser


def load_language_dataset_from_processor(processor: PhrasebookProcessor, language_name: str) -> pd.DataFrame:
    df = processor.dataframes[language_name][[language_name, "Francais", "Anglais"]].copy()
    df.columns = [language_name, "French", "English"]
    for col in [language_name, "French", "English"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df[["French", "English", language_name]]


def load_language_dataset(args: argparse.Namespace, language_name: str, processor: PhrasebookProcessor | None = None) -> pd.DataFrame:
    if args.excel:
        return load_phrasebook_sheet(args.excel, language_name)
    if processor is None:
        raise ValueError("Processor is required when loading from source files.")
    return load_language_dataset_from_processor(processor, language_name)


def build_language_sheet(master_df: pd.DataFrame, master_language: str, language_name: str) -> pd.DataFrame:
    if language_name == master_language:
        sheet = pd.DataFrame(
            {
                "GLOBAL_ID": master_df["GLOBAL_ID"],
                "Francais": master_df["French"],
                "Anglais": master_df["English"],
                language_name: master_df[language_name],
            }
        )
        return sheet

    counterpart_french_col = f"{language_name}__French"
    counterpart_english_col = f"{language_name}__English"
    sheet = pd.DataFrame(
        {
            "GLOBAL_ID": master_df["GLOBAL_ID"],
            "Francais": master_df.get(counterpart_french_col),
            "Anglais": master_df.get(counterpart_english_col),
            language_name: master_df[language_name],
        }
    )
    return sheet


def normalize_export_text(value):
    if pd.isna(value):
        return value
    return str(value).replace("’", "'").replace("`", "'")


def export_language_workbook(
    master_df: pd.DataFrame,
    available_languages: list[str],
    master_language: str,
    workbook_path: Path,
) -> None:
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for language_name in available_languages:
            if language_name not in master_df.columns:
                continue
            sheet_df = build_language_sheet(master_df, master_language, language_name)
            for column in ("Francais", "Anglais"):
                if column in sheet_df.columns:
                    sheet_df[column] = sheet_df[column].map(normalize_export_text)
            sheet_df.to_excel(writer, sheet_name=language_name[:31], index=False)


def build_csv_export_frame(master_df: pd.DataFrame, available_languages: list[str]) -> pd.DataFrame:
    csv_columns = ["GLOBAL_ID", "French", "English"] + [
        language_name for language_name in available_languages if language_name in master_df.columns
    ]
    return master_df[csv_columns].copy()


def get_available_languages(args: argparse.Namespace) -> list[str]:
    requested = None
    if args.language_names:
        requested = [name.strip() for name in args.language_names.split(",") if name.strip()]
    if args.excel:
        xls = pd.ExcelFile(args.excel)
        available = [name for name in sorted(LANGUAGE_PATHS.keys()) if name in xls.sheet_names]
    else:
        available = sorted(LANGUAGE_PATHS.keys())
    if requested is None:
        return available
    requested_set = set(requested)
    if args.master_language not in requested_set:
        requested_set.add(args.master_language)
    return [name for name in available if name in requested_set]


def main() -> None:
    args = build_parser().parse_args()

    available_languages = get_available_languages(args)
    if args.master_language not in available_languages:
        raise ValueError(f"Master language '{args.master_language}' is not in LANGUAGE_PATHS.")

    processor = None
    if not args.excel or args.master_language == "Nufi":
        processor = PhrasebookProcessor(base_path=args.base_path)
        print("Loading source phrasebooks once...")
        processor.load_all_languages(language_names=available_languages)
        print("Processing source phrasebooks once...")
        processor.process_all_languages()

    # Always use the true source Nufi phrasebook as the order authority.
    if args.master_language == "Nufi":
        master_df = load_language_dataset_from_processor(processor, args.master_language)
    else:
        master_df = load_language_dataset(args, args.master_language, processor)

    for lang_name in available_languages:
        if lang_name == args.master_language:
            continue
        print(f"Merging {lang_name} into {args.master_language}...")
        lang_df = load_language_dataset(args, lang_name, processor)
        master_df = smart_merge(
            df_master=master_df,
            df_new=lang_df,
            master_language=args.master_language,
            new_language=lang_name,
            text_column=args.text_column,
            threshold=args.threshold,
        )
        del lang_df
        gc.collect()

        if args.checkpoint_every_language:
            checkpoint_path = Path(args.output)
            master_df.to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
            filled = master_df[lang_name].fillna("").astype(str).str.strip().ne("").sum()
            print(f"Checkpoint saved to {checkpoint_path} after {lang_name} ({filled} filled rows)")

    # Preserve the original master-language order. Keep a unique GLOBAL_ID,
    # and store the original master-language source id separately.
    master_df = master_df.reset_index(drop=True)
    if "_source_id" in master_df.columns:
        source_ids = master_df["_source_id"].tolist()
        master_df = master_df.drop(columns=["_source_id"])
        master_df.insert(0, "NUFI_SOURCE_ID", source_ids)
        master_df.insert(0, "GLOBAL_ID", range(1, len(master_df) + 1))
    else:
        master_df.insert(0, "GLOBAL_ID", range(1, len(master_df) + 1))
    workbook_path = Path(args.excel_output)
    export_language_workbook(master_df, available_languages, args.master_language, workbook_path)
    print(f"Wrote per-language workbook to {workbook_path}")
    output_path = Path(args.output)
    csv_df = build_csv_export_frame(master_df, available_languages)
    csv_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote merged output to {output_path}")


if __name__ == "__main__":
    main()
