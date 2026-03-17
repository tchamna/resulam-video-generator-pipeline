#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic DOCX batch utilities for arbitrary uploaded Word documents.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.guidedeconversationphrasebook_processing import (
    extract_text_from_docx,
    load_json_mapping,
    replace_text_in_word_documents_inplace,
)


def find_docx_files(input_dir: str) -> list[str]:
    return sorted(str(path) for path in Path(input_dir).rglob("*.docx"))


def load_explicit_input_files(input_files_json: str | None) -> list[str]:
    if not input_files_json:
        return []
    data = json.loads(Path(input_files_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input-files-json must contain a JSON array of file paths")
    files = []
    for item in data:
        path = Path(str(item))
        if path.suffix.lower() != ".docx":
            continue
        files.append(str(path))
    return files


def build_nbsp_replacements(report_rows: list[dict[str, str]], file_path: str) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for row in report_rows:
        if row.get("file_path") != file_path:
            continue
        original_pair = row.get("original_pair", "")
        fixed_pair = row.get("fixed_pair", "")
        if original_pair and fixed_pair and original_pair != fixed_pair:
            replacements[original_pair] = fixed_pair
    return replacements


def export_report(report_rows: list[dict[str, str]], csv_path: str, xlsx_path: str) -> None:
    df = pd.DataFrame(report_rows)
    if df.empty:
        df = pd.DataFrame(columns=["file_path", "left_word", "right_word", "original_pair", "fixed_pair", "nbsp_type"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All", index=False)
        if "file_path" in df.columns:
            for file_path, group in df.groupby("file_path"):
                sheet_name = Path(str(file_path)).stem[:31] or "Sheet"
                group.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic DOCX cleanup batch")
    parser.add_argument("--input-dir", required=True, help="Directory containing uploaded .docx files")
    parser.add_argument("--input-files-json", help="Optional JSON array of explicit .docx paths to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and copies")
    parser.add_argument("--doc-replacements-json", help="Optional JSON file with extra text replacements")
    parser.add_argument("--report-only", action="store_true", help="Only generate the NBSP report")
    parser.add_argument("--create-fixed-copies", action="store_true", help="Create fixed copies instead of editing in place")
    parser.add_argument("--inplace", action="store_true", help="Apply fixes in place to the uploaded docs")
    args = parser.parse_args()

    input_files = load_explicit_input_files(args.input_files_json) or find_docx_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No .docx files found under {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, str]] = []
    for file_path in input_files:
        extract_text_from_docx(file_path, report_rows)

    export_report(
        report_rows,
        str(output_dir / "generic_nbsp_report.csv"),
        str(output_dir / "generic_nbsp_report.xlsx"),
    )

    if args.report_only:
        return

    extra_replacements = load_json_mapping(args.doc_replacements_json)
    if not args.create_fixed_copies and not args.inplace and not extra_replacements:
        return

    targets: list[tuple[str, str]] = []
    if args.create_fixed_copies:
        copies_dir = output_dir / "fixed_docx"
        for source in input_files:
            source_path = Path(source)
            target = copies_dir / source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            targets.append((source, str(target)))
    else:
        targets = [(path, path) for path in input_files]

    for source, target in targets:
        replacements = build_nbsp_replacements(report_rows, source)
        replacements.update(extra_replacements)
        if not replacements:
            continue
        replace_text_in_word_documents_inplace([target], replacements=replacements, visible=False)


if __name__ == "__main__":
    main()
