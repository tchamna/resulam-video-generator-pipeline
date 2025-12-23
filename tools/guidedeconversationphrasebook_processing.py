#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
African Language Phrasebook Processing Tool

This module processes multilingual phrasebook data from Word documents containing
African language translations. It extracts, cleans, normalizes, and exports
phrasebook content across multiple African languages.

Key Features:
    - Extract text from Word documents (.docx)
    - Clean and normalize multilingual phrase entries
    - Apply text replacements and standardization rules
    - Export to multiple formats (CSV, Excel)
    - Support for 23+ African languages
    - Detect and handle proper names
    - Identify capitalization issues

Author: Resulam
Date: 2025
License: See LICENSE file
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from functools import reduce

import pandas as pd
import numpy as np
import docx2txt
import xlsxwriter

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_PATH = r"G:\My Drive\Mbú'ŋwɑ̀'nì"

LANGUAGE_PATHS = {
    # Cameroon languages
    'Basaa': r'Livres Basaa\Basaa_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'DualaDouala': r'Livres Duala\DualaDouala_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Ewondo': r'Livres Ewondo\Ewondo_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Ghomala': r'Livres Ghomala\Ghomala_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Medumba': r'Livres Medumba\Medumba_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Nufi': r'Livres Nufi\Nufi_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Yemba': r'Livres Yemba\Yemba_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Shupamom': r'Livres Bamoun\Bamoun_CMR_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',

    # Other African languages
    'Chichewa': r'Livres Chichewa\Chichewa_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'EweTogo': r'Livres EweTogo\EweTogo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'FulfuldeBenin': r'Livres Fulfulde\FulfuldeBenin_Benin_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'FulfuldeNigeria': r'Livres Fulfulde_Nigeria\FulfuldeNigeria_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Hausa': r'Livres Hausa\Hausa_Benin_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Igbo': r'Livres Igbo\Igbo_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Kikongo': r'Livres Kikongo\Kikongo_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Kinyarwanda': r'Livres Kinyarwanda\Kinyarwanda_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Lingala': r'Livres Lingala\Lingala_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Swahili': r'Livres Swahili\Swahili_TanzaniaKenya_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Tshiluba': r'Livres Tshiluba\Tshiluba_RDCongo_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Twi': r'Livres Twi\Twi_Ghana_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Wolof': r'Livres Wolof\Wolof_Senegal_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Yoruba': r'Livres Yoruba\Yoruba_Nigeria_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
    'Zulu': r'Livres Zulu\Zulu_Phrasebook_ExpressionsUsuelles_GuideConversation.docx',
}

PUNCTUATION_REPLACEMENTS = {
    " .": ".",
    " (": "(",
    " ;": ";",
    " ?": "?",
    " !": "!",
    "'": "'"
}

SENTENCE_REPLACEMENTS = {
    "Bonsoir.": "Bonsoir!",
    "Bonsoir maman.": "Bonsoir maman!",
    "Bonne cÃ©rÃ©monie d'anniversaire de mariage.": "Bonne cÃ©rÃ©monie d'anniversaire de mariage!",
    "Bonne cÃ©rÃ©monie de mariage.": "Bonne cÃ©rÃ©monie de mariage!",
    "C'est vrai, je t'assure.": "C'est vrai, je t'assure!",
    "DÃ©solÃ©(e), pardon.": "DÃ©solÃ©(e), pardon!",
    "Embrasse-moi s'il te plait.": "Embrasse-moi s'il te plait!",
    "Kiss me please.": "Kiss me please!",
    "Bonne nuit.": "Bonne nuit!",
    "Good evening.": "Good evening!",
    "Good night.": "Good night!",
    "Good evening mother.": "Good evening mother!",
    "Qu'il en soit ainsi.": "Qu'il en soit ainsi!",
    "Soyez les bienvenus.": "Soyez les bienvenus!",
}

# Combined replacements dictionary
ALL_REPLACEMENTS = {**PUNCTUATION_REPLACEMENTS, **SENTENCE_REPLACEMENTS}


# =============================================================================
# Core Processing Functions
# =============================================================================

def clean_phrase(phrase: str, sep1: str = "|", sep2: str = "/") -> str:
    """
    Clean and normalize a phrasebook entry.

    Processes a phrase containing translations separated by delimiters and
    normalizes the formatting, capitalization, and spacing.

    Args:
        phrase: Input phrase string with format "Language1 | Language2 | Language3"
        sep1: Primary separator (default: "|")
        sep2: Secondary separator (default: "/")

    Returns:
        Cleaned and normalized phrase string

    Example:
        >>> clean_phrase("hello | bonjour . /salut")
        "Salut | bonjour | hello."
    """
    parts = phrase.split(sep1)

    if len(parts) < 3:
        return phrase

    sentence1, sentence2, sentence3 = parts[0], parts[1], parts[2]

    # Remove extra whitespace
    sentence3 = ' '.join(sentence3.split())

    # Normalize punctuation spacing
    sentence3 = sentence3.replace(". /", "./")
    sentence3 = sentence3.replace("! /", "!/")
    sentence3 = sentence3.replace("? /", "?/")

    # Pattern matching for sentence endings
    delimiters = {".": r"\./", "!": "!/", "?": r"\?/"}

    split_text = []
    english = ""
    ghomala = ""

    for punct, pattern in delimiters.items():
        split_text = re.split(pattern, sentence3)

        if len(split_text) > 1 and "" not in split_text:
            english = split_text[0] + punct
            ghomala = split_text[-1].strip()

            if ghomala:
                ghomala = ghomala[0].upper() + ghomala[1:] + " "
            break

    # Return original if no valid split found
    if len(split_text) <= 1 or "" in split_text:
        return phrase

    # Reconstruct the phrase
    new_sentence = f"{sep1} ".join([ghomala, sentence2, english])
    clean_sentence = new_sentence.replace(sentence1, ghomala)
    clean_sentence = ' '.join(clean_sentence.split())

    return clean_sentence


def extract_text_from_docx(file_path: str) -> List[str]:
    """
    Extract phrasebook entries from a Word document.

    Args:
        file_path: Path to the .docx file

    Returns:
        List of phrases containing exactly 2 pipe separators

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract text from document
    text_content = docx2txt.process(file_path)

    # Clean up special characters
    text_content = text_content.replace("\xa0", "")
    text_content = text_content.replace("\t", "")

    # Split into lines and filter for valid entries
    lines = text_content.splitlines()
    phrases = [line for line in lines if line.count("|") == 2]

    return phrases


def read_text_file(file_path: str, encoding: str = 'utf-8') -> List[str]:
    """
    Read phrasebook entries from a text file.

    Args:
        file_path: Path to the text file
        encoding: Text encoding (default: 'utf-8')

    Returns:
        List of phrases containing exactly 2 pipe separators
    """
    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()

    return [line for line in lines if line.count("|") == 2]


def find_starting_index(phrases: List[str]) -> int:
    """
    Find the starting index where actual content begins.

    Looks for phrases containing "| Salut." or "| Salut!" as markers
    for where the phrasebook content starts.

    Args:
        phrases: List of phrase strings

    Returns:
        Index where content begins (returns index - 2 for safety margin)
    """
    for idx, text in enumerate(phrases):
        if "| Salut." in text or "| Salut!" in text:
            return max(0, idx - 2)
    return 0


def apply_multi_replacements(text: str, replacements: Dict[str, str]) -> str:
    """
    Apply multiple string replacements to text.

    Args:
        text: Input text string
        replacements: Dictionary mapping old strings to new strings

    Returns:
        Text with all replacements applied
    """
    for old_str, new_str in replacements.items():
        text = text.replace(old_str, new_str)
    return text


def convert_to_dataframe(
    phrases: List[str],
    language_name: str,
    from_line: int = 0
) -> pd.DataFrame:
    """
    Convert phrase list to pandas DataFrame.

    Args:
        phrases: List of phrase strings
        language_name: Name of the language
        from_line: Starting line index (default: 0)

    Returns:
        DataFrame containing the phrases
    """
    df = pd.DataFrame(phrases).iloc[from_line:]
    df = df.reset_index(drop=True)
    return df


def expand_columns(
    df: pd.DataFrame,
    language_name: str,
    source_column: int = 0,
    sep: str = "|"
) -> pd.DataFrame:
    """
    Expand a single column into multiple language columns.

    Splits phrases by separator and creates separate columns for each language.

    Args:
        df: Input DataFrame
        language_name: Name of the primary language
        source_column: Column index to expand (default: 0)
        sep: Separator character (default: "|")

    Returns:
        DataFrame with expanded columns: [Language, Francais, Anglais]
    """
    # Split the source column into three parts
    df[[language_name, "Francais", "Anglais"]] = df[source_column].str.split(
        sep, expand=True
    )

    # Filter rows where Francais is not None
    df = df[df["Francais"].notna()]

    # Strip whitespace from all string columns
    string_columns = [language_name, "Francais", "Anglais"]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df


def has_proper_name(text: str) -> bool:
    """
    Check if text contains proper names (capitalized words mid-sentence).

    Args:
        text: Input text string

    Returns:
        True if proper names are detected, False otherwise
    """
    special_conditions = [
        "Ã€ la fin de ce chapitre" in text,
        "fÃ¨'Ã©fÄ›'Ã¨" in text
    ]

    has_chapter = "Chapitre" not in text
    has_capitals = any(c.isupper() for c in text[1:]) if len(text) > 1 else False

    return (has_capitals or any(special_conditions)) and has_chapter


def starts_with_lowercase(text: str) -> bool:
    """
    Check if text starts with a lowercase letter.

    Args:
        text: Input text string

    Returns:
        True if starts with lowercase, False otherwise
    """
    if not text or len(text) == 0:
        return True

    first_char = text.strip()[0]

    if not first_char.isalpha():
        return True

    return first_char.lower() == first_char


# =============================================================================
# Data Processing Pipeline
# =============================================================================

class PhrasebookProcessor:
    """
    Main processor for African language phrasebooks.

    Handles loading, cleaning, transforming, and exporting multilingual
    phrasebook data from Word documents.
    """

    def __init__(self, base_path: str = DEFAULT_BASE_PATH):
        """
        Initialize the processor.

        Args:
            base_path: Base directory path for language files
        """
        self.base_path = base_path
        self.language_data = {}
        self.dataframes = {}
        self.language_names = []

    def load_all_languages(self) -> None:
        """Load all configured language files."""
        for lang_name, rel_path in sorted(LANGUAGE_PATHS.items()):
            full_path = os.path.join(self.base_path, rel_path)

            try:
                print(f"Loading {lang_name}...")
                phrases = extract_text_from_docx(full_path)
                start_idx = find_starting_index(phrases)
                self.language_data[lang_name] = phrases[start_idx:]
                self.language_names.append(lang_name)
            except Exception as e:
                print(f"Warning: Failed to load {lang_name}: {e}")

    def process_all_languages(self) -> None:
        """Process all loaded language data into DataFrames."""
        for lang_name in self.language_names:
            print(f"Processing {lang_name}...")

            # Convert to DataFrame
            df = convert_to_dataframe(
                self.language_data[lang_name],
                lang_name,
                from_line=0
            )

            # Expand columns
            df = expand_columns(df, lang_name)

            # Apply replacements
            df = df.applymap(lambda x: apply_multi_replacements(str(x), ALL_REPLACEMENTS))

            # Strip whitespace
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            self.dataframes[lang_name] = df

    def merge_all_dataframes(self) -> pd.DataFrame:
        """
        Merge all language DataFrames on French and English columns.

        Returns:
            Merged DataFrame containing all languages
        """
        merge_columns = ["Francais", "Anglais"]

        dataframes_to_merge = []
        for lang_name in self.language_names:
            subset = self.dataframes[lang_name][[lang_name, "Francais", "Anglais"]]
            dataframes_to_merge.append(subset)

        if not dataframes_to_merge:
            return pd.DataFrame()

        # Perform iterative outer merge
        merged_df = dataframes_to_merge[0]
        for df in dataframes_to_merge[1:]:
            merged_df = pd.merge(
                merged_df,
                df,
                on=merge_columns,
                how="outer"
            )
            merged_df = merged_df.drop_duplicates(subset=merge_columns)

        return merged_df

    def export_to_excel(
        self,
        output_path: str = "African_Languages_dataframes.xlsx"
    ) -> None:
        """
        Export all language DataFrames to separate Excel sheets.

        Args:
            output_path: Path for the output Excel file
        """
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for lang_name in self.language_names:
                subset_cols = ["Francais", "Anglais", lang_name]
                df_subset = self.dataframes[lang_name][subset_cols]

                # Excel sheet names are limited to 31 characters
                sheet_name = lang_name[:31]
                df_subset.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Exported to {output_path}")

    def export_merged_to_csv(
        self,
        output_path: str = "African_Languages_dataframes_merged.csv"
    ) -> None:
        """
        Export merged DataFrame to CSV.

        Args:
            output_path: Path for the output CSV file
        """
        merged_df = self.merge_all_dataframes()
        merged_df.to_csv(output_path, encoding="utf-8-sig", index=False)
        print(f"Exported merged data to {output_path}")

    def find_lowercase_issues(
        self,
        output_path: str = "not_starting_with_capital.xlsx"
    ) -> None:
        """
        Identify and export phrases not starting with capital letters.

        Args:
            output_path: Path for the output Excel file
        """
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            for lang_name in self.language_names:
                df = self.dataframes[lang_name]

                # Filter for lowercase starts
                mask = df[lang_name].apply(starts_with_lowercase)
                lowercase_df = df[mask]

                if not lowercase_df.empty:
                    lowercase_df.to_excel(writer, sheet_name=lang_name[:31])

        print(f"Exported capitalization issues to {output_path}")

    def identify_proper_names(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify entries containing proper names.

        Returns:
            Tuple of (entries_with_proper_names, entries_without_proper_names)
        """
        if not self.language_names:
            return pd.DataFrame(), pd.DataFrame()

        merged_df = self.merge_all_dataframes()
        last_lang = self.language_names[-1]

        # Find missing translations
        missing_mask = merged_df[last_lang].isna()
        data_missing = merged_df[missing_mask]

        # Separate proper names from regular missing entries
        proper_name_mask = data_missing["Francais"].apply(has_proper_name)
        proper_names_df = data_missing[proper_name_mask]
        regular_missing_df = data_missing[~proper_name_mask]

        return proper_names_df, regular_missing_df


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("African Language Phrasebook Processor")
    print("=" * 50)

    # Initialize processor
    processor = PhrasebookProcessor()

    # Load all language files
    print("\nLoading language files...")
    processor.load_all_languages()
    print(f"Loaded {len(processor.language_names)} languages")

    # Process all languages
    print("\nProcessing languages...")
    processor.process_all_languages()

    # Export results
    print("\nExporting results...")
    processor.export_to_excel("African_Languages_dataframes.xlsx")
    processor.export_merged_to_csv("African_Languages_dataframes_merged.csv")

    # Quality checks
    print("\nRunning quality checks...")
    processor.find_lowercase_issues("not_starting_with_capital.xlsx")

    proper_names, missing = processor.identify_proper_names()
    if not proper_names.empty:
        proper_names.to_csv("data_ProperNames.csv", encoding="utf-8-sig", index=False)
        print(f"Found {len(proper_names)} entries with proper names")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

