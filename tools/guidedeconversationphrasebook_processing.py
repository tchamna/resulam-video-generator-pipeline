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
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
import argparse
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Mapping
from functools import reduce
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import docx2txt
import xlsxwriter
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_PATH = r"G:\My Drive\Mbú'ŋwɑ̀'nì"
EQUIVALENCE_MAP_CSV = "equivalence_map.csv"
EQUIVALENCE_SUGGESTIONS_CSV = "equivalence_suggestions.csv"

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


def normalize_merge_key_text(text: Any) -> str:
    """
    Normalize key text used for cross-language merge alignment.

    This reduces false mismatches caused by punctuation/spacing variants
    (e.g. " : " vs ":", "…" vs "...", "(09 :45)" vs "(09:45)").
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\u00a0", " ")
    s = s.replace("\u202f", " ")
    s = s.replace("…", "...")
    s = s.replace("’", "'")
    s = s.replace("`", "'")
    s = s.strip("\"'“”‘’")

    # Collapse whitespace and normalize spacing around punctuation.
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+\(", "(", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s*:\s*", ":", s)
    s = re.sub(r"([.!?])\(", r"\1 (", s)
    s = re.sub(r"(?<=[0-9A-Za-z])\(", " (", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Ignore trailing parenthetical glosses/annotations which vary across
    # sources but should not split merge keys. This covers:
    # - "sentence (Litt. xxx)" == "sentence (Litt. yyy)"
    # - "sentence (Lit. xxx)" == "sentence (Lit. yyy)"
    # - "sentence (xxx)." == "sentence (yyy)."
    # - "sentence (xxx)!" == "sentence (yyy)!"
    # - "sentence (xxx)" == "sentence (yyy)"
    # Only trailing parenthetical groups are removed to avoid changing the
    # core meaning of sentences with inline parentheses.
    s = re.sub(r"(?:\s*\([^)]*\)[.?!;:]*\s*)+$", "", s).strip()

    # Ignore trailing square-bracket annotations, e.g.:
    # "Ma belle-sœur est malade. [C'est l'époux qui parle]"
    s = re.sub(r"\s*\[[^\]]*\]\s*$", "", s).strip()

    # Be tolerant to missing terminal punctuation across source files.
    # Example: "The oven/microwave" vs "The oven/microwave."
    s = re.sub(r"[.?!;:]+$", "", s).strip()

    # Normalize known teacher/professor template variants.
    # Handles:
    # - "Je suis enseignant (e)/professeur (e)"
    # - "Je suis enseignant (e)"
    # - "I am a teacher."
    # - "I am a teacher/professor."
    s = re.sub(
        r"(?i)^je suis enseignant\(e\)(?:/professeur\(e\))?$",
        "je suis enseignant(e)/professeur(e)",
        s,
    )
    s = re.sub(
        r"(?i)^i am a teacher(?:/professor)?$",
        "i am a teacher/professor",
        s,
    )

    # Normalize name-template variants so examples with concrete names match
    # placeholder forms:
    # "Je m'appelle Awa mon nom est Awa" == "Je m'appelle X mon nom est X"
    s = re.sub(
        r"(?i)^je m'appelle\s+([^\s.,;:!?()]+)\s+mon nom est\s+([^\s.,;:!?()]+)$",
        "je m'appelle X mon nom est X",
        s,
    )

    # Additional safe normalization for key matching:
    # - Case-insensitive merge keys
    # - Ignore punctuation-only differences inside the sentence
    # - Normalize common spacing/hyphen variants
    s = s.lower()
    # Accent-insensitive matching (e.g. "À table" vs "A table").
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[-‐‑‒–—]", " ", s)
    s = re.sub(r"\bkg\b", "kg", s)
    s = re.sub(r"[.,;:!?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Many books prepend a proverb/introduction before the actual chapter
    # learning objective. Keep the objective itself for merge alignment.
    if "a la fin de ce chapitre" in s:
        s = re.sub(r"^.*?(a la fin de ce chapitre\b)", r"\1", s).strip()
    if ("by the end of this chapter" in s) or ("at the end of this chapter" in s):
        s = re.sub(r"^.*?\b((?:by|at) the end of this chapter\b)", r"\1", s).strip()

    # Generalize example/template rows that differ only by sample names.
    # The source books often use different concrete names for the same
    # conversational slot; merge keys should align those examples.
    if re.match(r"^je m'appelle .+$", s) or re.match(r"^mon .*loge.*\s*est .+$", s):
        s = "self name example"
    elif re.match(r"^my (?:honorific )?name is .+$", s):
        s = "self name example"

    relationship_name_patterns = [
        (r"^mon ami s'appelle .+$", "friend name example"),
        (r"^ami s'appelle .+$", "friend name example"),
        (r"^my friend's name is .+$", "friend name example"),
        (r"^c'est mon ami(?: e)?$", "he or she is my friend"),
        (r"^il/elle est mon ami(?: e)?$", "he or she is my friend"),
        (r"^he/she is my friend$", "he or she is my friend"),
        (r"^it's my friend$", "he or she is my friend"),
        (r"^il s'appelle .+$", "his name example"),
        (r"^his name is .+$", "his name example"),
        (r"^elle s'appelle .+$", "her name example"),
        (r"^her name is .+$", "her name example"),
        (r"^ils s'appellent .+$", "their names example"),
        (r"^elles s'appellent .+$", "their names example"),
        (r"^their name(?:s)? are .+$", "their names example"),
        (r"^mon fr.* s'appelle .+$", "my brother's name example"),
        (r"^fr.* s'appelle .+$", "my brother's name example"),
        (r"^my brother's name is .+$", "my brother's name example"),
        (r"^ma s.*ur s'appelle .+$", "my sister's name example"),
        (r"^s.*ur s'appelle .+$", "my sister's name example"),
        (r"^my sister's name is .+$", "my sister's name example"),
        (r"^mon p.*re s'appelle .+$", "my father's name example"),
        (r"^p.*re s'appelle .+$", "my father's name example"),
        (r"^my father's name is .+$", "my father's name example"),
        (r"^ma m.*re s'appelle .+$", "my mother's name example"),
        (r"^m.*re s'appelle .+$", "my mother's name example"),
        (r"^my mother's name is .+$", "my mother's name example"),
        (r"^mon mari s'appelle .+$", "my husband's name example"),
        (r"^mari s'appelle .+$", "my husband's name example"),
        (r"^my husband's name is .+$", "my husband's name example"),
        (r"^ma femme s'appelle .+$", "my wife's name example"),
        (r"^femme s'appelle .+$", "my wife's name example"),
        (r"^my wife's name is .+$", "my wife's name example"),
        (r"^mon enseignant s'appelle .+$", "my teacher's name example"),
        (r"^enseignant s'appelle .+$", "my teacher's name example"),
        (r"^my teacher is called .+$", "my teacher's name example"),
        (r"^mon professeur s'appelle .+$", "my teacher's name example"),
        (r"^professeur s'appelle .+$", "my teacher's name example"),
    ]
    for pattern, replacement in relationship_name_patterns:
        if re.match(pattern, s):
            s = replacement
            break

    person_name_patterns = [
        (r"^c'est [a-z0-9'-]+(?: [a-z0-9'-]+){0,2}$", "c'est x person"),
        (r"^it is [a-z0-9'-]+(?: [a-z0-9'-]+){0,2}$", "it is x person"),
        (r"^je suis l'enfant de papa .+$", "i am papa x's child"),
        (r"^je suis l enfant de papa .+$", "i am papa x's child"),
        (r"^i am papa .+'s child$", "i am papa x's child"),
        (r"^i am the child of daddy .+$", "i am papa x's child"),
        (r"^i am the child of papa .+$", "i am papa x's child"),
        (r"^i am the son of daddy .+$", "i am papa x's child"),
        (r"^i am the daughter of daddy .+$", "i am papa x's child"),
        (r"^notre village est .+$", "our village is x"),
        (r"^our village is .+$", "our village is x"),
        (r"^cher m .+$", "cher m x"),
        (r"^dear mr .+$", "dear mr x"),
    ]
    for pattern, replacement in person_name_patterns:
        if re.match(pattern, s):
            s = replacement
            break

    # Generalize location examples that differ only by the concrete place.
    location_patterns = [
        (r"^j'habite .+$", "i live in x place"),
        (r"^j'habite (?:a|au|aux|en) .+$", "i live in x place"),
        (r"^i live in .+$", "i live in x place"),
        (r"^i am from .+$", "i am from x place"),
    ]
    for pattern, replacement in location_patterns:
        if re.match(pattern, s):
            s = replacement
            break
    if re.match(r"^je suis de\s+[a-z0-9' \-]+$", s) and not re.match(r"^je suis de (?:bonne|mauvaise) humeur$", s):
        s = "i am from x place"

    # Generalize language-name examples that differ only by the concrete
    # language being mentioned.
    language_patterns = [
        (r"^je sais parler le\s+.+\s+un tout petit peu$", "i can speak x language a little"),
        (r"^je sais parler la\s+.+\s+un tout petit peu$", "i can speak x language a little"),
        (r"^je sais parler l\s+.+\s+un tout petit peu$", "i can speak x language a little"),
        (r"^je peux parler le\s+.+\s+un tout petit peu$", "i can speak x language a little"),
        (r"^i can speak\s+.+\s+a little$", "i can speak x language a little"),
        (r"^i can speak\s+.+\s+a little bit$", "i can speak x language a little"),
        (r"^i speak\s+.+\s+a little$", "i can speak x language a little"),
        (r"^i speak\s+.+\s+a little bit$", "i can speak x language a little"),
        (r"^i know how to speak\s+.+\s+a little$", "i can speak x language a little"),
        (r"^i know how to speak\s+.+\s+a little bit$", "i can speak x language a little"),
    ]
    for pattern, replacement in language_patterns:
        if re.match(pattern, s):
            s = replacement
            break

    # Generalize sentences that differ only by a named item/food/person.
    named_item_patterns = [
        (r"^j'aime le .+$", "j'aime le x"),
        (r"^j'aime la .+$", "j'aime la x"),
        (r"^i like .+$", "i like x"),
        (r"^je ne parle pas bien .+$", "je ne parle pas bien x"),
        (r"^i do not speak .+ well$", "i do not speak x well"),
        (r"^i don't speak .+ well$", "i do not speak x well"),
    ]
    for pattern, replacement in named_item_patterns:
        if re.match(pattern, s):
            s = replacement
            break

    # Curated equivalence rules for frequent near-identical variants.
    s = s.replace("en bonne santé", "en santé")
    s = s.replace("je ne sais pas moi", "je ne sais pas")
    s = s.replace("do not suffer from anything", "do not suffer anything")
    s = s.replace("best regards (at the end of a letter)", "best wishes (at the end of a letter)")
    s = s.replace("best regards", "best wishes")
    s = s.replace("veux tu faire les selles veux tu aller a la selle veux tu chier", "veux tu faire les selles veux tu aller a la selle veux tu faire caca")
    s = s.replace("do you want to go to the bathroom do you want to shit/excrete", "do you want to go to the bathroom do you want to excrete")
    s = s.replace("si c'est vrai je t'acheterai a boire sinon je te frapperai", "si c'est vrai je t'acheterais a boire sinon je te frapperais")
    s = s.replace("if it is true i will buy you some drinks if not i will hit you", "if it is true i will buy you some drinks if not i will slap you")
    s = s.replace("where is your home/village", "where is your village")
    s = s.replace("joyeux noel/joyeux jour de naissance de l'enfant de dieu", "joyeux noel")
    s = s.replace("soyez la bienvenue", "bienvenue")
    s = s.replace(
        "what is the meaning of this what is this all about what's this story about",
        "what is the meaning of this what is this all about",
    )
    s = s.replace(
        "you have exaggerated you have crossed the line",
        "you exaggerated you crossed the line",
    )
    s = s.replace("le matin dans la matinee", "le matin")
    s = s.replace(
        "desoles nous ne pouvons pas entrer nous sommes en retard",
        "desole nous ne pouvons pas entrer nous sommes en retard",
    )
    s = s.replace("j'aime tellement la biere", "j'aime beaucoup la biere")
    s = s.replace(
        "you are better",
        "yours is better",
    )
    s = re.sub(r"^what are your friends'? name$", "what are your friends names", s)
    s = re.sub(r"^what are your friends'? names$", "what are your friends names", s)
    s = re.sub(r"^what is your friends'? name$", "what are your friends names", s)
    s = s.replace("i'm going to greet him", "i am going to greet him")
    s = s.replace("i am going greet him", "i am going to greet him")
    s = s.replace("i am going greet", "i am going to greet him")
    s = s.replace("welcome to many people", "welcome")
    s = s.replace("soyez les bienvenus", "bienvenue")
    s = s.replace("soyez le bienvenu", "bienvenue")
    s = s.replace(
        "fine good morning my husband",
        "good morning my husband",
    )
    s = s.replace(
        "c'est grand dommage pour moi litt c'est une grande perte pour moi",
        "c'est grand dommage pour moi",
    )
    # Generalize chapter-outcome language mentions:
    # "À la fin de ce chapitre ... en langue hausa" -> "... en langue x"
    # "By the end of this chapter ... in the hausa language" -> "... in the x language"
    if "a la fin de ce chapitre" in s:
        s = re.sub(r"\ben langue\s+[a-z0-9' \-]+\b", "en langue x", s)
    if ("by the end of this chapter" in s) or ("at the end of this chapter" in s):
        s = re.sub(r"\bin the\s+[a-z0-9' \-]+\s+language\b", "in the x language", s)
    return s


def _pair_key(fr: str, en: str) -> str:
    return f"{fr}\u241f{en}"


def _split_pair_key(key: str) -> Tuple[str, str]:
    if "\u241f" not in key:
        return key, ""
    fr, en = key.split("\u241f", 1)
    return fr, en


def _resolve_equivalent_pair_key(
    key: str,
    equivalence_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Resolve a normalized FR/EN pair through the approved equivalence map.

    Supports chained mappings while remaining safe against accidental cycles.
    """
    if not key or not equivalence_map:
        return key

    seen = set()
    resolved = key
    while resolved in equivalence_map and resolved not in seen:
        seen.add(resolved)
        next_key = equivalence_map[resolved]
        if not next_key or next_key == resolved:
            break
        resolved = next_key
    return resolved


def _canonical_merge_pair(
    fr_text: Any,
    en_text: Any,
    equivalence_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Build the canonical normalized merge pair for a FR/EN row.
    """
    key = _pair_key(
        normalize_merge_key_text(fr_text),
        normalize_merge_key_text(en_text),
    )
    return _split_pair_key(_resolve_equivalent_pair_key(key, equivalence_map))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _raw_pair_key(fr: Any, en: Any) -> Tuple[str, str]:
    return (str(fr).strip(), str(en).strip())


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


def replace_text_in_word_documents_inplace(
    document_paths: List[str],
    replacements: Mapping[str, str],
    visible: bool = False,
) -> Dict[str, bool]:
    """
    Replace text in Word documents in place while preserving formatting.

    This uses Microsoft Word automation to perform native Find/Replace on
    `.docx` files, which keeps layout, styles, and embedded images intact.

    Args:
        document_paths: Absolute paths to `.docx` files to modify in place
        replacements: Mapping of source text -> replacement text
        visible: Whether to show the Word UI while processing

    Returns:
        Mapping of document path -> whether at least one replacement was made

    Raises:
        ImportError: If `pywin32` is not installed
        RuntimeError: If Word automation cannot be started
    """
    if not replacements:
        return {}

    try:
        import win32com.client  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "replace_text_in_word_documents_inplace requires pywin32 "
            "(win32com.client) and Microsoft Word on Windows."
        ) from exc

    results: Dict[str, bool] = {}

    # Longest-first reduces accidental partial replacements when keys overlap.
    ordered_replacements = sorted(
        ((str(old), str(new)) for old, new in replacements.items() if str(old)),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    # Pre-compute which pairs are "dangerous" (old is a substring of new).
    # Only these need corruption fix and Pass 0 protection.
    dangerous_pairs = [
        (idx, old, new)
        for idx, (old, new) in enumerate(ordered_replacements)
        if old != new and old in new
    ]

    # Flush helper so every log line appears immediately in the terminal.
    def _log(msg: str) -> None:
        tqdm.write(msg)
        sys.stdout.flush()
        sys.stderr.flush()

    # Per-document timeout (seconds).  If Word hangs on a document (e.g.
    # Google-Drive sync dialog), we kill the Word process and restart.
    DOC_TIMEOUT = 90

    def _start_word():
        """Create and configure a fresh Word COM instance."""
        w = win32com.client.DispatchEx("Word.Application")
        w.Visible = visible
        w.DisplayAlerts = 0  # wdAlertsNone
        try:
            w.Options.SaveNormalPrompt = False
        except Exception:
            pass
        try:
            w.Options.DoNotPromptForConvert = True
        except Exception:
            pass
        try:
            # msoAutomationSecurityForceDisable = 3
            w.AutomationSecurity = 3
        except Exception:
            pass
        return w

    def _kill_word_process():
        """Force-kill all WINWORD.EXE processes."""
        try:
            os.system('taskkill /F /IM WINWORD.EXE >nul 2>&1')
        except Exception:
            pass
        time.sleep(1)

    def _process_one_doc(word_app, document_path, doc_name):
        """Open, find/replace, close one document.  Returns (replaced_any, error_msg|None)."""
        doc = None
        replaced_any = False
        try:
            _log(f"    [open] Opening {doc_name}…")
            doc = word_app.Documents.Open(
                FileName=str(document_path),
                ConfirmConversions=False,
                ReadOnly=False,
                AddToRecentFiles=False,
                Visible=visible,
                OpenAndRepair=False,
                NoEncodingDialog=True,
            )
            _log(f"    [open] Opened")

            def _do_replace_all(find_text, replace_text, use_wildcards=False):
                rng = doc.Content
                fnd = rng.Find
                fnd.ClearFormatting()
                fnd.Replacement.ClearFormatting()
                return bool(fnd.Execute(
                    find_text,               # FindText
                    not use_wildcards,        # MatchCase
                    False,                    # MatchWholeWord
                    use_wildcards,            # MatchWildcards
                    False,                    # MatchSoundsLike
                    False,                    # MatchAllWordForms
                    True,                     # Forward
                    1,                        # Wrap = wdFindContinue
                    False,                    # Format
                    replace_text,             # ReplaceWith
                    2,                        # Replace = wdReplaceAll
                ))

            _log(f"    [1/4] Cleaning placeholders…")
            # Clean up both old-style and new-style leftover placeholders.
            if _do_replace_all("__PBR_*__", "", use_wildcards=True):
                _log(f"    ⚠ Cleaned leftover placeholders")
            if _do_replace_all("__PHRASEBOOK_REPLACE_*__", "", use_wildcards=True):
                _log(f"    ⚠ Cleaned old-style leftover placeholders")
                replaced_any = True

            if dangerous_pairs:
                _log(f"    [2/4] Checking for corruption ({len(dangerous_pairs)} pair(s))…")
                for _, old_text, new_text in dangerous_pairs:
                    suffix = new_text[len(old_text):]
                    for n in range(6, 0, -1):
                        corrupted = new_text + suffix * n
                        if _do_replace_all(corrupted, new_text):
                            _log(f"    ✓ Fixed: '{old_text[:40]}…'")
                            replaced_any = True
                            break

            _log(f"    [3/4] Replacing ({len(ordered_replacements)} rules)…")
            ph_map = {
                i: f"__PBR_{i}_{uuid.uuid4().hex[:6]}__"
                for i in range(len(ordered_replacements))
            }
            used = set()

            for idx, _old, new_text in dangerous_pairs:
                _do_replace_all(new_text, ph_map[idx])
                used.add(idx)

            for idx, (old_text, new_text) in enumerate(ordered_replacements):
                if _do_replace_all(old_text, ph_map[idx]):
                    used.add(idx)
                    replaced_any = True

            _log(f"    [4/4] Restoring ({len(used)} placeholder(s))…")
            for idx in sorted(used):
                _do_replace_all(ph_map[idx], ordered_replacements[idx][1])

            if replaced_any:
                doc.Save()
                _log(f"    ✓ Saved {doc_name}")
            else:
                _log(f"    – No changes needed")

            doc.Close(SaveChanges=False)
            doc = None
            return replaced_any, None

        except Exception as exc:
            return False, str(exc)
        finally:
            if doc is not None:
                try:
                    doc.Close(SaveChanges=False)
                except Exception:
                    pass

    # ---------- main loop ----------
    # Strategy: copy each Google-Drive file to a local temp folder before
    # opening it in Word.  This eliminates hangs caused by Google Drive File
    # Stream locking / sync dialogs.  After editing, copy the file back.
    word = None
    tmp_dir = tempfile.mkdtemp(prefix="phrasebook_")
    _log(f"Using temp dir: {tmp_dir}")
    try:
        _log("Starting Word…")
        word = _start_word()
        _log("Word started")

        for document_path in tqdm(document_paths, desc="Replacing in docs", unit="doc"):
            doc_name = os.path.basename(document_path)
            _log(f"  → {doc_name}")

            # 1. Copy to local temp
            local_path = os.path.join(tmp_dir, doc_name)
            try:
                _log(f"    [copy] Copying to temp…")
                shutil.copy2(document_path, local_path)
                _log(f"    [copy] Done")
            except Exception as exc:
                _log(f"    ✗ Copy failed: {exc}")
                results[document_path] = False
                continue

            # 2. Process the local copy
            try:
                ok, err = _process_one_doc(word, local_path, doc_name)
                if err:
                    _log(f"    ✗ Error: {err}")
                    # COM errors leave Word in a bad state — restart it.
                    _log(f"    Restarting Word…")
                    try:
                        word.Quit()
                    except Exception:
                        _kill_word_process()
                    word = _start_word()
                    _log(f"    Word restarted")
                    results[document_path] = False
                else:
                    results[document_path] = ok
                    # 3. Copy back only if we made changes
                    if ok:
                        _log(f"    [copy-back] Writing back to original location…")
                        shutil.copy2(local_path, document_path)
                        _log(f"    [copy-back] Done")
            except Exception as exc:
                _log(f"    ✗ Error: {exc}")
                results[document_path] = False
                # If Word died, restart it
                _log(f"    Restarting Word…")
                try:
                    word.Quit()
                except Exception:
                    _kill_word_process()
                word = _start_word()
                _log(f"    Word restarted")
            finally:
                # Clean up temp file
                try:
                    os.remove(local_path)
                except Exception:
                    pass
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                _kill_word_process()
        # Clean up temp dir
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return results


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
        self.equivalence_map: Dict[str, str] = {}
        self._merged_cache: Optional[pd.DataFrame] = None

    def ensure_equivalence_map_template(self, path: str = EQUIVALENCE_MAP_CSV) -> None:
        p = Path(path)
        if p.exists():
            return
        template = pd.DataFrame([
            {
                "source_fr": "c'est dommage de parler à ses parents sans respect",
                "source_en": "it's a shame to talk to one's parents without respect",
                "target_fr": "c'est dommage de manquer de respect à ses parents",
                "target_en": "it's a shame to disrespect one's parents",
                "approved": 0,
                "notes": "Set approved=1 after manual review.",
            }
        ])
        template.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"Created template equivalence map: {path}")

    def load_equivalence_map(self, path: str = EQUIVALENCE_MAP_CSV) -> Dict[str, str]:
        p = Path(path)
        mapping: Dict[str, str] = {}
        if not p.exists():
            return mapping

        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            return mapping

        needed = {"source_fr", "source_en", "target_fr", "target_en"}
        if not needed.issubset(set(df.columns)):
            print(f"Warning: {path} missing required columns: {sorted(needed)}")
            return mapping

        if "approved" in df.columns:
            df = df[df["approved"].fillna(0).astype(int) == 1]

        for _, r in df.iterrows():
            s_fr = normalize_merge_key_text(r.get("source_fr", ""))
            s_en = normalize_merge_key_text(r.get("source_en", ""))
            t_fr = normalize_merge_key_text(r.get("target_fr", ""))
            t_en = normalize_merge_key_text(r.get("target_en", ""))
            if not s_fr or not s_en or not t_fr or not t_en:
                continue
            src = _pair_key(s_fr, s_en)
            tgt = _pair_key(t_fr, t_en)
            mapping[src] = tgt
            # Keep target self-mapped for stability.
            mapping.setdefault(tgt, tgt)

        return mapping

    def load_all_languages(self) -> None:
        """Load all configured language files."""
        items = sorted(LANGUAGE_PATHS.items())
        for lang_name, rel_path in tqdm(items, desc="Loading languages", unit="lang"):
            full_path = os.path.join(self.base_path, rel_path)

            try:
                tqdm.write(f"  → Loading {lang_name}")
                phrases = extract_text_from_docx(full_path)
                start_idx = find_starting_index(phrases)
                self.language_data[lang_name] = phrases[start_idx:]
                self.language_names.append(lang_name)
            except Exception as e:
                tqdm.write(f"  ⚠ Failed to load {lang_name}: {e}")

    def replace_text_in_language_docs_inplace(
        self,
        replacements: Mapping[str, str],
        language_names: Optional[List[str]] = None,
        visible: bool = False,
    ) -> Dict[str, bool]:
        """
        Apply in-place text replacements to the source Word documents.

        Args:
            replacements: Mapping like {"replace_x": "by_y"}
            language_names: Optional subset of LANGUAGE_PATHS keys to process
            visible: Whether to show Microsoft Word while processing

        Returns:
            Mapping of absolute document path -> whether replacements occurred
        """
        selected_languages = language_names or list(LANGUAGE_PATHS.keys())
        document_paths = []

        for lang_name in selected_languages:
            rel_path = LANGUAGE_PATHS.get(lang_name)
            if not rel_path:
                print(f"Warning: Unknown language name: {lang_name}")
                continue

            full_path = os.path.join(self.base_path, rel_path)
            if not os.path.exists(full_path):
                print(f"Warning: File not found for {lang_name}: {full_path}")
                continue
            document_paths.append(full_path)

        document_paths.sort()

        return replace_text_in_word_documents_inplace(
            document_paths=document_paths,
            replacements=replacements,
            visible=visible,
        )

    def process_all_languages(self) -> None:
        """Process all loaded language data into DataFrames."""
        for lang_name in tqdm(self.language_names, desc="Processing languages", unit="lang"):
            tqdm.write(f"  → Processing {lang_name}")

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

            # Ensure space before '(' when directly attached: akye(Maakye → akye (Maakye, toi.(Sers → toi. (Sers
            df = df.applymap(lambda x: re.sub(r'(?<=\S)\(', ' (', str(x)) if isinstance(x, str) else x)

            # Strip whitespace
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # Preserve original row order inside each source language file.
            df = df.reset_index(drop=True)
            df["LOCAL_ID"] = np.arange(1, len(df) + 1, dtype=int)

            self.dataframes[lang_name] = df

    def invalidate_merge_cache(self) -> None:
        """Clear the cached merged DataFrame so the next call recomputes it."""
        self._merged_cache = None

    def _prepare_merge_rows(self) -> Dict[str, pd.DataFrame]:
        """Prepare per-language rows with both raw and normalized merge keys."""
        self.equivalence_map = self.load_equivalence_map(EQUIVALENCE_MAP_CSV)
        prepared: Dict[str, pd.DataFrame] = {}

        for lang_name in tqdm(self.language_names, desc="Preparing merge keys", unit="lang"):
            subset = self.dataframes[lang_name][[lang_name, "Francais", "Anglais", "LOCAL_ID"]].copy()
            subset["Francais"] = subset["Francais"].astype(str).str.strip()
            subset["Anglais"] = subset["Anglais"].astype(str).str.strip()
            subset["_raw_fr"] = subset["Francais"]
            subset["_raw_en"] = subset["Anglais"]
            canonical_pairs = subset.apply(
                lambda r: _canonical_merge_pair(
                    r["Francais"],
                    r["Anglais"],
                    self.equivalence_map,
                ),
                axis=1,
            )
            subset["_merge_fr"] = canonical_pairs.map(lambda x: x[0])
            subset["_merge_en"] = canonical_pairs.map(lambda x: x[1])
            subset = subset[
                (subset["_merge_fr"].str.len() > 0) &
                (subset["_merge_en"].str.len() > 0)
            ].copy()
            subset = subset.reset_index(drop=True)
            subset["_row_id"] = subset.index.astype(int)
            prepared[lang_name] = subset

        return prepared

    def _build_candidate_queues(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Dict[Tuple[str, str], deque], Dict[Tuple[str, str], deque]]:
        """Build exact and normalized lookup queues preserving source order."""
        exact_map: Dict[Tuple[str, str], deque] = defaultdict(deque)
        normalized_map: Dict[Tuple[str, str], deque] = defaultdict(deque)

        ordered = df.sort_values("LOCAL_ID", kind="mergesort")
        for row_id, raw_fr, raw_en, merge_fr, merge_en in zip(
            ordered["_row_id"],
            ordered["_raw_fr"],
            ordered["_raw_en"],
            ordered["_merge_fr"],
            ordered["_merge_en"],
        ):
            exact_map[_raw_pair_key(raw_fr, raw_en)].append(int(row_id))
            normalized_map[(merge_fr, merge_en)].append(int(row_id))

        return exact_map, normalized_map

    def _build_anchor_buckets(
        self,
        anchor_df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Group anchor rows by normalized key while preserving first occurrence order.

        Repeated Nufi rows often represent the same template slot with different
        sample names/places. These should form one correspondence bucket.
        """
        buckets: List[Dict[str, Any]] = []
        grouped = anchor_df.sort_values("LOCAL_ID", kind="mergesort").groupby(
            ["_merge_fr", "_merge_en"],
            sort=False,
            dropna=False,
        )
        for (merge_fr, merge_en), group in grouped:
            ordered = group.sort_values("LOCAL_ID", kind="mergesort")
            first = ordered.iloc[0]
            buckets.append({
                "_merge_fr": merge_fr,
                "_merge_en": merge_en,
                "_merge_order": first["LOCAL_ID"],
                "Francais": first["_raw_fr"],
                "Anglais": first["_raw_en"],
                "anchor_rows": ordered.to_dict("records"),
            })
        return buckets

    @staticmethod
    def _pop_next_available(queue: deque, used_ids: set) -> Optional[int]:
        """Pop the next unused row id from a queue."""
        while queue:
            row_id = queue.popleft()
            if row_id not in used_ids:
                return row_id
        return None

    def _merge_remaining_rows(
        self,
        remaining_by_language: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Merge rows that could not be aligned to the anchor language.

        Stage 1 groups by exact FR/EN pair. Stage 2 groups the leftover rows
        by normalized FR/EN pair. Duplicates are preserved via occurrence order.
        """
        merged_rows: List[Dict[str, Any]] = []

        def build_rows(group_attr: str, source_rows: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
            grouped: Dict[Any, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
            for lang_name, df in source_rows.items():
                if df.empty:
                    continue
                ordered = df.sort_values("LOCAL_ID", kind="mergesort")
                for row in ordered.to_dict("records"):
                    grouped[row[group_attr]][lang_name].append(row)

            rows: List[Dict[str, Any]] = []
            for _, lang_groups in grouped.items():
                occurrences = max(len(items) for items in lang_groups.values())
                for idx in range(occurrences):
                    row_data: Dict[str, Any] = {lang: np.nan for lang in self.language_names}
                    fr_value = None
                    en_value = None
                    merge_order = np.nan
                    for lang_name in self.language_names:
                        items = lang_groups.get(lang_name, [])
                        if idx >= len(items):
                            continue
                        item = items[idx]
                        row_data[lang_name] = item[lang_name]
                        if fr_value is None:
                            fr_value = item["_raw_fr"]
                        if en_value is None:
                            en_value = item["_raw_en"]
                        if pd.isna(merge_order):
                            merge_order = item["LOCAL_ID"]
                    row_data["Francais"] = fr_value
                    row_data["Anglais"] = en_value
                    row_data["_merge_order"] = merge_order
                    rows.append(row_data)
            return rows

        exact_rows = build_rows("_exact_key", remaining_by_language)
        if exact_rows:
            merged_rows.extend(exact_rows)

        consumed_ids: Dict[str, set] = {lang: set() for lang in remaining_by_language}
        for row in exact_rows:
            for lang_name in self.language_names:
                if lang_name not in remaining_by_language:
                    continue
                value = row.get(lang_name)
                if pd.isna(value):
                    continue
                matches = remaining_by_language[lang_name][remaining_by_language[lang_name][lang_name] == value]
                for row_id in matches["_row_id"].tolist():
                    if row_id not in consumed_ids[lang_name]:
                        consumed_ids[lang_name].add(int(row_id))
                        break

        normalized_remaining: Dict[str, pd.DataFrame] = {}
        for lang_name, df in remaining_by_language.items():
            if not consumed_ids[lang_name]:
                normalized_remaining[lang_name] = df.copy()
            else:
                normalized_remaining[lang_name] = df[~df["_row_id"].isin(consumed_ids[lang_name])].copy()
            normalized_remaining[lang_name]["_norm_key"] = list(
                zip(normalized_remaining[lang_name]["_merge_fr"], normalized_remaining[lang_name]["_merge_en"])
            )

        merged_rows.extend(build_rows("_norm_key", normalized_remaining))
        return merged_rows

    def merge_all_dataframes(self) -> pd.DataFrame:
        """
        Build correspondence rows anchored on Nufi when available.

        For each Nufi row, try to find the corresponding row in every other
        language by:
        1. exact FR/EN text match
        2. normalized FR/EN match

        Remaining unmatched rows are then grouped in two passes:
        1. exact FR/EN pair
        2. normalized FR/EN pair

        Results are cached; call invalidate_merge_cache() to force recomputation.

        Returns:
            Merged DataFrame containing all languages
        """
        if self._merged_cache is not None:
            return self._merged_cache

        prepared = self._prepare_merge_rows()
        if not prepared:
            return pd.DataFrame()

        anchor_lang = "Nufi" if "Nufi" in prepared else next(iter(prepared))
        anchor_df = prepared[anchor_lang].sort_values("LOCAL_ID", kind="mergesort").copy()
        anchor_buckets = self._build_anchor_buckets(anchor_df)
        matched_ids: Dict[str, set] = {lang: set() for lang in prepared}
        merged_rows: List[Dict[str, Any]] = []

        candidate_maps: Dict[str, Tuple[Dict[Tuple[str, str], deque], Dict[Tuple[str, str], deque]]] = {}
        for lang_name, df in prepared.items():
            if lang_name == anchor_lang:
                continue
            candidate_maps[lang_name] = self._build_candidate_queues(df)

        for bucket in tqdm(anchor_buckets, total=len(anchor_buckets), desc="Matching anchor buckets", unit="bucket"):
            row_data: Dict[str, Any] = {lang: np.nan for lang in self.language_names}
            first_anchor = bucket["anchor_rows"][0]
            row_data[anchor_lang] = first_anchor[anchor_lang]
            row_data["Francais"] = bucket["Francais"]
            row_data["Anglais"] = bucket["Anglais"]
            row_data["_merge_order"] = bucket["_merge_order"]
            for anchor_row in bucket["anchor_rows"]:
                matched_ids[anchor_lang].add(int(anchor_row["_row_id"]))

            exact_keys = [
                _raw_pair_key(anchor_row["_raw_fr"], anchor_row["_raw_en"])
                for anchor_row in bucket["anchor_rows"]
            ]
            normalized_key = (bucket["_merge_fr"], bucket["_merge_en"])

            for lang_name in self.language_names:
                if lang_name == anchor_lang or lang_name not in prepared:
                    continue
                exact_map, normalized_map = candidate_maps[lang_name]
                matched_row_id = None
                for exact_key in exact_keys:
                    matched_row_id = self._pop_next_available(exact_map[exact_key], matched_ids[lang_name])
                    if matched_row_id is not None:
                        break
                if matched_row_id is None:
                    matched_row_id = self._pop_next_available(normalized_map[normalized_key], matched_ids[lang_name])
                if matched_row_id is None:
                    continue

                matched_ids[lang_name].add(matched_row_id)
                matched_row = prepared[lang_name].loc[prepared[lang_name]["_row_id"] == matched_row_id].iloc[0]
                row_data[lang_name] = matched_row[lang_name]

                # For template-style buckets, consume remaining rows with the
                # same normalized key in that language so they do not reappear
                # as unrelated leftover rows.
                bucket_is_template = (
                    bucket["_merge_fr"].endswith("example") or
                    bucket["_merge_en"].endswith("example") or
                    bucket["_merge_fr"] in {"i live in x place", "i am from x place", "he or she is my friend"} or
                    bucket["_merge_en"] in {"i live in x place", "i am from x place", "he or she is my friend"}
                )
                if bucket_is_template:
                    same_key_rows = prepared[lang_name][
                        (prepared[lang_name]["_merge_fr"] == normalized_key[0]) &
                        (prepared[lang_name]["_merge_en"] == normalized_key[1])
                    ]
                    matched_ids[lang_name].update(int(row_id) for row_id in same_key_rows["_row_id"].tolist())

            merged_rows.append(row_data)

        remaining_by_language: Dict[str, pd.DataFrame] = {}
        for lang_name, df in prepared.items():
            remaining = df[~df["_row_id"].isin(matched_ids[lang_name])].copy()
            if remaining.empty:
                continue
            remaining["_exact_key"] = list(zip(remaining["_raw_fr"], remaining["_raw_en"]))
            remaining_by_language[lang_name] = remaining

        if remaining_by_language:
            merged_rows.extend(self._merge_remaining_rows(remaining_by_language))

        merged_df = pd.DataFrame(merged_rows)
        if merged_df.empty:
            merged_df = pd.DataFrame(columns=["GLOBAL_ID", "Francais", "Anglais"] + self.language_names)
        else:
            merged_df = merged_df.sort_values(
                by=["_merge_order", "Francais", "Anglais"],
                kind="mergesort",
                na_position="last",
            ).reset_index(drop=True)
            merged_df["GLOBAL_ID"] = np.arange(1, len(merged_df) + 1, dtype=int)
            merged_df = merged_df[["GLOBAL_ID", "Francais", "Anglais"] + self.language_names]

        self._merged_cache = merged_df
        return merged_df

    def export_equivalence_suggestions(
        self,
        output_path: str = EQUIVALENCE_SUGGESTIONS_CSV
    ) -> None:
        """
        Suggest potential equivalence mappings for rows missing exactly one language.
        """
        merged_df = self.merge_all_dataframes()
        language_cols = [lang for lang in self.language_names if lang in merged_df.columns]
        if not language_cols:
            pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Exported equivalence suggestions to {output_path} (no language columns found)")
            return

        merged_df["missing_count"] = merged_df[language_cols].isna().sum(axis=1)
        merged_df["missing_languages"] = merged_df.apply(
            lambda r: [lang for lang in language_cols if pd.isna(r[lang])],
            axis=1
        )

        # Restrict to clear cases to avoid noisy suggestions.
        candidates = merged_df[merged_df["missing_count"] == 1].copy()
        out_rows = []

        for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Finding equivalences", unit="row"):
            miss_lang = row["missing_languages"][0]
            lang_df = self.dataframes.get(miss_lang)
            if lang_df is None or lang_df.empty:
                continue

            src_fr = str(row["Francais"])
            src_en = str(row["Anglais"])
            src_kfr, src_ken = _canonical_merge_pair(src_fr, src_en, self.equivalence_map)

            lang_tmp = lang_df[["Francais", "Anglais"]].copy()
            lang_pairs = lang_tmp.apply(
                lambda r: _canonical_merge_pair(
                    r["Francais"],
                    r["Anglais"],
                    self.equivalence_map,
                ),
                axis=1
            )
            lang_tmp["_kfr"] = lang_pairs.map(lambda x: x[0])
            lang_tmp["_ken"] = lang_pairs.map(lambda x: x[1])

            fr_hits = lang_tmp[(lang_tmp["_kfr"] == src_kfr) & (lang_tmp["_ken"] != src_ken)]
            en_hits = lang_tmp[(lang_tmp["_ken"] == src_ken) & (lang_tmp["_kfr"] != src_kfr)]

            # Highest-confidence deterministic suggestions first.
            if not fr_hits.empty:
                hit = fr_hits.iloc[0]
                out_rows.append({
                    "global_id": int(row["GLOBAL_ID"]),
                    "missing_language": miss_lang,
                    "reason": "same_french_different_english",
                    "source_fr": src_fr,
                    "source_en": src_en,
                    "candidate_fr": hit["Francais"],
                    "candidate_en": hit["Anglais"],
                    "source_key_fr": src_kfr,
                    "source_key_en": src_ken,
                    "candidate_key_fr": hit["_kfr"],
                    "candidate_key_en": hit["_ken"],
                    "fr_similarity": 1.0,
                    "en_similarity": _similarity(src_ken, hit["_ken"]),
                })
                continue

            if not en_hits.empty:
                hit = en_hits.iloc[0]
                out_rows.append({
                    "global_id": int(row["GLOBAL_ID"]),
                    "missing_language": miss_lang,
                    "reason": "same_english_different_french",
                    "source_fr": src_fr,
                    "source_en": src_en,
                    "candidate_fr": hit["Francais"],
                    "candidate_en": hit["Anglais"],
                    "source_key_fr": src_kfr,
                    "source_key_en": src_ken,
                    "candidate_key_fr": hit["_kfr"],
                    "candidate_key_en": hit["_ken"],
                    "fr_similarity": _similarity(src_kfr, hit["_kfr"]),
                    "en_similarity": 1.0,
                })
                continue

            # Soft suggestion: nearest candidate by combined FR+EN similarity.
            lang_tmp["_fr_sim"] = lang_tmp["_kfr"].map(lambda t: _similarity(src_kfr, t))
            lang_tmp["_en_sim"] = lang_tmp["_ken"].map(lambda t: _similarity(src_ken, t))
            lang_tmp["_score"] = (lang_tmp["_fr_sim"] + lang_tmp["_en_sim"]) / 2.0
            best = lang_tmp.sort_values("_score", ascending=False).head(1)
            if not best.empty:
                b = best.iloc[0]
                if float(b["_score"]) >= 0.90:
                    out_rows.append({
                        "global_id": int(row["GLOBAL_ID"]),
                        "missing_language": miss_lang,
                        "reason": "high_similarity_candidate",
                        "source_fr": src_fr,
                        "source_en": src_en,
                        "candidate_fr": b["Francais"],
                        "candidate_en": b["Anglais"],
                        "source_key_fr": src_kfr,
                        "source_key_en": src_ken,
                        "candidate_key_fr": b["_kfr"],
                        "candidate_key_en": b["_ken"],
                        "fr_similarity": float(b["_fr_sim"]),
                        "en_similarity": float(b["_en_sim"]),
                    })

        pd.DataFrame(out_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Exported equivalence suggestions to {output_path} ({len(out_rows)} suggestion(s))")

    def export_merge_diagnostics(
        self,
        output_path: str = "merge_problematic_lines.csv"
    ) -> None:
        """
        Export rows with missing translations after merge for easier auditing.
        """
        merged_df = self.merge_all_dataframes()
        language_cols = [lang for lang in self.language_names if lang in merged_df.columns]
        if not language_cols:
            pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Exported merge diagnostics to {output_path} (no language columns found)")
            return

        merged_df["missing_count"] = merged_df[language_cols].isna().sum(axis=1)
        merged_df["missing_languages"] = merged_df.apply(
            lambda r: ", ".join([lang for lang in language_cols if pd.isna(r[lang])]),
            axis=1
        )
        merged_df["available_languages"] = merged_df.apply(
            lambda r: ", ".join([lang for lang in language_cols if not pd.isna(r[lang])]),
            axis=1
        )
        diagnostics = merged_df[merged_df["missing_count"] > 0].copy()
        diagnostics = diagnostics.sort_values("missing_count", ascending=False)
        diagnostics.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Exported merge diagnostics to {output_path}")

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
            merged = self.merge_all_dataframes()
            key_to_gid = {
                _canonical_merge_pair(fr, en, self.equivalence_map): int(gid)
                for gid, fr, en in zip(merged["GLOBAL_ID"], merged["Francais"], merged["Anglais"])
            }

            for lang_name in tqdm(self.language_names, desc="Writing Excel sheets", unit="lang"):
                subset_cols = ["Francais", "Anglais", lang_name]
                df_subset = self.dataframes[lang_name][subset_cols].copy()
                canonical_pairs = df_subset.apply(
                    lambda r: _canonical_merge_pair(
                        r["Francais"],
                        r["Anglais"],
                        self.equivalence_map,
                    ),
                    axis=1
                )
                df_subset["_merge_fr"] = canonical_pairs.map(lambda x: x[0])
                df_subset["_merge_en"] = canonical_pairs.map(lambda x: x[1])
                df_subset["GLOBAL_ID"] = df_subset.apply(
                    lambda r: key_to_gid.get((r["_merge_fr"], r["_merge_en"])),
                    axis=1
                )
                df_subset["GLOBAL_ID"] = df_subset["GLOBAL_ID"].astype("Int64")
                df_subset = df_subset.sort_values("GLOBAL_ID", na_position="last", kind="mergesort")
                df_subset = df_subset[["GLOBAL_ID", "Francais", "Anglais", lang_name]]

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
            for lang_name in tqdm(self.language_names, desc="Checking capitalization", unit="lang"):
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


def build_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for running selected pipeline stages."""
    parser = argparse.ArgumentParser(description="African Language Phrasebook Processor")
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help="Base folder containing the source language documents.",
    )
    parser.add_argument(
        "--skip-doc-replacements",
        action="store_true",
        help="Skip in-place Word document replacements before loading data.",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel export.",
    )
    parser.add_argument(
        "--skip-quality-checks",
        action="store_true",
        help="Skip lowercase issues, proper-name detection, merge diagnostics, and equivalence suggestions.",
    )
    parser.add_argument(
        "--skip-merge-csv",
        action="store_true",
        help="Skip merged CSV export.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only export CSV outputs and skip document replacement, Excel export, and other quality-check outputs.",
    )
    return parser


def main(argv: Optional[List[str]] = None):
    """Main execution function."""
    args = build_cli_parser().parse_args(argv)

    print("African Language Phrasebook Processor")
    print("=" * 50)

    # Initialize processor
    processor = PhrasebookProcessor(base_path=args.base_path)

    # Dictionary Replacement
    dict_replace = {
        "Je suis étudiant informatique.": "Je suis étudiant en informatique.",
        "My father doesn't like car": "My father doesn't like cars",
        "Je ne sais pas moi.": "Je ne sais pas, moi.",
        "Whom with? With whom?": "With whom?",
        "Yes, I already took my bath.": "Yes, I have already taken a shower.",
        "Yes, I have already taken my bath.": "Yes, I have already taken a shower.",
        "Ils sont nombreux. | There are many.": "Ils sont nombreux. | They are many.",
        "My boss is so strict.": "My boss is very strict.",
        "Je suis un comptable.": "Je suis comptable.",
        "My friend is Hairdresser.": "My friend is a hairdresser.",
        "Bonne fête d'Independence.": "Bonne fête d'Indépendance.",
        "Enlevez vos chaussures, s'il vous plaît": "Enlevez vos chaussures, s'il vous plaît!",
        "I wish God to take my life.": "I wish God would take my life.",
        "Ask for forgiveness.": "Ask for forgiveness. Apologizing",
        "J'ai 1, 65 m.": "Je fais 1, 65 m.",
        "Could I take your phone number": "Could I have your phone number",
        "Quelle heure il est": "Quelle heure est-il",
        "What's the temperature?": "What is the temperature?",
        "It did not rain (more than 6 hours ago and less than a day).": "It did not rain (more than 6 hours ago but less than a day).",
        "It rained (more than 6 hours ago and less than a day).": "It rained (more than 6 hours ago but less than a day).",
        "It is nine forty-five (9:45 am).": "It is nine forty-five a.m. (9:45 AM).",
        "It is scandalous/disgraceful/outrageous!": "This is scandalous/disgraceful/outrageous!",
        "Thanks, we spent a great evening/i had a great evening": "Thanks, we spent a great evening; I had a great evening",
        "That works for me/it's suitable for me. Sounds good": "That works for me;it's suitable for me. Sounds good",
        "Si c'est vrai, je t'achèterais à boire; sinon, je te frapperais.": "Si c'est vrai, je t'achèterai à boire; sinon, je te frapperai.",
        "Il a commencé à pleuvoir. | It started raining.": "Il a commencé à pleuvoir. | It has started raining.",
        "I got married with a Dustman/garbage man/garbage collector.": "I got married to a Dustman/garbage man/garbage collector.",
        "Whom with? With whom? (When it is the 2nd time the question has been asked)": "With whom? (When it is the 2nd time the question is being asked.)",
    }

    run_doc_replacements = not (args.skip_doc_replacements or args.csv_only)
    run_excel_export = not (args.skip_excel or args.csv_only)
    run_merge_csv = not args.skip_merge_csv
    run_quality_checks = not (args.skip_quality_checks or args.csv_only)

    if run_doc_replacements:
        results = processor.replace_text_in_language_docs_inplace(
            replacements=dict_replace,
            # language_names=["Basaa", "DualaDouala", "Ewondo"], # All Languages by default
        )
        print(results)
    else:
        print("Skipping in-place document replacements")


    processor.ensure_equivalence_map_template(EQUIVALENCE_MAP_CSV)

    # Load all language files
    print("\nLoading language files...")
    processor.load_all_languages()
    print(f"Loaded {len(processor.language_names)} languages")

    # Process all languages
    print("\nProcessing languages...")
    processor.process_all_languages()

    # Export results
    print("\nExporting results...")
    if run_excel_export:
        processor.export_to_excel("African_Languages_dataframes.xlsx")
    else:
        print("Skipping Excel export")
    if run_merge_csv:
        processor.export_merged_to_csv("African_Languages_dataframes_merged.csv")
    else:
        print("Skipping merged CSV export")

    # Quality checks
    if run_quality_checks:
        print("\nRunning quality checks...")
        processor.find_lowercase_issues("not_starting_with_capital.xlsx")

        proper_names, missing = processor.identify_proper_names()
        if not proper_names.empty:
            proper_names.to_csv("data_ProperNames.csv", encoding="utf-8-sig", index=False)
            print(f"Found {len(proper_names)} entries with proper names")

        processor.export_merge_diagnostics("merge_problematic_lines.csv")
        processor.export_equivalence_suggestions(EQUIVALENCE_SUGGESTIONS_CSV)
    else:
        print("\nSkipping quality checks")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
