import glob
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET


DOCX_PATH = ""
DOCX_FILENAME = "Ewondo_for_Reading_Phrasebook_ExpressionsUsuelles_GuideConversation.docx"
DOCX_SEARCH_ROOT = r"G:\My Drive"
LIST_PATH = r"C:\Users\tcham\Wokspace\Temp\missing_files_summary_to_be_restarted.txt"
OUT_PATH = r"C:\Users\tcham\Wokspace\Temp\missing_files_sentences.txt"


def read_docx_paragraphs(path):
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
        if texts:
            paragraphs.append("".join(texts))
    return paragraphs


def build_number_map(paragraphs):
    num_map = {}
    pattern = re.compile(r"^\s*(\d+)\s*[\.)-]?\s*(.+)$")
    for text in paragraphs:
        s = text.strip()
        m = pattern.match(s)
        if not m:
            continue
        num = int(m.group(1))
        if num not in num_map:
            num_map[num] = s
    return num_map


def resolve_docx_path():
    if DOCX_PATH and os.path.isfile(DOCX_PATH):
        return DOCX_PATH
    pattern = os.path.join(DOCX_SEARCH_ROOT, "*", "Livres Ewondo", DOCX_FILENAME)
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    for root, _, files in os.walk(DOCX_SEARCH_ROOT):
        if DOCX_FILENAME in files:
            return os.path.join(root, DOCX_FILENAME)
    return None


def read_number_requests(path):
    requests = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, nums_text = line.split(":", 1)
            nums = [int(n) for n in re.findall(r"\d+", nums_text)]
            if nums:
                requests.append((label.strip(), nums))
    return requests


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    docx_path = resolve_docx_path()
    if not docx_path:
        print("Could not find the docx file. Update DOCX_PATH or DOCX_SEARCH_ROOT.")
        raise SystemExit(1)

    paragraphs = read_docx_paragraphs(docx_path)
    num_map = build_number_map(paragraphs)
    requests = read_number_requests(LIST_PATH)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for label, nums in requests:
            out.write(f"{label}\n")
            missing = []
            for n in nums:
                sentence = num_map.get(n)
                if sentence:
                    out.write(f"{sentence}\n")
                else:
                    missing.append(n)
            if missing:
                out.write("Missing in docx: " + ", ".join(map(str, missing)) + "\n")
            out.write("\n")

    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
