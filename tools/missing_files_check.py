import csv
import os
import re
import sys

# Specify the folder path to inspect
fpath = r"G:\My Drive\Mbú'ŋwɑ̀'nì\Livres Ewondo\Audio Phrasebook Ewondo\ewondo_audio_processed_files\Ewondo New"

# Avoid Windows console encoding issues with non-ASCII paths
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Extensions to consider when scanning
extensions = {".mp3", ".wav"}

def analyze_folder(folder_path):
    try:
        files = os.listdir(folder_path)
    except PermissionError:
        return {
            "folder": folder_path,
            "status": "permission_denied",
        }

    numbered_exact = []
    numbered_with_suffix = []
    bad_suffix_files = []
    mp3_exact = set()
    mp3_underscore = set()
    non_numeric = []
    ignored = []

    for f in files:
        base, ext = os.path.splitext(f)
        base = base.rstrip()
        if ext.lower() not in extensions:
            ignored.append(f)
            continue
        if re.fullmatch(r"\d+", base):
            n = int(base)
            numbered_exact.append(n)
            if ext.lower() == ".mp3":
                mp3_exact.add(n)
        else:
            m = re.match(r"^(\d+)(.*)$", base)
            if m:
                n = int(m.group(1))
                numbered_with_suffix.append((n, f))
                if ext.lower() == ".mp3":
                    if m.group(2) == "_":
                        mp3_underscore.add(n)
                    else:
                        bad_suffix_files.append(f)
            else:
                non_numeric.append(f)

    all_numbers = set(numbered_exact) | {n for n, _ in numbered_with_suffix}
    if all_numbers:
        min_num = min(all_numbers)
        max_num = max(all_numbers)
        all_nums = set(range(min_num, max_num + 1))
        missing_nums = sorted(all_nums - all_numbers)
        missing_mp3_exact = sorted(all_nums - mp3_exact)
        missing_mp3_underscore = sorted(all_nums - mp3_underscore)
    else:
        min_num = 0
        max_num = 0
        missing_nums = []
        missing_mp3_exact = []
        missing_mp3_underscore = []

    return {
        "folder": folder_path,
        "folder_display": os.path.basename(folder_path),
        "status": "ok",
        "min_num": min_num,
        "max_num": max_num,
        "missing_nums": missing_nums,
        "missing_n_mp3": missing_mp3_exact,
        "missing_n__mp3": missing_mp3_underscore,
        "exact_numeric_count": len(numbered_exact),
        "numeric_prefix_with_suffix_count": len(numbered_with_suffix),
        "non_numeric_count": len(non_numeric),
        "ignored_count": len(ignored),
        "suffix_examples": [f for _, f in numbered_with_suffix[:10]],
        "non_numeric_examples": non_numeric[:10],
        "bad_suffix_files": bad_suffix_files,
        "non_numeric_files": non_numeric,
    }


# Check if the directory exists
if not os.path.isdir(fpath):
    print(f"Directory '{fpath}' does not exist.")
else:
    report_rows = []
    for root, _, files in os.walk(fpath):
        if "Header" in os.path.basename(root):
            continue
        # Only analyze folders that have audio files
        if not any(os.path.splitext(f)[1].lower() in extensions for f in files):
            continue
        result = analyze_folder(root)
        report_rows.append(result)

    def natural_key(text):
        parts = re.split(r"(\d+)", text)
        key = []
        for part in parts:
            if part.isdigit():
                key.append(int(part))
            else:
                key.append(part.lower())
        return key

    report_rows.sort(key=lambda r: natural_key(r.get("folder_display", "")))

    report_path = os.path.join(os.getcwd(), "missing_files_report.csv")
    problems_path = os.path.join(os.getcwd(), "missing_files_problems.txt")
    summary_path = os.path.join(os.getcwd(), "missing_files_summary.txt")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "folder",
                "status",
                "min_num",
                "max_num",
                "missing_nums",
                "missing_n_mp3",
                "missing_n__mp3",
                "exact_numeric_count",
                "numeric_prefix_with_suffix_count",
                "non_numeric_count",
                "ignored_count",
                "suffix_examples",
                "non_numeric_examples",
            ]
        )
        for r in report_rows:
            writer.writerow(
                [
                    r.get("folder_display", r["folder"]),
                    r["status"],
                    r.get("min_num", ""),
                    r.get("max_num", ""),
                    " ".join(map(str, r.get("missing_nums", []))),
                    " ".join(map(str, r.get("missing_n_mp3", []))),
                    " ".join(map(str, r.get("missing_n__mp3", []))),
                    r.get("exact_numeric_count", ""),
                    r.get("numeric_prefix_with_suffix_count", ""),
                    r.get("non_numeric_count", ""),
                    r.get("ignored_count", ""),
                    "; ".join(r.get("suffix_examples", [])),
                    "; ".join(r.get("non_numeric_examples", [])),
                ]
            )

    with open(problems_path, "w", encoding="utf-8") as f:
        for r in report_rows:
            missing_n = r.get("missing_n_mp3", [])
            missing_n_ = r.get("missing_n__mp3", [])
            audios_to_read_again = sorted(set(missing_n) | set(missing_n_))
            bad_files = r.get("bad_suffix_files", []) + r.get("non_numeric_files", [])
            if not (missing_n or missing_n_ or bad_files):
                continue
            f.write(f"{r.get('folder_display', r['folder'])}\n")
            if missing_n:
                f.write(f"Missing n.mp3: {', '.join(map(str, missing_n))}\n")
            if missing_n_:
                f.write(f"Missing n_.mp3: {', '.join(map(str, missing_n_))}\n")
            if audios_to_read_again:
                f.write(
                    "Audios_to_read_again: "
                    + ", ".join(map(str, audios_to_read_again))
                    + "\n"
                )
            for name in bad_files:
                f.write(f"- {name}\n")
            f.write("\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        for r in report_rows:
            folder = r.get("folder_display", r["folder"])
            missing_n = r.get("missing_n_mp3", [])
            missing_n_ = r.get("missing_n__mp3", [])
            audios_to_read_again = sorted(set(missing_n) | set(missing_n_))
            if not audios_to_read_again:
                continue
            f.write(f"{folder}: {', '.join(map(str, audios_to_read_again))}\n")

    print(f"Report written to: {report_path}")
    print(f"Problems written to: {problems_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Folders analyzed: {len(report_rows)}")
