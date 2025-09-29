import pandas as pd
from pathlib import Path

# === Settings ===
fp = Path(r"G:\My Drive\Mbú'ŋwɑ̀'nì\Livres Nufi\African_Languages_Phrasebook_GuideConversationExpressionsUsuelles_TabularPrototype_bon.xlsx")
language = "Bamoun"
language = "Wolof"

# === Load data ===
df = pd.read_excel(fp, sheet_name=language, engine="openpyxl")

# === Combine columns ===
df["Combined"] = (
    df["ID"].astype(str) + ") "
    + df["Anglais"].fillna("") + " | "
    + df[language].fillna("") + " | "
    + df["Francais"].fillna("")
)

# === Export as plain text (no CSV quoting) ===
out_file = f"{language.lower()}_english_french_phrasebook_sentences_list.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(df["Combined"].astype(str)))

print(f"✅ Saved to {out_file}")
