import pandas as pd
from config import REPORTS_DIR

d = pd.read_csv(REPORTS_DIR / "gradcam_review" / "gradcam_review.csv",
                sep=None, engine="python", encoding="utf-8-sig")
d = d.dropna(subset=["olhou_lesao"])
d["olhou_lesao"] = d["olhou_lesao"].str.strip().str.lower()

print("Geral:")
print(d["olhou_lesao"].value_counts(normalize=True).round(2).to_string())
print("\nAcertos x erros:")
print(pd.crosstab(d["correto"], d["olhou_lesao"], normalize="index").round(2).to_string())
print("\nPor classe real:")
print(pd.crosstab(d["real"], d["olhou_lesao"], normalize="index").round(2).to_string())