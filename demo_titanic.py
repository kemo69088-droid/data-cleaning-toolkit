# -*- coding: utf-8 -*-
"""
demo_titanic.py
===============
تطبيق عملي لأداة data_cleaner.py على مجموعة بيانات حقيقية (Titanic).

المشاكل الموجودة في البيانات الخام:
  - 177 قيمة مفقودة في عمود age
  - 688 قيمة مفقودة في عمود deck  (77% من العمود)
  - 107 صف مكرر تماماً
  - أعمدة مكررة المعنى (class / pclass ، alive / survived)

المخرجات:
  titanic_cleaned.csv      الملف النظيف
  cleaning_report.html     تقرير قبل / بعد
  01_missing_before_after.png
  02_cleaning_impact.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_cleaner import DataCleaner

# ---------------------------------------------------------------- الألوان
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BEFORE, AFTER = "#eb6834", "#2a78d6"          # برتقالي = قبل ، أزرق = بعد

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})

# البيانات تُقرأ مباشرة من المصدر العام، فلا حاجة لتحميلها يدوياً
SRC = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------- 1) تشخيص البيانات الخام
dc = DataCleaner(SRC, missing_threshold=0.60)
raw_missing = dc.df.isna().sum()

print("=" * 62)
print("تشخيص البيانات الخام")
print("=" * 62)
print(dc.profile().to_string(index=False))
print("\nملخص:", dc.summary())

# ------------------------------------------------------ 2) تنفيذ التنظيف
dc.clean()

print("\n" + "=" * 62)
print("الخطوات المنفّذة")
print("=" * 62)
for i, step in enumerate(dc.log, 1):
    print(f"{i}. {step}")

print("\nقبل :", dc.before)
print("بعد :", dc.after)
if not dc.outliers.empty:
    print("\nالقيم الشاذة المرصودة (لم تُحذف):")
    print(dc.outliers.to_string(index=False))

dc.save(f"{OUT}/titanic_cleaned.csv")
dc.report(f"{OUT}/cleaning_report.html")

# --------------------------------- 3) رسم: القيم المفقودة قبل وبعد التنظيف
cols = raw_missing[raw_missing > 0].sort_values(ascending=True)
after_missing = [int(dc.df[c].isna().sum()) if c in dc.df.columns else 0 for c in cols.index]

fig, ax = plt.subplots(figsize=(9, 3.8))
y = range(len(cols))
h = 0.38
ax.barh([i + h / 2 for i in y], cols.values, height=h,
        color=BEFORE, label="Before cleaning", zorder=3)
ax.barh([i - h / 2 for i in y], after_missing, height=h,
        color=AFTER, label="After cleaning", zorder=3)
# القيم الصفرية لا ترسم عموداً، فنضع علامة عندها حتى تظهر السلسلة
zero_y = [i - h / 2 for i, v in zip(y, after_missing) if v == 0]
ax.scatter([0] * len(zero_y), zero_y, s=64, color=AFTER, zorder=4)

for i, v in enumerate(cols.values):
    ax.text(v + 8, i + h / 2, f"{v}", va="center", color=INK2, fontsize=10)
for i, v in enumerate(after_missing):
    ax.text(v + 12, i - h / 2, f"{v}", va="center", color=INK2, fontsize=10)

ax.set_yticks(list(y), list(cols.index))
ax.set_xlabel("Missing values (count)")
ax.set_title("Missing values before vs. after cleaning",
             fontsize=14, fontweight="bold", color=INK, pad=14, loc="left")
ax.set_xlim(0, max(cols.values) * 1.16)
ax.grid(axis="x", zorder=0)
ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
fig.tight_layout()
fig.savefig(f"{OUT}/01_missing_before_after.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ------------------------------------- 4) رسم: أثر التنظيف على حجم البيانات
labels = ["Rows", "Columns", "Empty cells", "Duplicate rows"]
before_v = [dc.before["rows"], dc.before["columns"],
            dc.before["missing_cells"], dc.before["duplicate_rows"]]
after_v = [dc.after["rows"], dc.after["columns"],
           dc.after["missing_cells"], dc.after["duplicate_rows"]]

fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
for ax, lab, b, a in zip(axes, labels, before_v, after_v):
    ax.bar([0], [b], width=0.5, color=BEFORE, zorder=3)
    ax.bar([1], [a], width=0.5, color=AFTER, zorder=3)
    ax.text(0, b, f"{b:,}", ha="center", va="bottom", color=INK2, fontsize=10)
    ax.text(1, a, f"{a:,}", ha="center", va="bottom", color=INK2, fontsize=10)
    ax.set_xticks([0, 1], ["Before", "After"])
    ax.set_title(lab, fontsize=11, color=INK2, pad=10)
    ax.set_ylim(0, max(b, a) * 1.28 if max(b, a) else 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
fig.suptitle("Cleaning impact — Titanic dataset",
             fontsize=14, fontweight="bold", color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(f"{OUT}/02_cleaning_impact.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

print(f"\nتم إنشاء الملفات في مجلد {OUT}/")
