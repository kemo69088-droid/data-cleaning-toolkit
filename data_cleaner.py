# -*- coding: utf-8 -*-
"""
data_cleaner.py
===============
أداة قابلة لإعادة الاستخدام لتنظيف وتجهيز أي ملف CSV للتحليل.

A reusable toolkit that profiles a raw CSV, cleans it with an auditable
set of rules, and produces a before/after report.

Author: Kareem Ayman
Usage:
    from data_cleaner import DataCleaner

    dc = DataCleaner("raw.csv")
    dc.profile()                 # تقرير عن حالة البيانات الخام
    dc.clean()                   # تنفيذ خطوات التنظيف
    dc.save("clean.csv")         # حفظ الملف النظيف
    dc.report("report.html")     # تقرير قبل/بعد
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


class DataCleaner:
    """ينظّف أي DataFrame بخطوات موثّقة ويسجّل كل تعديل في سجل التغييرات."""

    def __init__(self, source, missing_threshold: float = 0.60):
        """
        source            : مسار ملف CSV أو DataFrame جاهز
        missing_threshold : نسبة القيم المفقودة التي بعدها يُحذف العمود بالكامل
        """
        self.raw = pd.read_csv(source) if isinstance(source, str) else source.copy()
        self.df = self.raw.copy()
        self.missing_threshold = missing_threshold
        self.log: list[str] = []

    # ------------------------------------------------------------------
    # 1) التشخيص  —  ما هي المشاكل الموجودة في البيانات؟
    # ------------------------------------------------------------------
    def profile(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """يرجّع جدول تشخيص لكل عمود: النوع، المفقود، التكرار، القيم الفريدة."""
        d = self.df if df is None else df
        rows = []
        for col in d.columns:
            s = d[col]
            n_missing = int(s.isna().sum())
            rows.append({
                "column": col,
                "dtype": str(s.dtype),
                "missing": n_missing,
                "missing_%": round(100 * n_missing / len(d), 2) if len(d) else 0.0,
                "unique": int(s.nunique(dropna=True)),
                "sample": self._sample_value(s),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _sample_value(s: pd.Series) -> str:
        non_null = s.dropna()
        return "" if non_null.empty else str(non_null.iloc[0])[:28]

    def summary(self) -> dict:
        """أرقام مختصرة تصلح للعرض في التقرير."""
        return {
            "rows": len(self.df),
            "columns": self.df.shape[1],
            "missing_cells": int(self.df.isna().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    # ------------------------------------------------------------------
    # 2) خطوات التنظيف  —  كل خطوة مستقلة وقابلة للتشغيل وحدها
    # ------------------------------------------------------------------
    def standardize_columns(self):
        """توحيد أسماء الأعمدة: حروف صغيرة، بدون مسافات أو رموز."""
        old = list(self.df.columns)
        self.df.columns = [
            re.sub(r"[^\w]+", "_", str(c).strip().lower()).strip("_")
            for c in self.df.columns
        ]
        changed = sum(a != b for a, b in zip(old, self.df.columns))
        if changed:
            self.log.append(f"توحيد أسماء الأعمدة: تم تعديل {changed} عمود")
        return self

    def drop_duplicates(self, label: str = "حذف الصفوف المكررة"):
        """حذف الصفوف المكررة تماماً."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            self.log.append(f"{label}: {removed} صف")
        return self

    def clean_text(self):
        """تنظيف الأعمدة النصية: إزالة المسافات الزائدة وتوحيد حالة الأحرف."""
        cols = self.df.select_dtypes(include="object").columns
        for c in cols:
            self.df[c] = (
                self.df[c].astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .replace({"": pd.NA, "nan": pd.NA, "NA": pd.NA, "N/A": pd.NA, "-": pd.NA})
            )
        if len(cols):
            self.log.append(f"تنظيف النصوص: {len(cols)} عمود نصي")
        return self

    def drop_empty_columns(self):
        """حذف الأعمدة التي تجاوزت نسبة الفراغ فيها الحد المسموح."""
        ratio = self.df.isna().mean()
        drop = ratio[ratio > self.missing_threshold].index.tolist()
        if drop:
            self.df = self.df.drop(columns=drop)
            pct = int(self.missing_threshold * 100)
            self.log.append(f"حذف أعمدة فارغة (أكثر من {pct}%): {', '.join(drop)}")
        return self
    def fill_missing(self):
        """معالجة القيم المفقودة: الوسيط للأرقام، والأكثر تكراراً للفئات."""
        filled = []
        for c in self.df.columns:
            n = int(self.df[c].isna().sum())
            if n == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[c]):
                value = self.df[c].median()
                how = f"الوسيط ({round(float(value), 2)})"
            else:
                mode = self.df[c].mode(dropna=True)
                if mode.empty:
                    continue
                value = mode.iloc[0]
                how = f"الأكثر تكراراً ({value})"
            self.df[c] = self.df[c].fillna(value)
            filled.append(f"{c}: {n} قيمة بـ {how}")
        if filled:
            self.log.append("معالجة القيم المفقودة — " + " | ".join(filled))
        return self

    def fix_dtypes(self):
        """تصحيح أنواع البيانات: أرقام مخزّنة كنص، وتواريخ مخزّنة كنص."""
        fixed = []
        for c in self.df.select_dtypes(include=["object", "string"]).columns:
            converted = pd.to_numeric(self.df[c], errors="coerce")
            if converted.notna().mean() > 0.95:
                self.df[c] = converted
                fixed.append(f"{c} إلى رقم")
                continue
            if re.search(r"date|time|تاريخ", str(c), flags=re.I):
                converted = pd.to_datetime(self.df[c], errors="coerce")
                if converted.notna().mean() > 0.90:
                    self.df[c] = converted
                    fixed.append(f"{c} إلى تاريخ")
        if fixed:
            self.log.append("تصحيح أنواع البيانات — " + " | ".join(fixed))
        return self

    def flag_outliers(self, k: float = 1.5) -> pd.DataFrame:
        """رصد القيم الشاذة بطريقة IQR — ترصد ولا تحذف، والقرار للعميل."""
        rows = []
        for c in self.df.select_dtypes(include=np.number).columns:
            q1, q3 = self.df[c].quantile(0.25), self.df[c].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - k * iqr, q3 + k * iqr
            n = int(((self.df[c] < low) | (self.df[c] > high)).sum())
            if n:
                rows.append({"column": c, "outliers": n,
                             "lower_bound": round(float(low), 2),
                             "upper_bound": round(float(high), 2)})
        out = pd.DataFrame(rows)
        if not out.empty:
            self.log.append(f"رصد القيم الشاذة: {int(out['outliers'].sum())} قيمة في {len(out)} عمود")
        return out

    # ------------------------------------------------------------------
    # 3) التشغيل الكامل والحفظ
    # ------------------------------------------------------------------
    def clean(self):
        """تشغيل خطوات التنظيف بالترتيب الصحيح."""
        self.before = self.summary()
        (self.standardize_columns()
             .drop_duplicates()
             .clean_text()
             .drop_empty_columns()
             .fix_dtypes()
             .fill_missing()
             # ملء القيم المفقودة قد يجعل صفوفاً متطابقة، فنعيد الفحص مرة أخيرة
             .drop_duplicates(label="حذف تكرارات ظهرت بعد ملء القيم المفقودة"))
        self.outliers = self.flag_outliers()
        self.after = self.summary()
        return self

    def save(self, path: str):
        self.df.to_csv(path, index=False, encoding="utf-8-sig")
        return self

    def report(self, path: str = "cleaning_report.html"):
        """تقرير HTML يقارن حالة البيانات قبل وبعد التنظيف."""
        before, after = self.before, self.after
        metrics = [
            ("عدد الصفوف", before["rows"], after["rows"]),
            ("عدد الأعمدة", before["columns"], after["columns"]),
            ("الخلايا الفارغة", before["missing_cells"], after["missing_cells"]),
            ("الصفوف المكررة", before["duplicate_rows"], after["duplicate_rows"]),
        ]
        cards = "".join(
            f"<div class='card'><div class='label'>{name}</div>"
            f"<div class='row'><span class='b'>{b:,}</span>"
            f"<span class='arrow'>&#8592;</span>"
            f"<span class='a'>{a:,}</span></div></div>"
            for name, b, a in metrics
        )
        steps = "".join(f"<li>{s}</li>" for s in self.log)
        html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<title>تقرير تنظيف البيانات</title><style>
body{{background:#f9f9f7;color:#0b0b0b;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:32px}}
.wrap{{max-width:820px;margin:0 auto;background:#fcfcfb;border:1px solid rgba(11,11,11,.10);border-radius:12px;padding:28px}}
h1{{font-size:22px;margin:0 0 4px}} p.sub{{color:#52514e;margin:0 0 24px;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:26px}}
.card{{border:1px solid #e1e0d9;border-radius:10px;padding:14px}}
.label{{color:#898781;font-size:12px;margin-bottom:8px}}
.row{{display:flex;align-items:baseline;gap:8px}}
.b{{color:#898781;font-size:18px;text-decoration:line-through}}
.arrow{{color:#c3c2b7}} .a{{color:#2a78d6;font-size:24px;font-weight:700}}
h2{{font-size:15px;margin:22px 0 10px}}
ol{{padding-inline-start:20px;color:#52514e;font-size:14px;line-height:1.9}}
</style></head><body><div class="wrap">
<h1>تقرير تنظيف البيانات</h1>
<p class="sub">مقارنة حالة الملف قبل وبعد التنظيف — الرقم الرمادي هو الحالة الخام.</p>
<div class="grid">{cards}</div>
<h2>الخطوات المنفّذة</h2><ol>{steps}</ol>
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return self
