# ================================================================
# universal_parser.py — AI Agents IDS v4
# Universal Dataset Parser + Auto-Detector
# ================================================================

import pandas as pd
import numpy as np
from schema_registry import (
    NORMALIZED_SCHEMA,
    DATASET_SCHEMAS,
    detect_dataset,
    get_schema)
from taxonomy import unify_series


class UniversalParser:
    """
    يقرأ أي داتاست شبكية ويحوّلها
    للـ Normalized Schema الموحّد
    """

    def __init__(self, dataset_name: str = "AUTO"):
        self.dataset_name = dataset_name
        self.schema       = None
        self.detected     = None

    def parse(self,
              df: pd.DataFrame,
              unify_labels: bool = True
              ) -> pd.DataFrame:

        # ── Auto-detect ────────────────────────────
        if self.dataset_name == "AUTO":
            self.detected = detect_dataset(df)
        else:
            self.detected = self.dataset_name.upper(
            ).replace("-","_").replace(" ","_")

        self.schema = get_schema(self.detected)
        print(f"  📋 Detected: {self.detected}")

        result = pd.DataFrame(index=df.index)

        # ── Map columns ────────────────────────────
        for norm_col, dtype in \
                NORMALIZED_SCHEMA.items():
            if norm_col == "label":
                continue

            src_col = self.schema.get(norm_col)

            if src_col and src_col in df.columns:
                result[norm_col] = df[src_col]
            elif norm_col in df.columns:
                result[norm_col] = df[norm_col]
            else:
                # default values
                if dtype == str:
                    result[norm_col] = ""
                else:
                    result[norm_col] = 0.0

        # ── Numeric conversion ─────────────────────
        numeric_cols = [
            c for c, t in NORMALIZED_SCHEMA.items()
            if t == float and c in result.columns]

        for col in numeric_cols:
            result[col] = pd.to_numeric(
                result[col], errors="coerce"
            ).fillna(0).clip(lower=0)

        # ── Label mapping ──────────────────────────
        label_col = self.schema.get("label")
        if label_col and label_col in df.columns:
            result["label_original"] = \
                df[label_col].astype(str)
            if unify_labels:
                result["label"] = unify_series(
                    result["label_original"])
            else:
                result["label"] = \
                    result["label_original"]
        else:
            result["label_original"] = "unknown"
            result["label"]          = "unknown"

        # ── Duration normalization ─────────────────
        # بعض الداتاسيت تعطي duration بالميكروثانية
        if "duration" in result.columns:
            dur_median = result["duration"].median()
            if dur_median > 1e6:
                # ميكروثانية → ثوانٍ
                result["duration"] = \
                    result["duration"] / 1e6
            elif dur_median > 1e3:
                # ميلي ثانية → ثوانٍ
                result["duration"] = \
                    result["duration"] / 1e3

        print(f"  ✅ Parsed: {len(result):,} rows "
              f"× {len(result.columns)} cols")
        print(f"  📊 Labels: "
              f"{result['label'].value_counts().head(5).to_dict()}")

        return result

    def get_benign_label(self) -> str:
        if self.schema:
            return self.schema.get(
                "benign_label", "benign")
        return "benign"
