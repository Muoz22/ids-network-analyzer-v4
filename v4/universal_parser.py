
import pandas as pd
import numpy as np
from schema_registry import NORMALIZED_SCHEMA,detect_dataset,get_schema
from taxonomy import unify_series

class UniversalParser:
    def __init__(self,dataset_name="AUTO"):
        self.dataset_name=dataset_name
        self.schema=None
        self.detected=None
    def parse(self,df,unify_labels=True):
        if self.dataset_name=="AUTO":
            self.detected=detect_dataset(df)
        else:
            self.detected=self.dataset_name.upper().replace("-","_").replace(" ","_")
        self.schema=get_schema(self.detected)
        print(f"  📋 Detected: {self.detected}")
        result=pd.DataFrame(index=df.index)
        for norm_col,dtype in NORMALIZED_SCHEMA.items():
            if norm_col=="label": continue
            src_col=self.schema.get(norm_col)
            if src_col and src_col in df.columns:
                result[norm_col]=df[src_col]
            elif norm_col in df.columns:
                result[norm_col]=df[norm_col]
            else:
                result[norm_col]="" if dtype==str else 0.0
        numeric_cols=[c for c,t in NORMALIZED_SCHEMA.items() if t==float and c in result.columns]
        for col in numeric_cols:
            result[col]=pd.to_numeric(result[col],errors="coerce").fillna(0).clip(lower=0)
        label_col=self.schema.get("label")
        if label_col and label_col in df.columns:
            result["label_original"]=df[label_col].astype(str)
            result["label"]=unify_series(result["label_original"]) if unify_labels else result["label_original"]
        else:
            result["label_original"]="unknown"
            result["label"]="unknown"
        if "duration" in result.columns:
            med=result["duration"].median()
            if med>1e6: result["duration"]=result["duration"]/1e6
            elif med>1e3: result["duration"]=result["duration"]/1e3
        print(f"  ✅ Parsed: {len(result):,} rows × {len(result.columns)} cols")
        print(f"  📊 Labels: {dict(result['label'].value_counts().head(5))}")
        return result
    def get_benign_label(self):
        return self.schema.get("benign_label","benign") if self.schema else "benign"
