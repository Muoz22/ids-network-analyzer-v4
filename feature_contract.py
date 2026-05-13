# ================================================================
# feature_contract.py — AI Agents IDS v4
# Feature Validation — Semantic Drift Prevention
# ================================================================

import numpy as np
import pandas as pd

# ── Contract Definition ───────────────────────────────────────
FEATURE_CONTRACT = {

    "total_bytes": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e12,
        "meaning" : "Total bytes in flow (src+dst)",
        "unit"    : "bytes",
    },
    "bytes_per_packet": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 65535.0,
        "meaning" : "Average bytes per packet",
        "unit"    : "bytes/packet",
    },
    "packet_ratio": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e6,
        "meaning" : "src_pkts / (dst_pkts + 1)",
        "unit"    : "ratio",
    },
    "byte_ratio": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e6,
        "meaning" : "src_bytes / (dst_bytes + 1)",
        "unit"    : "ratio",
    },
    "flow_asymmetry": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "|src-dst| / (src+dst+1)",
        "unit"    : "normalized ratio",
    },
    "duration_log": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 20.0,
        "meaning" : "log(duration_seconds + 1)",
        "unit"    : "log-seconds",
    },
    "packet_rate": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e7,
        "meaning" : "packets per second",
        "unit"    : "pkts/sec",
    },
    "inter_arrival_var": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e10,
        "meaning" : "variance of inter-arrival times",
        "unit"    : "seconds^2",
    },
    "src_port_risk": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "1 if ephemeral port (>1024)",
        "unit"    : "binary",
    },
    "dst_port_risk": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "1 if known attack port",
        "unit"    : "binary",
    },
    "port_entropy": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "normalized entropy of ports",
        "unit"    : "bits (normalized)",
    },
    "protocol_encoded": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 10.0,
        "meaning" : "encoded protocol (tcp=1,udp=2...)",
        "unit"    : "categorical",
    },
    "tcp_flag_pattern": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "suspicious flag combo score",
        "unit"    : "score 0-1",
    },
    "connection_freq": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e6,
        "meaning" : "connections per time window",
        "unit"    : "conn/sec",
    },
    "failed_conn_ratio": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "failed / total connections",
        "unit"    : "ratio",
    },
    "payload_entropy": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "entropy of payload bytes",
        "unit"    : "bits (normalized)",
    },
    "session_uniqueness": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1.0,
        "meaning" : "unique sessions / total",
        "unit"    : "ratio",
    },
    "burstiness": {
        "dtype"   : float,
        "min"     : 0.0,
        "max"     : 1e4,
        "meaning" : "std(pkts) / (mean(pkts) + 1)",
        "unit"    : "coefficient of variation",
    },
}

REQUIRED_FEATURES = list(FEATURE_CONTRACT.keys())


class FeatureContractValidator:
    """
    يتحقق من صحة الـ features بعد الحساب
    يمنع semantic drift
    """

    def __init__(self):
        self.violations = []

    def validate(self, df: pd.DataFrame) -> bool:
        self.violations = []

        for feat, spec in FEATURE_CONTRACT.items():
            if feat not in df.columns:
                self.violations.append(
                    f"MISSING: {feat}")
                continue

            col = df[feat]

            # dtype check
            if not pd.api.types.is_numeric_dtype(col):
                self.violations.append(
                    f"DTYPE: {feat} not numeric")

            # range check
            if col.min() < spec["min"] - 1e-6:
                self.violations.append(
                    f"RANGE_LOW: {feat} "
                    f"min={col.min():.4f} "
                    f"expected>={spec['min']}")

            if col.max() > spec["max"] * 10:
                self.violations.append(
                    f"RANGE_HIGH: {feat} "
                    f"max={col.max():.4f} "
                    f"expected<={spec['max']}")

            # NaN check
            nan_pct = col.isna().mean()
            if nan_pct > 0.1:
                self.violations.append(
                    f"NAN: {feat} "
                    f"{nan_pct*100:.1f}% missing")

        return len(self.violations) == 0

    def report(self) -> str:
        if not self.violations:
            return "✅ Contract validated — all clear"
        return (f"❌ {len(self.violations)} violations:\n"
                + "\n".join(
                    f"  • {v}"
                    for v in self.violations))
