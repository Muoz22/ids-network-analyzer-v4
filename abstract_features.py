# ================================================================
# abstract_features.py — AI Agents IDS v4
# 18 Isolated, Stateless, Explainable Features
# ================================================================

import numpy as np
import pandas as pd
from feature_contract import (
    REQUIRED_FEATURES,
    FeatureContractValidator)

# ── Known Attack Ports ────────────────────────────────────────
ATTACK_PORTS = {
    21,22,23,25,53,80,110,135,139,
    443,445,1433,1521,3306,3389,
    4444,5900,6379,8080,8443,27017,
}

PROTOCOL_MAP = {
    "tcp":1,"udp":2,"icmp":3,"http":4,
    "https":5,"dns":6,"smtp":7,"ftp":8,
    "ssh":9,"telnet":10,
}

SUSPICIOUS_FLAG_COMBOS = {
    0x29,  # SYN+FIN
    0x03,  # SYN+FIN (alt)
    0x00,  # NULL scan
    0x3F,  # XMAS scan
}


# ================================================================
# Individual Feature Functions — Isolated + Stateless
# ================================================================

def compute_total_bytes(
        src_bytes: pd.Series,
        dst_bytes: pd.Series) -> pd.Series:
    """Total bytes in flow"""
    return (src_bytes.fillna(0) +
            dst_bytes.fillna(0)).clip(lower=0)


def compute_bytes_per_packet(
        total_bytes: pd.Series,
        src_pkts: pd.Series,
        dst_pkts: pd.Series) -> pd.Series:
    """Average bytes per packet"""
    total_pkts = (src_pkts.fillna(0) +
                  dst_pkts.fillna(1))
    return (total_bytes /
            total_pkts.clip(lower=1)).clip(
        lower=0, upper=65535)


def compute_packet_ratio(
        src_pkts: pd.Series,
        dst_pkts: pd.Series) -> pd.Series:
    """src_pkts / (dst_pkts + 1)"""
    return (src_pkts.fillna(0) /
            (dst_pkts.fillna(0) + 1)).clip(
        lower=0, upper=1e6)


def compute_byte_ratio(
        src_bytes: pd.Series,
        dst_bytes: pd.Series) -> pd.Series:
    """src_bytes / (dst_bytes + 1)"""
    return (src_bytes.fillna(0) /
            (dst_bytes.fillna(0) + 1)).clip(
        lower=0, upper=1e6)


def compute_flow_asymmetry(
        src_bytes: pd.Series,
        dst_bytes: pd.Series) -> pd.Series:
    """|src-dst| / (src+dst+1) → 0 to 1"""
    s = src_bytes.fillna(0)
    d = dst_bytes.fillna(0)
    return (np.abs(s - d) /
            (s + d + 1)).clip(0, 1)


def compute_duration_log(
        duration: pd.Series) -> pd.Series:
    """log(duration_seconds + 1)"""
    dur = duration.fillna(0).clip(lower=0)
    return np.log1p(dur)


def compute_packet_rate(
        src_pkts: pd.Series,
        dst_pkts: pd.Series,
        duration: pd.Series) -> pd.Series:
    """Total packets per second"""
    total = (src_pkts.fillna(0) +
             dst_pkts.fillna(0))
    dur   = duration.fillna(0).clip(lower=0.001)
    return (total / dur).clip(lower=0, upper=1e7)


def compute_inter_arrival_var(
        duration: pd.Series,
        src_pkts: pd.Series) -> pd.Series:
    """
    Proxy: duration / pkts^2
    (inter-arrival variance proxy بدون raw timestamps)
    """
    dur  = duration.fillna(0).clip(lower=0)
    pkts = src_pkts.fillna(1).clip(lower=1)
    return (dur / pkts**2).clip(lower=0, upper=1e10)


def compute_src_port_risk(
        src_port: pd.Series) -> pd.Series:
    """1 if ephemeral port (>1024), else 0"""
    return (src_port.fillna(0) > 1024
            ).astype(float)


def compute_dst_port_risk(
        dst_port: pd.Series) -> pd.Series:
    """1 if known attack destination port"""
    return dst_port.fillna(0).apply(
        lambda p: 1.0 if int(p) in ATTACK_PORTS
        else 0.0)


def compute_port_entropy(
        src_port: pd.Series,
        dst_port: pd.Series) -> pd.Series:
    """Normalized entropy of port values"""
    def _entropy(row):
        vals = np.array([
            row["src_port"],
            row["dst_port"]], dtype=float)
        vals = vals[vals > 0]
        if len(vals) < 2:
            return 0.0
        vals = vals / vals.sum()
        ent  = -np.sum(
            vals * np.log2(vals + 1e-10))
        return min(ent / np.log2(2), 1.0)

    tmp = pd.DataFrame({
        "src_port": src_port.fillna(0),
        "dst_port": dst_port.fillna(0)})
    return tmp.apply(_entropy, axis=1)


def compute_protocol_encoded(
        protocol: pd.Series) -> pd.Series:
    """Encode protocol to numeric"""
    return protocol.fillna("unknown").apply(
        lambda p: float(
            PROTOCOL_MAP.get(
                str(p).lower(), 0)))


def compute_tcp_flag_pattern(
        tcp_flags: pd.Series) -> pd.Series:
    """Suspicious TCP flag combination score"""
    return tcp_flags.fillna(0).apply(
        lambda f: 1.0
        if int(f) in SUSPICIOUS_FLAG_COMBOS
        else 0.0)


def compute_connection_freq(
        duration: pd.Series,
        window: float = 1.0) -> pd.Series:
    """
    Proxy: 1 / duration
    (connection frequency estimate)
    """
    dur = duration.fillna(1).clip(lower=0.001)
    return (window / dur).clip(
        lower=0, upper=1e6)


def compute_failed_conn_ratio(
        connection_state: pd.Series) -> pd.Series:
    """
    Ratio of failed/rejected connections
    States: S0, REJ, RSTO = failed
    """
    failed_states = {"s0","rej","rsto","rstos",
                     "rstrh","sh","oth"}
    return connection_state.fillna("").apply(
        lambda s: 1.0
        if str(s).lower() in failed_states
        else 0.0)


def compute_payload_entropy(
        payload_size: pd.Series,
        total_bytes: pd.Series) -> pd.Series:
    """
    Proxy: payload / total_bytes ratio
    (high ratio = more payload = suspicious)
    """
    pay   = payload_size.fillna(0).clip(lower=0)
    total = total_bytes.fillna(1).clip(lower=1)
    return (pay / total).clip(0, 1)


def compute_session_uniqueness(
        src_port: pd.Series,
        dst_port: pd.Series) -> pd.Series:
    """
    Proxy: how unique is src_port per flow
    High ephemeral = scanning behavior
    """
    is_ephemeral = (src_port.fillna(0) > 1024
                    ).astype(float)
    is_random    = (src_port.fillna(0) > 49152
                    ).astype(float)
    return ((is_ephemeral + is_random) / 2
            ).clip(0, 1)


def compute_burstiness(
        src_pkts: pd.Series,
        dst_pkts: pd.Series) -> pd.Series:
    """
    std(pkts) / (mean(pkts) + 1)
    Computed per row as proxy
    """
    s    = src_pkts.fillna(0)
    d    = dst_pkts.fillna(0)
    mean = (s + d) / 2
    std  = np.abs(s - d) / 2
    return (std / (mean + 1)).clip(0, 1e4)


# ================================================================
# Main Extractor
# ================================================================

class AbstractFeatureExtractor:
    """
    يحسب الـ 18 abstract features من أي داتاست
    بعد normalization
    """

    def __init__(self, validate: bool = True):
        self.validate  = validate
        self.validator = FeatureContractValidator()

    def extract(self, df: pd.DataFrame
                ) -> pd.DataFrame:
        """
        Input:  normalized DataFrame
        Output: DataFrame بـ 18 abstract features
        """
        result = pd.DataFrame(index=df.index)

        # ── 1. Total Bytes ─────────────────────────
        result["total_bytes"] = \
            compute_total_bytes(
                df.get("src_bytes",
                       pd.Series(0, index=df.index)),
                df.get("dst_bytes",
                       pd.Series(0, index=df.index)))

        # ── 2. Bytes per Packet ────────────────────
        result["bytes_per_packet"] = \
            compute_bytes_per_packet(
                result["total_bytes"],
                df.get("src_pkts",
                       pd.Series(1, index=df.index)),
                df.get("dst_pkts",
                       pd.Series(1, index=df.index)))

        # ── 3. Packet Ratio ────────────────────────
        result["packet_ratio"] = \
            compute_packet_ratio(
                df.get("src_pkts",
                       pd.Series(1, index=df.index)),
                df.get("dst_pkts",
                       pd.Series(1, index=df.index)))

        # ── 4. Byte Ratio ──────────────────────────
        result["byte_ratio"] = \
            compute_byte_ratio(
                df.get("src_bytes",
                       pd.Series(0, index=df.index)),
                df.get("dst_bytes",
                       pd.Series(0, index=df.index)))

        # ── 5. Flow Asymmetry ──────────────────────
        result["flow_asymmetry"] = \
            compute_flow_asymmetry(
                df.get("src_bytes",
                       pd.Series(0, index=df.index)),
                df.get("dst_bytes",
                       pd.Series(0, index=df.index)))

        # ── 6. Duration Log ────────────────────────
        result["duration_log"] = \
            compute_duration_log(
                df.get("duration",
                       pd.Series(0, index=df.index)))

        # ── 7. Packet Rate ─────────────────────────
        result["packet_rate"] = \
            compute_packet_rate(
                df.get("src_pkts",
                       pd.Series(1, index=df.index)),
                df.get("dst_pkts",
                       pd.Series(1, index=df.index)),
                df.get("duration",
                       pd.Series(1, index=df.index)))

        # ── 8. Inter-Arrival Variance ──────────────
        result["inter_arrival_var"] = \
            compute_inter_arrival_var(
                df.get("duration",
                       pd.Series(1, index=df.index)),
                df.get("src_pkts",
                       pd.Series(1, index=df.index)))

        # ── 9. Src Port Risk ───────────────────────
        result["src_port_risk"] = \
            compute_src_port_risk(
                df.get("src_port",
                       pd.Series(0, index=df.index)))

        # ── 10. Dst Port Risk ──────────────────────
        result["dst_port_risk"] = \
            compute_dst_port_risk(
                df.get("dst_port",
                       pd.Series(0, index=df.index)))

        # ── 11. Port Entropy ───────────────────────
        result["port_entropy"] = \
            compute_port_entropy(
                df.get("src_port",
                       pd.Series(0, index=df.index)),
                df.get("dst_port",
                       pd.Series(0, index=df.index)))

        # ── 12. Protocol Encoded ───────────────────
        result["protocol_encoded"] = \
            compute_protocol_encoded(
                df.get("protocol",
                       pd.Series("tcp",
                                 index=df.index)))

        # ── 13. TCP Flag Pattern ───────────────────
        result["tcp_flag_pattern"] = \
            compute_tcp_flag_pattern(
                df.get("tcp_flags",
                       pd.Series(0, index=df.index)))

        # ── 14. Connection Frequency ───────────────
        result["connection_freq"] = \
            compute_connection_freq(
                df.get("duration",
                       pd.Series(1, index=df.index)))

        # ── 15. Failed Connection Ratio ────────────
        result["failed_conn_ratio"] = \
            compute_failed_conn_ratio(
                df.get("connection_state",
                       pd.Series("",
                                 index=df.index)))

        # ── 16. Payload Entropy ────────────────────
        result["payload_entropy"] = \
            compute_payload_entropy(
                df.get("payload_size",
                       pd.Series(0, index=df.index)),
                result["total_bytes"])

        # ── 17. Session Uniqueness ─────────────────
        result["session_uniqueness"] = \
            compute_session_uniqueness(
                df.get("src_port",
                       pd.Series(0, index=df.index)),
                df.get("dst_port",
                       pd.Series(0, index=df.index)))

        # ── 18. Burstiness ─────────────────────────
        result["burstiness"] = \
            compute_burstiness(
                df.get("src_pkts",
                       pd.Series(1, index=df.index)),
                df.get("dst_pkts",
                       pd.Series(1, index=df.index)))

        # ── NaN cleanup ────────────────────────────
        result = result.fillna(0).replace(
            [np.inf, -np.inf], 0)

        # ── Contract validation ────────────────────
        if self.validate:
            ok = self.validator.validate(result)
            if not ok:
                print(self.validator.report())

        return result[REQUIRED_FEATURES]
