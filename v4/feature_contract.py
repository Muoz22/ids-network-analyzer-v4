
import numpy as np
import pandas as pd

FEATURE_CONTRACT = {
    "total_bytes":{"dtype":float,"min":0.0,"max":1e12,"meaning":"Total bytes","unit":"bytes"},
    "bytes_per_packet":{"dtype":float,"min":0.0,"max":65535.0,"meaning":"Avg bytes/pkt","unit":"bytes/pkt"},
    "packet_ratio":{"dtype":float,"min":0.0,"max":1e6,"meaning":"src/dst pkts","unit":"ratio"},
    "byte_ratio":{"dtype":float,"min":0.0,"max":1e6,"meaning":"src/dst bytes","unit":"ratio"},
    "flow_asymmetry":{"dtype":float,"min":0.0,"max":1.0,"meaning":"|src-dst|/total","unit":"ratio"},
    "duration_log":{"dtype":float,"min":0.0,"max":20.0,"meaning":"log(dur+1)","unit":"log-sec"},
    "packet_rate":{"dtype":float,"min":0.0,"max":1e7,"meaning":"pkts/sec","unit":"pkts/sec"},
    "inter_arrival_var":{"dtype":float,"min":0.0,"max":1e10,"meaning":"IAT variance","unit":"sec^2"},
    "src_port_risk":{"dtype":float,"min":0.0,"max":1.0,"meaning":"ephemeral port","unit":"binary"},
    "dst_port_risk":{"dtype":float,"min":0.0,"max":1.0,"meaning":"attack port","unit":"binary"},
    "port_entropy":{"dtype":float,"min":0.0,"max":1.0,"meaning":"port entropy","unit":"bits"},
    "protocol_encoded":{"dtype":float,"min":0.0,"max":10.0,"meaning":"protocol","unit":"categorical"},
    "tcp_flag_pattern":{"dtype":float,"min":0.0,"max":1.0,"meaning":"suspicious flags","unit":"score"},
    "connection_freq":{"dtype":float,"min":0.0,"max":1e6,"meaning":"conn/sec","unit":"conn/sec"},
    "failed_conn_ratio":{"dtype":float,"min":0.0,"max":1.0,"meaning":"failed/total","unit":"ratio"},
    "payload_entropy":{"dtype":float,"min":0.0,"max":1.0,"meaning":"payload entropy","unit":"bits"},
    "session_uniqueness":{"dtype":float,"min":0.0,"max":1.0,"meaning":"unique sessions","unit":"ratio"},
    "burstiness":{"dtype":float,"min":0.0,"max":1e4,"meaning":"std/mean pkts","unit":"CoV"},
}
REQUIRED_FEATURES = list(FEATURE_CONTRACT.keys())

class FeatureContractValidator:
    def __init__(self):
        self.violations = []
    def validate(self, df):
        self.violations = []
        for feat, spec in FEATURE_CONTRACT.items():
            if feat not in df.columns:
                self.violations.append(f"MISSING: {feat}")
        return len(self.violations) == 0
    def report(self):
        if not self.violations:
            return "✅ Contract OK"
        return "❌ " + "; ".join(self.violations)
