
import numpy as np
import pandas as pd
from feature_contract import REQUIRED_FEATURES, FeatureContractValidator

ATTACK_PORTS = {21,22,23,25,53,80,110,135,139,443,445,1433,1521,3306,3389,4444,5900,6379,8080,8443,27017}
PROTOCOL_MAP = {"tcp":1,"udp":2,"icmp":3,"http":4,"https":5,"dns":6,"smtp":7,"ftp":8,"ssh":9,"telnet":10}
SUSPICIOUS_FLAG_COMBOS = {0x29,0x03,0x3F}

def compute_total_bytes(s,d):
    return (s.fillna(0)+d.fillna(0)).clip(lower=0)
def compute_bytes_per_packet(tb,sp,dp):
    tp=(sp.fillna(0)+dp.fillna(1))
    return (tb/(tp.clip(lower=1))).clip(0,65535)
def compute_packet_ratio(sp,dp):
    return (sp.fillna(0)/(dp.fillna(0)+1)).clip(0,1e6)
def compute_byte_ratio(sb,db):
    return (sb.fillna(0)/(db.fillna(0)+1)).clip(0,1e6)
def compute_flow_asymmetry(sb,db):
    s,d=sb.fillna(0),db.fillna(0)
    return (np.abs(s-d)/(s+d+1)).clip(0,1)
def compute_duration_log(dur):
    return np.log1p(dur.fillna(0).clip(lower=0))
def compute_packet_rate(sp,dp,dur):
    return ((sp.fillna(0)+dp.fillna(0))/dur.fillna(0).clip(lower=0.001)).clip(0,1e7)
def compute_inter_arrival_var(dur,sp):
    return (dur.fillna(0).clip(lower=0)/sp.fillna(1).clip(lower=1)**2).clip(0,1e10)
def compute_src_port_risk(sp):
    return (sp.fillna(0)>1024).astype(float)
def compute_dst_port_risk(dp):
    return dp.fillna(0).apply(lambda p: 1.0 if int(p) in ATTACK_PORTS else 0.0)
def compute_port_entropy(sp,dp):
    def _e(row):
        vals=np.array([row["s"],row["d"]],dtype=float)
        vals=vals[vals>0]
        if len(vals)<2: return 0.0
        vals=vals/vals.sum()
        return min(-np.sum(vals*np.log2(vals+1e-10))/np.log2(2),1.0)
    return pd.DataFrame({"s":sp.fillna(0),"d":dp.fillna(0)}).apply(_e,axis=1)
def compute_protocol_encoded(proto):
    return proto.fillna("unknown").apply(lambda p: float(PROTOCOL_MAP.get(str(p).lower(),0)))
def compute_tcp_flag_pattern(flags):
    return flags.fillna(-1).apply(lambda f: 1.0 if int(f) in SUSPICIOUS_FLAG_COMBOS and int(f)!=-1 else 0.0)
def compute_connection_freq(dur,window=1.0):
    return (window/dur.fillna(1).clip(lower=0.001)).clip(0,1e6)
def compute_failed_conn_ratio(state):
    failed={"s0","rej","rsto","rstos","rstrh","sh","oth"}
    return state.fillna("").apply(lambda s: 1.0 if str(s).lower() in failed else 0.0)
def compute_payload_entropy(pay,tb):
    return (pay.fillna(0).clip(lower=0)/tb.fillna(1).clip(lower=1)).clip(0,1)
def compute_session_uniqueness(sp,dp):
    e=(sp.fillna(0)>1024).astype(float)
    r=(sp.fillna(0)>49152).astype(float)
    return ((e+r)/2).clip(0,1)
def compute_burstiness(sp,dp):
    s,d=sp.fillna(0),dp.fillna(0)
    mean=(s+d)/2
    std=np.abs(s-d)/2
    return (std/(mean+1)).clip(0,1e4)

class AbstractFeatureExtractor:
    def __init__(self, validate=True):
        self.validate=validate
        self.validator=FeatureContractValidator()
    def extract(self, df):
        z=pd.Series(0,index=df.index)
        s=pd.Series("",index=df.index)
        r=pd.DataFrame(index=df.index)
        r["total_bytes"]=compute_total_bytes(df.get("src_bytes",z),df.get("dst_bytes",z))
        r["bytes_per_packet"]=compute_bytes_per_packet(r["total_bytes"],df.get("src_pkts",z),df.get("dst_pkts",z))
        r["packet_ratio"]=compute_packet_ratio(df.get("src_pkts",z),df.get("dst_pkts",z))
        r["byte_ratio"]=compute_byte_ratio(df.get("src_bytes",z),df.get("dst_bytes",z))
        r["flow_asymmetry"]=compute_flow_asymmetry(df.get("src_bytes",z),df.get("dst_bytes",z))
        r["duration_log"]=compute_duration_log(df.get("duration",z))
        r["packet_rate"]=compute_packet_rate(df.get("src_pkts",z),df.get("dst_pkts",z),df.get("duration",z))
        r["inter_arrival_var"]=compute_inter_arrival_var(df.get("duration",z),df.get("src_pkts",z))
        r["src_port_risk"]=compute_src_port_risk(df.get("src_port",z))
        r["dst_port_risk"]=compute_dst_port_risk(df.get("dst_port",z))
        r["port_entropy"]=compute_port_entropy(df.get("src_port",z),df.get("dst_port",z))
        r["protocol_encoded"]=compute_protocol_encoded(df.get("protocol",s))
        r["tcp_flag_pattern"]=compute_tcp_flag_pattern(df.get("tcp_flags",z))
        r["connection_freq"]=compute_connection_freq(df.get("duration",z))
        r["failed_conn_ratio"]=compute_failed_conn_ratio(df.get("connection_state",s))
        r["payload_entropy"]=compute_payload_entropy(df.get("payload_size",z),r["total_bytes"])
        r["session_uniqueness"]=compute_session_uniqueness(df.get("src_port",z),df.get("dst_port",z))
        r["burstiness"]=compute_burstiness(df.get("src_pkts",z),df.get("dst_pkts",z))
        r=r.fillna(0).replace([np.inf,-np.inf],0)
        return r[REQUIRED_FEATURES]
