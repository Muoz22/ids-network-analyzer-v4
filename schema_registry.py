# ================================================================
# schema_registry.py — AI Agents IDS v4
# Dataset Schema Mappings — No Hardcoding
# ================================================================

NORMALIZED_SCHEMA = {
    "src_ip"           : str,
    "dst_ip"           : str,
    "src_port"         : float,
    "dst_port"         : float,
    "protocol"         : str,
    "timestamp"        : float,
    "duration"         : float,
    "src_bytes"        : float,
    "dst_bytes"        : float,
    "src_pkts"         : float,
    "dst_pkts"         : float,
    "tcp_flags"        : float,
    "payload_size"     : float,
    "connection_state" : str,
    "label"            : str,
}

DATASET_SCHEMAS = {

    "TON_IOT": {
        "src_ip"           : "saddr",
        "dst_ip"           : "daddr",
        "src_port"         : "sport",
        "dst_port"         : "dport",
        "protocol"         : "proto",
        "timestamp"        : "ts",
        "duration"         : "duration",
        "src_bytes"        : "src_ip_bytes",
        "dst_bytes"        : "dst_ip_bytes",
        "src_pkts"         : "src_pkts",
        "dst_pkts"         : "dst_pkts",
        "tcp_flags"        : None,
        "payload_size"     : "src_bytes",
        "connection_state" : "conn_state",
        "label"            : "type",
        "benign_label"     : "normal",
    },

    "CIC_IOT": {
        "src_ip"           : None,
        "dst_ip"           : None,
        "src_port"         : None,
        "dst_port"         : None,
        "protocol"         : "Protocol Type",
        "timestamp"        : None,
        "duration"         : "flow_duration",
        # Srate/Drate كـ proxy لـ src/dst bytes
        "src_bytes"        : "Srate",
        "dst_bytes"        : "Drate",
        # Rate كـ proxy لـ packets
        "src_pkts"         : "Rate",
        "dst_pkts"         : "Drate",
        "tcp_flags"        : "fin_flag_number",
        "payload_size"     : "Header_Length",
        "connection_state" : None,
        "label"            : "label",
        "benign_label"     : "BenignTraffic",
    },

    "UNSW_NB15": {
    "src_ip"           : "srcip",
    "dst_ip"           : "dstip",
    "src_port"         : "sport",
    "dst_port"         : "dsport",
    "protocol"         : "proto",
    "timestamp"        : "stime",
    "duration"         : "dur",
    "src_bytes"        : "sbytes",
    "dst_bytes"        : "dbytes",
    "src_pkts"         : "spkts",
    "dst_pkts"         : "dpkts",
    "tcp_flags"        : "swin",
    "payload_size"     : "smean",
    "connection_state" : "state",
    "label"            : "attack_cat",  
    "benign_label"     : "Normal",      
},

    "BOT_IOT": {
    "src_ip"           : "saddr",
    "dst_ip"           : "daddr",
    "src_port"         : "sport",
    "dst_port"         : "dport",
    "protocol"         : "proto",
    "timestamp"        : "stime",
    "duration"         : "dur",
    "src_bytes"        : "sbytes",
    "dst_bytes"        : "dbytes",
    "src_pkts"         : "spkts",
    "dst_pkts"         : "dpkts",
    "tcp_flags"        : "flgs_number",
    "payload_size"     : "mean",
    "connection_state" : "state",
    "label"            : "category",
    "benign_label"     : "Normal",
},
    "CICIDS2017": {
        "src_ip"           : "Source IP",
        "dst_ip"           : "Destination IP",
        "src_port"         : "Source Port",
        "dst_port"         : "Destination Port",
        "protocol"         : "Protocol",
        "timestamp"        : "Timestamp",
        "duration"         : "Flow Duration",
        "src_bytes"        : "Total Fwd Packets",
        "dst_bytes"        : "Total Backward Packets",
        "src_pkts"         : "Total Fwd Packets",
        "dst_pkts"         : "Total Backward Packets",
        "tcp_flags"        : "FIN Flag Count",
        "payload_size"     : "Average Packet Size",
        "connection_state" : None,
        "label"            : "Label",
        "benign_label"     : "BENIGN",
    },

    "AUTO": {
        "label"        : None,
        "benign_label" : None,
    },
}


def get_schema(dataset_name: str) -> dict:
    name = dataset_name.upper().replace(
        "-","_").replace(" ","_")
    return DATASET_SCHEMAS.get(
        name, DATASET_SCHEMAS["AUTO"])


def detect_dataset(df) -> str:
    """
    يكتشف نوع الداتاست تلقائياً
    من خلال أسماء الأعمدة
    """
    cols_lower = {c.lower() for c in df.columns}

    signatures = {
        "TON_IOT"   : {
            "saddr","daddr","proto","type",
            "src_pkts","dst_pkts"},
        "CIC_IOT"   : {
            "label","header_length",
            "protocol type","flow_duration",
            "srate","drate"},
        "UNSW_NB15" : {
            "srcip","dstip","sbytes",
            "dbytes","category","dur"},
        "BOT_IOT"   : {
            "saddr","daddr","category",
            "subcategory","pkts"},
        "CICIDS2017": {
            "source ip","destination ip",
            "flow duration","total fwd packets"},
    }

    best_match = "AUTO"
    best_score = 0

    for name, sig in signatures.items():
        score = len(
            sig & cols_lower)
        if score > best_score:
            best_score = score
            best_match = name

    return best_match
