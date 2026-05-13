# ================================================================
# log_parser_v4.py — AI Agents IDS v4
# Universal Log Parser + Feature Extractor
# Supports: SSH, Apache, Windows, Firewall,
#           Syslog, CSV, Generic
# ================================================================

import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from typing import Optional


# ================================================================
# Known Attack Patterns
# ================================================================

ATTACK_PATTERNS = {
    "brute_force": [
        r"failed password",
        r"authentication failure",
        r"invalid user",
        r"failed login",
        r"too many authentication",
        r"connection closed by.*\[preauth\]",
    ],
    "scanning": [
        r"port scan",
        r"connection refused",
        r"no route to host",
        r"connection timed out",
        r"syn flood",
    ],
    "web_attack": [
        r"sql injection",
        r"union select",
        r"<script>",
        r"javascript:",
        r"\.\./\.\.",
        r"etc/passwd",
        r"cmd\.exe",
        r"eval\(",
        r"base64_decode",
        r"union.*select",
        r"drop\s+table",
        r"insert\s+into",
        r"xp_cmdshell",
    ],
    "malware": [
        r"malware",
        r"virus",
        r"trojan",
        r"ransomware",
        r"backdoor",
        r"rootkit",
        r"exploit",
        r"payload",
        r"reverse shell",
        r"meterpreter",
    ],
    "exfiltration": [
        r"large.*transfer",
        r"data.*exfil",
        r"unusual.*upload",
        r"ftp.*put",
        r"scp.*\d{3,}mb",
    ],
    "privilege_escalation": [
        r"sudo.*failed",
        r"su.*authentication failure",
        r"privilege.*escalat",
        r"access denied",
        r"permission denied",
        r"unauthorized",
    ],
}

KNOWN_ATTACK_PORTS = {
    21, 22, 23, 25, 53, 80, 110,
    135, 139, 443, 445, 1433, 1521,
    3306, 3389, 4444, 5900, 6379,
    8080, 8443, 27017,
}

HIGH_RISK_PATHS = {
    "/admin", "/wp-admin", "/phpmyadmin",
    "/etc/passwd", "/etc/shadow",
    "/.env", "/config", "/backup",
    "/shell", "/.git", "/api/admin",
}

# ================================================================
# Log Type Detector
# ================================================================

LOG_SIGNATURES = {
    "SSH": [
        r"sshd\[",
        r"Failed password for",
        r"Accepted publickey",
        r"Invalid user",
        r"OpenSSH",
    ],
    "APACHE": [
        r'"\w+ /\S+ HTTP/\d\.\d"',
        r"\d{3}\s+\d+$",
        r"Mozilla/\d\.\d",
        r"GET|POST|PUT|DELETE.*HTTP",
    ],
    "WINDOWS": [
        r"EventID|Event ID",
        r"Security|Application|System",
        r"Account Name:",
        r"Logon Type:",
        r"Source Network Address:",
        r"4624|4625|4648|4672",
    ],
    "FIREWALL": [
        r"ALLOW|DENY|DROP|REJECT|BLOCK",
        r"SRC=\d+\.\d+",
        r"DST=\d+\.\d+",
        r"DPT=\d+",
        r"SPT=\d+",
        r"IN=|OUT=",
    ],
    "SYSLOG": [
        r"\w+\s+\d+\s+\d+:\d+:\d+",
        r"kernel:|systemd:|cron:",
        r"\[\d+\]:",
    ],
    "NGINX": [
        r"nginx",
        r'"\w+ /\S+ HTTP/\d\.\d" \d{3}',
        r"upstream",
    ],
    "DNS": [
        r"query\[",
        r"NOERROR|NXDOMAIN|SERVFAIL",
        r"IN A|IN AAAA|IN MX",
        r"named\[",
    ],
    "ZEEK": [
        r"#fields\t",
        r"#separator",
        r"conn\.log|dns\.log|http\.log",
        r"\t\d+\.\d+\t",
    ],
}


def detect_log_type(lines: list) -> str:
    """يكتشف نوع الـ log من أول 50 سطر"""
    sample = "\n".join(lines[:50]).lower()
    scores = {}

    for log_type, patterns in \
            LOG_SIGNATURES.items():
        score = 0
        for pat in patterns:
            if re.search(
                    pat.lower(), sample,
                    re.IGNORECASE):
                score += 1
        scores[log_type] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "GENERIC"
    return best


# ================================================================
# Individual Log Parsers
# ================================================================

def parse_ssh_line(line: str) -> Optional[dict]:
    """يحلل سطر SSH log"""
    result = {
        "timestamp"  : None,
        "src_ip"     : None,
        "dst_port"   : 22,
        "action"     : None,
        "user"       : None,
        "message"    : line.strip(),
        "log_type"   : "SSH",
    }

    # timestamp
    ts_match = re.search(
        r"(\w+\s+\d+\s+\d+:\d+:\d+)", line)
    if ts_match:
        result["timestamp"] = ts_match.group(1)

    # IP
    ip_match = re.search(
        r"from\s+(\d+\.\d+\.\d+\.\d+)", line,
        re.IGNORECASE)
    if ip_match:
        result["src_ip"] = ip_match.group(1)

    # port
    port_match = re.search(
        r"port\s+(\d+)", line, re.IGNORECASE)
    if port_match:
        result["src_port"] = int(
            port_match.group(1))

    # user
    user_match = re.search(
        r"(?:for|user)\s+(\w+)", line,
        re.IGNORECASE)
    if user_match:
        result["user"] = user_match.group(1)

    # action
    line_lower = line.lower()
    if any(p in line_lower for p in [
            "failed", "invalid", "error",
            "refused"]):
        result["action"] = "FAIL"
    elif any(p in line_lower for p in [
            "accepted", "success",
            "opened"]):
        result["action"] = "SUCCESS"
    else:
        result["action"] = "INFO"

    return result


def parse_apache_line(
        line: str) -> Optional[dict]:
    """يحلل سطر Apache/Nginx log"""
    result = {
        "timestamp"  : None,
        "src_ip"     : None,
        "dst_port"   : 80,
        "action"     : None,
        "method"     : None,
        "path"       : None,
        "status_code": None,
        "bytes_sent" : 0,
        "user_agent" : None,
        "message"    : line.strip(),
        "log_type"   : "APACHE",
    }

    # Combined log format
    # IP - - [timestamp] "method path proto"
    # status bytes
    combined = re.match(
        r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+'
        r'\[([^\]]+)\]\s+"(\w+)\s+(\S+)'
        r'\s+\S+"\s+(\d+)\s+(\d+|-)'
        r'(?:\s+"[^"]*"\s+"([^"]*)")?',
        line)

    if combined:
        result["src_ip"]      = combined.group(1)
        result["timestamp"]   = combined.group(2)
        result["method"]      = combined.group(3)
        result["path"]        = combined.group(4)
        result["status_code"] = int(
            combined.group(5))
        result["bytes_sent"]  = int(
            combined.group(6)) \
            if combined.group(6) != "-" else 0
        result["user_agent"]  = combined.group(7)

        sc = result["status_code"]
        if sc >= 400:
            result["action"] = "FAIL"
        elif sc >= 200:
            result["action"] = "SUCCESS"
        else:
            result["action"] = "INFO"
    else:
        # fallback
        ip_m = re.search(
            r"(\d+\.\d+\.\d+\.\d+)", line)
        if ip_m:
            result["src_ip"] = ip_m.group(1)
        sc_m = re.search(r'" (\d{3}) ', line)
        if sc_m:
            result["status_code"] = int(
                sc_m.group(1))
            result["action"] = (
                "FAIL"
                if result["status_code"] >= 400
                else "SUCCESS")

    return result


def parse_windows_line(
        line: str) -> Optional[dict]:
    """يحلل Windows Event log"""
    result = {
        "timestamp"  : None,
        "src_ip"     : None,
        "dst_port"   : None,
        "action"     : None,
        "event_id"   : None,
        "user"       : None,
        "message"    : line.strip(),
        "log_type"   : "WINDOWS",
    }

    # Event ID
    eid_m = re.search(
        r"(?:EventID|Event ID)[:\s]+(\d+)",
        line, re.IGNORECASE)
    if eid_m:
        result["event_id"] = int(
            eid_m.group(1))
        eid = result["event_id"]
        if eid in [4625, 4648, 4776]:
            result["action"] = "FAIL"
        elif eid in [4624, 4672]:
            result["action"] = "SUCCESS"
        else:
            result["action"] = "INFO"

    # IP
    ip_m = re.search(
        r"(?:Source Network Address|"
        r"Workstation Name)[:\s]+"
        r"(\d+\.\d+\.\d+\.\d+)",
        line, re.IGNORECASE)
    if ip_m:
        result["src_ip"] = ip_m.group(1)

    # timestamp
    ts_m = re.search(
        r"(\d{4}-\d{2}-\d{2}"
        r"\s+\d{2}:\d{2}:\d{2})",
        line)
    if ts_m:
        result["timestamp"] = ts_m.group(1)

    # user
    user_m = re.search(
        r"Account Name[:\s]+(\w+)",
        line, re.IGNORECASE)
    if user_m:
        result["user"] = user_m.group(1)

    return result


def parse_firewall_line(
        line: str) -> Optional[dict]:
    """يحلل Firewall/iptables log"""
    result = {
        "timestamp"  : None,
        "src_ip"     : None,
        "dst_ip"     : None,
        "src_port"   : None,
        "dst_port"   : None,
        "protocol"   : None,
        "action"     : None,
        "bytes"      : 0,
        "message"    : line.strip(),
        "log_type"   : "FIREWALL",
    }

    # iptables format
    for key, pattern in [
        ("src_ip",  r"SRC=(\d+\.\d+\.\d+\.\d+)"),
        ("dst_ip",  r"DST=(\d+\.\d+\.\d+\.\d+)"),
        ("src_port",r"SPT=(\d+)"),
        ("dst_port",r"DPT=(\d+)"),
        ("protocol",r"PROTO=(\w+)"),
        ("bytes",   r"LEN=(\d+)"),
    ]:
        m = re.search(pattern, line,
                      re.IGNORECASE)
        if m:
            val = m.group(1)
            if key in ["src_port",
                       "dst_port","bytes"]:
                val = int(val)
            result[key] = val

    line_upper = line.upper()
    if any(a in line_upper for a in [
            "DROP","DENY","REJECT","BLOCK"]):
        result["action"] = "DENY"
    elif any(a in line_upper for a in [
            "ALLOW","ACCEPT","PERMIT"]):
        result["action"] = "ALLOW"
    else:
        result["action"] = "INFO"

    ts_m = re.search(
        r"(\w+\s+\d+\s+\d+:\d+:\d+)", line)
    if ts_m:
        result["timestamp"] = ts_m.group(1)

    return result


def parse_dns_line(
        line: str) -> Optional[dict]:
    """يحلل DNS log"""
    result = {
        "timestamp" : None,
        "src_ip"    : None,
        "dst_port"  : 53,
        "action"    : None,
        "query"     : None,
        "qtype"     : None,
        "response"  : None,
        "message"   : line.strip(),
        "log_type"  : "DNS",
    }

    ip_m = re.search(
        r"(\d+\.\d+\.\d+\.\d+)", line)
    if ip_m:
        result["src_ip"] = ip_m.group(1)

    q_m = re.search(
        r"query\[(\w+)\]\s+(\S+)", line,
        re.IGNORECASE)
    if q_m:
        result["qtype"] = q_m.group(1)
        result["query"] = q_m.group(2)

    resp_m = re.search(
        r"(NOERROR|NXDOMAIN|SERVFAIL"
        r"|REFUSED)", line)
    if resp_m:
        result["response"] = resp_m.group(1)
        result["action"] = (
            "FAIL"
            if result["response"] in [
                "NXDOMAIN","SERVFAIL",
                "REFUSED"]
            else "SUCCESS")

    return result


def parse_generic_line(
        line: str) -> Optional[dict]:
    """Parser عام لأي log"""
    result = {
        "timestamp" : None,
        "src_ip"    : None,
        "dst_port"  : None,
        "action"    : None,
        "message"   : line.strip(),
        "log_type"  : "GENERIC",
    }

    ip_m = re.search(
        r"(\d{1,3}\.\d{1,3}"
        r"\.\d{1,3}\.\d{1,3})", line)
    if ip_m:
        result["src_ip"] = ip_m.group(1)

    port_m = re.search(
        r"(?:port|dpt|spt)[:\s]+(\d+)",
        line, re.IGNORECASE)
    if port_m:
        result["dst_port"] = int(
            port_m.group(1))

    ts_patterns = [
        r"(\d{4}-\d{2}-\d{2}"
        r"\s+\d{2}:\d{2}:\d{2})",
        r"(\w+\s+\d+\s+\d+:\d+:\d+)",
        r"(\d{2}/\w+/\d{4}:\d{2}:\d{2}:\d{2})",
    ]
    for pat in ts_patterns:
        ts_m = re.search(pat, line)
        if ts_m:
            result["timestamp"] = ts_m.group(1)
            break

    line_lower = line.lower()
    if any(w in line_lower for w in [
            "error","fail","denied",
            "refused","invalid","blocked"]):
        result["action"] = "FAIL"
    elif any(w in line_lower for w in [
            "success","accept","allow",
            "permit","connected"]):
        result["action"] = "SUCCESS"
    else:
        result["action"] = "INFO"

    return result


# ================================================================
# Attack Pattern Detector
# ================================================================

def detect_attack_patterns(
        line: str) -> dict:
    """يكتشف أنماط الهجوم في سطر واحد"""
    line_lower = line.lower()
    detected = {}

    for attack_type, patterns in \
            ATTACK_PATTERNS.items():
        for pat in patterns:
            if re.search(
                    pat, line_lower,
                    re.IGNORECASE):
                detected[attack_type] = True
                break

    return detected


def get_path_risk(path: Optional[str]) -> float:
    """يحسب مستوى خطورة الـ URL path"""
    if not path:
        return 0.0
    path_lower = path.lower()
    for risk_path in HIGH_RISK_PATHS:
        if risk_path in path_lower:
            return 1.0
    if re.search(
            r"\.\./|%2e%2e|%252e",
            path_lower):
        return 1.0
    if re.search(
            r"(?:select|union|insert|"
            r"drop|exec|eval)",
            path_lower):
        return 1.0
    return 0.0


# ================================================================
# Log to Abstract Features Converter
# ================================================================

def log_record_to_features(
        record: dict,
        ip_freq: dict,
        fail_counts: dict,
        total_lines: int) -> dict:
    """
    يحول سطر log محلّل إلى
    18 abstract behavioral features
    """
    src_ip   = record.get("src_ip") or ""
    dst_port = record.get(
        "dst_port") or 0
    action   = record.get(
        "action") or "INFO"
    msg      = record.get("message") or ""
    log_type = record.get("log_type","")

    # حساب IP frequency
    ip_count = ip_freq.get(src_ip, 1)
    ip_freq_norm = min(
        ip_count / max(total_lines, 1), 1.0)

    # حساب fail ratio لهذا الـ IP
    fail_c = fail_counts.get(src_ip, 0)
    fail_ratio = min(
        fail_c / max(ip_count, 1), 1.0)

    # attack patterns
    patterns  = detect_attack_patterns(msg)
    n_patterns = len(patterns)

    # path risk (Apache/Web)
    path_risk = get_path_risk(
        record.get("path"))

    # bytes
    bytes_val = float(
        record.get("bytes_sent",
        record.get("bytes", 0)) or 0)

    # status code
    sc = record.get("status_code") or 0

    # port risk
    dst_port_risk = 1.0 \
        if int(dst_port) in \
        KNOWN_ATTACK_PORTS else 0.0

    # action encoding
    action_score = {
        "FAIL": 1.0,
        "DENY": 0.8,
        "INFO": 0.2,
        "SUCCESS": 0.0,
        "ALLOW": 0.0,
    }.get(action, 0.3)

    # log type encoding
    log_type_score = {
        "SSH"     : 1,
        "FIREWALL": 2,
        "APACHE"  : 3,
        "NGINX"   : 3,
        "WINDOWS" : 4,
        "DNS"     : 5,
        "SYSLOG"  : 6,
        "GENERIC" : 0,
    }.get(log_type, 0)

    # بناء الـ 18 features
    features = {
        # Volume
        "total_bytes"      : bytes_val,
        "bytes_per_packet" : bytes_val /
                             max(ip_count,1),
        # Ratio
        "packet_ratio"     : ip_freq_norm * 10,
        "byte_ratio"       : (
            sc / 100.0 if sc > 0
            else action_score),
        "flow_asymmetry"   : fail_ratio,
        # Timing
        "duration_log"     : np.log1p(
            ip_count),
        "packet_rate"      : ip_freq_norm *
                             1000,
        "inter_arrival_var": float(
            n_patterns * 100),
        # Ports
        "src_port_risk"    : 1.0
                             if action ==
                             "FAIL" else 0.0,
        "dst_port_risk"    : dst_port_risk,
        "port_entropy"     : min(
            dst_port / 65535.0, 1.0),
        # Protocol
        "protocol_encoded" : float(
            log_type_score),
        "tcp_flag_pattern" : float(
            n_patterns > 0),
        # Behavior
        "connection_freq"  : ip_freq_norm *
                             500,
        "failed_conn_ratio": fail_ratio,
        "payload_entropy"  : path_risk,
        "session_uniqueness": float(
            n_patterns > 1),
        "burstiness"       : min(
            ip_count / 100.0, 1.0) *
                             action_score,
    }

    return features


# ================================================================
# Main Log Parser Class
# ================================================================

class UniversalLogParser:
    """
    يحلل أي نوع log ويحوله لـ
    abstract features للنموذج
    """

    def __init__(self):
        self.log_type    = None
        self.parsed_lines = []
        self.stats       = {}

    def parse_file(
            self,
            content: str,
            max_lines: int = 10000
    ) -> pd.DataFrame:
        """
        Input:  نص الـ log file
        Output: DataFrame بـ 18 features
        """
        lines = [
            l for l in content.split("\n")
            if l.strip()][:max_lines]

        if not lines:
            return pd.DataFrame()

        # 1. Detect type
        self.log_type = detect_log_type(lines)
        print(f"  📋 Log Type: {self.log_type}")

        # 2. اختار الـ parser المناسب
        parser_map = {
            "SSH"     : parse_ssh_line,
            "APACHE"  : parse_apache_line,
            "NGINX"   : parse_apache_line,
            "WINDOWS" : parse_windows_line,
            "FIREWALL": parse_firewall_line,
            "DNS"     : parse_dns_line,
        }
        parser = parser_map.get(
            self.log_type, parse_generic_line)

        # 3. حلّل كل سطر
        records = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = parser(line)
                if rec:
                    # أضف attack patterns
                    rec["attack_patterns"] = \
                        detect_attack_patterns(
                            line)
                    records.append(rec)
            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        self.parsed_lines = records

        # 4. احسب IP statistics
        ip_freq = Counter(
            r.get("src_ip","")
            for r in records
            if r.get("src_ip"))
        fail_counts = Counter(
            r.get("src_ip","")
            for r in records
            if r.get("src_ip") and
            r.get("action") in [
                "FAIL","DENY"])

        # 5. حول لـ features
        features_list = []
        for rec in records:
            feat = log_record_to_features(
                rec, ip_freq,
                fail_counts, len(records))
            features_list.append(feat)

        df_features = pd.DataFrame(
            features_list)

        # 6. إحصاءات
        self.stats = {
            "total_lines"   : len(lines),
            "parsed_lines"  : len(records),
            "log_type"      : self.log_type,
            "unique_ips"    : len(ip_freq),
            "top_ips"       : ip_freq.most_common(
                10),
            "fail_count"    : sum(
                1 for r in records
                if r.get("action") in
                ["FAIL","DENY"]),
            "attack_patterns": Counter(
                p for r in records
                for p in r.get(
                    "attack_patterns",
                    {}).keys()),
            "actions"       : Counter(
                r.get("action","INFO")
                for r in records),
        }

        print(f"  ✅ Parsed: "
              f"{len(records):,} lines")
        print(f"  📊 Unique IPs: "
              f"{len(ip_freq):,}")
        print(f"  🚨 Failed: "
              f"{self.stats['fail_count']:,}")

        return df_features

    def get_suspicious_ips(
            self,
            top_n: int = 20) -> pd.DataFrame:
        """يرجع أكثر الـ IPs اشتباهاً"""
        if not self.parsed_lines:
            return pd.DataFrame()

        ip_data = {}
        for rec in self.parsed_lines:
            ip = rec.get("src_ip")
            if not ip:
                continue
            if ip not in ip_data:
                ip_data[ip] = {
                    "ip"          : ip,
                    "total"       : 0,
                    "fails"       : 0,
                    "patterns"    : set(),
                    "ports"       : set(),
                }
            ip_data[ip]["total"] += 1
            if rec.get("action") in [
                    "FAIL","DENY"]:
                ip_data[ip]["fails"] += 1
            for p in rec.get(
                    "attack_patterns",
                    {}).keys():
                ip_data[ip]["patterns"].add(p)
            if rec.get("dst_port"):
                ip_data[ip]["ports"].add(
                    rec["dst_port"])

        rows = []
        for ip, d in ip_data.items():
            fail_rate = d["fails"] / \
                        max(d["total"], 1)
            risk_score = (
                fail_rate * 0.4 +
                min(len(d["patterns"]) /
                    5.0, 1.0) * 0.4 +
                min(d["total"] /
                    100.0, 1.0) * 0.2)
            rows.append({
                "IP"         : ip,
                "Requests"   : d["total"],
                "Failures"   : d["fails"],
                "Fail Rate"  : f"{fail_rate*100:.1f}%",
                "Patterns"   : ", ".join(
                    d["patterns"]) or "—",
                "Risk Score" : f"{risk_score:.2f}",
                "Risk Level" : (
                    "🔴 High"
                    if risk_score > 0.6
                    else "🟡 Medium"
                    if risk_score > 0.3
                    else "🟢 Low"),
            })

        rows.sort(
            key=lambda x: float(
                x["Risk Score"]),
            reverse=True)
        return pd.DataFrame(rows[:top_n])

    def get_timeline(self) -> pd.DataFrame:
        """يبني timeline للأحداث"""
        if not self.parsed_lines:
            return pd.DataFrame()

        events = []
        for i, rec in enumerate(
                self.parsed_lines):
            patterns = rec.get(
                "attack_patterns", {})
            severity = (
                "HIGH"   if patterns else
                "MEDIUM" if rec.get(
                    "action") in
                ["FAIL","DENY"]
                else "LOW")
            events.append({
                "index"    : i,
                "timestamp": rec.get(
                    "timestamp", f"t{i}"),
                "src_ip"   : rec.get(
                    "src_ip", "unknown"),
                "action"   : rec.get(
                    "action", "INFO"),
                "severity" : severity,
                "patterns" : ", ".join(
                    patterns.keys()) or "—",
                "log_type" : rec.get(
                    "log_type", "GENERIC"),
            })

        return pd.DataFrame(events)
