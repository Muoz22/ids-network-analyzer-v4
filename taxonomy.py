# ================================================================
# taxonomy.py — AI Agents IDS v4
# Unified Attack Taxonomy — 2-Level Hierarchical
# ================================================================

TAXONOMY = {

    "benign": {
        "level1" : "benign",
        "level2" : "benign",
        "labels" : [
            "normal","Normal","BENIGN","benign",
            "Benign","BenignTraffic","legitimate",
            "background","Background","safe","Safe",
            "none","None",
        ]
    },

    "ddos": {
        "level1" : "attack",
        "level2" : "ddos",
        "labels" : [
            "ddos","DDoS",
            "DDoS-ICMP_Flood","DDoS-UDP_Flood",
            "DDoS-TCP_Flood","DDoS-SYN_Flood",
            "DDoS-HTTP_Flood","DDoS-SlowLoris",
            "DDoS-PSHACK_Flood","DDoS-RSTFINFlood",
            "DDoS-ACK_Fragmentation",
            "DDoS-ICMP_Fragmentation",
            "DDoS-UDP_Fragmentation",
            "DDoS-SynonymousIP_Flood",
        ]
    },

    "dos": {
        "level1" : "attack",
        "level2" : "dos",
        "labels" : [
            "dos","DoS",
            "DoS-SYN_Flood","DoS-UDP_Flood",
            "DoS-TCP_Flood","DoS-HTTP_Flood",
            "DoS",
        ]
    },

    "recon": {
        "level1" : "attack",
        "level2" : "recon",
        "labels" : [
            "scanning","Reconnaissance",
            "Recon-PortScan","Recon-OSScan",
            "Recon-PingSweep","Recon-HostDiscovery",
            "VulnerabilityScan","Service_Scan",
            "OS_Fingerprint",
        ]
    },

    "web_attack": {
        "level1" : "attack",
        "level2" : "web_attack",
        "labels" : [
            "xss","XSS","injection",
            "SQL Injection","CommandInjection",
            "FileUpload","LDAP Injection",
            "Exploits",
            "Generic",
            "Fuzzers",
        ]
    },

    "malware": {
        "level1" : "attack",
        "level2" : "malware",
        "labels" : [
            "backdoor","Backdoor",
            "ransomware","Ransomware",
            "Trojan","Worm",
            "Shellcode",
            "Worms",
        ]
    },

    "botnet": {
        "level1" : "attack",
        "level2" : "botnet",
        "labels" : [
            "Mirai-greeth_flood",
            "Mirai-greip_flood",
            "Mirai-udpplain",
            "Mirai","IRCBot","Botnet","BotNet",
        ]
    },

    "credential": {
        "level1" : "attack",
        "level2" : "credential",
        "labels" : [
            "password","Password",
            "DictionaryBruteForce",
            "BruteForce","bruteforce",
            "Brute Force",
        ]
    },

    "mitm": {
        "level1" : "attack",
        "level2" : "mitm",
        "labels" : [
            "mitm","MITM",
            "MITM-ArpSpoofing",
            "DNS_Spoofing","ARP_Spoofing",
        ]
    },

    "exfiltration": {
        "level1" : "attack",
        "level2" : "exfiltration",
        "labels" : [
            "Theft","Data_Exfiltration",
            "Keylogging","exfiltration",
            "Analysis",
            "Backdoors",
            "Shellcode",
        ]
    },

    "unknown": {
        "level1" : "attack",
        "level2" : "unknown",
        "labels" : []
    },
}

# ── Lookup table للسرعة ───────────────────────────────────────
_LABEL_TO_CATEGORY = {}
for _cat, _info in TAXONOMY.items():
    for _lbl in _info["labels"]:
        _LABEL_TO_CATEGORY[_lbl.lower()] = _cat

LEVEL2_CLASSES = [
    k for k in TAXONOMY.keys() if k != "unknown"]

LEVEL1_CLASSES = ["benign", "attack"]


def get_level1(label: str) -> str:
    cat = _LABEL_TO_CATEGORY.get(
        label.lower(), "unknown")
    return TAXONOMY[cat]["level1"]


def get_level2(label: str) -> str:
    return _LABEL_TO_CATEGORY.get(
        label.lower(), "unknown")


def unify_label(label: str) -> str:
    return get_level2(label)


def unify_series(series) -> "pd.Series":
    import pandas as pd
    return series.apply(
        lambda x: unify_label(str(x)))


def get_attack_families() -> list:
    return [k for k, v in TAXONOMY.items()
            if v["level1"] == "attack"
            and k != "unknown"]


def is_benign(label: str) -> bool:
    return get_level1(label) == "benign"


if __name__ == "__main__":
    tests = [
        "DDoS-UDP_Flood","normal",
        "Mirai-greip_flood","scanning",
        "XSS","BenignTraffic","NEW_ATTACK",
        "Exploits","Fuzzers","DoS",
        "Reconnaissance","Backdoor",
        "Brute Force","Analysis",]
    print(f"{'Label':<30} {'L1':<10} {'L2'}")
    print("-" * 55)
    for t in tests:
        print(f"{t:<30} "
              f"{get_level1(t):<10} "
              f"{get_level2(t)}")
