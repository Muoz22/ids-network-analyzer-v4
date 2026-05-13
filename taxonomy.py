# ── كتابة taxonomy.py مباشرة ──────────────────────────────────
taxonomy_code = '''
TAXONOMY = {
    "benign": {
        "level1": "benign", "level2": "benign",
        "labels": [
            "normal","Normal","BENIGN","benign",
            "Benign","BenignTraffic","legitimate",
            "background","Background","safe","Safe",
            "none","None",
        ]
    },
    "ddos": {
        "level1": "attack", "level2": "ddos",
        "labels": [
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
        "level1": "attack", "level2": "dos",
        "labels": [
            "dos","DoS",
            "DoS-SYN_Flood","DoS-UDP_Flood",
            "DoS-TCP_Flood","DoS-HTTP_Flood",
        ]
    },
    "recon": {
        "level1": "attack", "level2": "recon",
        "labels": [
            "scanning","Reconnaissance",
            "Recon-PortScan","Recon-OSScan",
            "Recon-PingSweep","Recon-HostDiscovery",
            "VulnerabilityScan","Service_Scan",
            "OS_Fingerprint",
        ]
    },
    "web_attack": {
        "level1": "attack", "level2": "web_attack",
        "labels": [
            "xss","XSS","injection",
            "SQL Injection","CommandInjection",
            "FileUpload","LDAP Injection",
            "Exploits","Generic","Fuzzers",
        ]
    },
    "malware": {
        "level1": "attack", "level2": "malware",
        "labels": [
            "backdoor","Backdoor",
            "ransomware","Ransomware",
            "Trojan","Worm","Shellcode","Worms",
        ]
    },
    "botnet": {
        "level1": "attack", "level2": "botnet",
        "labels": [
            "Mirai-greeth_flood","Mirai-greip_flood",
            "Mirai-udpplain","Mirai",
            "IRCBot","Botnet","BotNet",
        ]
    },
    "credential": {
        "level1": "attack", "level2": "credential",
        "labels": [
            "password","Password",
            "DictionaryBruteForce",
            "BruteForce","bruteforce",
            "Brute Force",
        ]
    },
    "mitm": {
        "level1": "attack", "level2": "mitm",
        "labels": [
            "mitm","MITM","MITM-ArpSpoofing",
            "DNS_Spoofing","ARP_Spoofing",
        ]
    },
    "exfiltration": {
        "level1": "attack", "level2": "exfiltration",
        "labels": [
            "Theft","Data_Exfiltration",
            "Keylogging","exfiltration",
            "Analysis","Backdoors","Shellcode",
        ]
    },
    "unknown": {
        "level1": "attack", "level2": "unknown",
        "labels": []
    },
}

_LABEL_TO_CATEGORY = {}
for _cat, _info in TAXONOMY.items():
    for _lbl in _info["labels"]:
        _LABEL_TO_CATEGORY[_lbl.lower()] = _cat

LEVEL2_CLASSES = [k for k in TAXONOMY.keys()
                  if k != "unknown"]
LEVEL1_CLASSES = ["benign", "attack"]

def get_level1(label):
    cat = _LABEL_TO_CATEGORY.get(
        label.lower(), "unknown")
    return TAXONOMY[cat]["level1"]

def get_level2(label):
    return _LABEL_TO_CATEGORY.get(
        label.lower(), "unknown")

def unify_label(label):
    return get_level2(label)

def unify_series(series):
    import pandas as pd
    return series.apply(
        lambda x: unify_label(str(x)))

def get_attack_families():
    return [k for k, v in TAXONOMY.items()
            if v["level1"] == "attack"
            and k != "unknown"]

def is_benign(label):
    return get_level1(label) == "benign"
'''

# اكتب الملف
with open("/kaggle/working/taxonomy.py", "w") as f:
    f.write(taxonomy_code)

# احذف من cache
for mod in list(sys.modules.keys()):
    if "taxonomy" in mod:
        del sys.modules[mod]

# استيراد مباشر
import importlib.util
spec = importlib.util.spec_from_file_location(
    "taxonomy", "/kaggle/working/taxonomy.py")
tax = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tax)

# اختبار
tests = ["Exploits","Fuzzers","Analysis",
         "Worms","DoS","Reconnaissance",
         "normal","DDoS-UDP_Flood"]
print(f"{'Label':<25} {'L2'}")
print("-" * 35)
for t in tests:
    print(f"{t:<25} → {tax.get_level2(t)}")
