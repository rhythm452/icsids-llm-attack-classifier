"""Analyst mapping of each dataset's attack labels to MITRE ATT&CK for ICS techniques.

Nearest-technique mapping by the study author (not from the dataset publishers);
review before using it as ground truth. IT-only attacks (e.g. SQLi/XSS on a web
server) map only loosely to ICS techniques and are marked 'loose'.
"""

T = {
    "T0801": "Monitor Process State", "T0802": "Automated Collection", "T0807": "Command-Line Interface",
    "T0809": "Data Destruction", "T0814": "Denial of Service", "T0819": "Exploit Public-Facing Application",
    "T0828": "Loss of Productivity and Revenue", "T0830": "Adversary-in-the-Middle",
    "T0831": "Manipulation of Control", "T0832": "Manipulation of View", "T0836": "Modify Parameter",
    "T0840": "Network Connection Enumeration", "T0842": "Network Sniffing", "T0846": "Remote System Discovery",
    "T0855": "Unauthorized Command Message", "T0856": "Spoof Reporting Message",
    "T0859": "Valid Accounts", "T0861": "Point & Tag Identification", "T0866": "Exploitation of Remote Services",
    "T0869": "Standard Application Layer Protocol", "T0882": "Theft of Operational Information",
    "T0884": "Connection Proxy", "T0886": "Remote Services", "T0888": "Remote System Information Discovery",
    "T0891": "Hardcoded Credentials",
}

MAP = {
    "BATADAL": {
        "attack": (["T0831", "T0832", "T0836", "T0855", "T0856"], "direct",
                   "sensor/actuator manipulation and concealment (tank levels, pump/valve settings)")},
    "Edge-IIoTset": {
        "DDoS_UDP": (["T0814"], "direct", ""), "DDoS_ICMP": (["T0814"], "direct", ""),
        "DDoS_TCP": (["T0814"], "direct", ""), "DDoS_HTTP": (["T0814"], "direct", ""),
        "Port_Scanning": (["T0846", "T0840"], "direct", ""), "Fingerprinting": (["T0888"], "direct", ""),
        "Vulnerability_scanner": (["T0846", "T0888"], "direct", ""),
        "Password": (["T0859"], "loose", "credential brute force; ICS matrix has no generic brute-force technique"),
        "SQL_injection": (["T0819"], "loose", "web-app attack"), "XSS": (["T0819"], "loose", "web-app attack"),
        "Uploading": (["T0819"], "loose", "malicious file upload to web app"),
        "Backdoor": (["T0869", "T0886"], "loose", ""), "Ransomware": (["T0809", "T0828"], "direct", ""),
        "MITM": (["T0830"], "direct", "ARP spoofing"),
    },
    "X-IIoTID": {  # class2 (stage) with the class1 types it contains
        "Reconnaissance": (["T0846", "T0840", "T0888"], "direct", "Generic_scanning, Scanning_vulnerability, Discovering_resources, fuzzing"),
        "Weaponization": (["T0859", "T0891"], "loose", "BruteForce, Dictionary, insider_malcious"),
        "Exploitation": (["T0866", "T0807"], "direct", "Reverse_shell"),
        "Lateral _movement": (["T0886", "T0861", "T0802"], "direct", "MQTT_cloud_broker_subscription, Modbus_register_reading, TCP Relay"),
        "C&C": (["T0869", "T0884"], "direct", ""),
        "Exfiltration": (["T0882"], "direct", ""),
        "Tampering": (["T0856", "T0832", "T0836"], "direct", "False_data_injection, Fake_notification"),
        "RDOS": (["T0814"], "direct", ""), "crypto-ransomware": (["T0809", "T0828"], "direct", ""),
    },
    "TON_IoT-Network": {
        "scanning": (["T0846", "T0840"], "direct", ""), "dos": (["T0814"], "direct", ""),
        "ddos": (["T0814"], "direct", ""), "injection": (["T0819"], "loose", "web/data injection"),
        "xss": (["T0819"], "loose", ""), "password": (["T0859"], "loose", ""),
        "backdoor": (["T0869", "T0886"], "loose", ""), "ransomware": (["T0809", "T0828"], "direct", ""),
        "mitm": (["T0830"], "direct", ""),
    },
    "TON_IoT-Modbus": {
        "injection": (["T0855", "T0836", "T0856"], "direct", "data injection into Modbus register values"),
        "backdoor": (["T0869", "T0886"], "loose", ""), "password": (["T0859"], "loose", ""),
        "xss": (["T0819"], "loose", ""), "scanning": (["T0846", "T0861"], "direct", ""),
    },
}
