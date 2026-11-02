"""Config-driven loaders: one spec per dataset, one shared cleaning routine.

Every loader returns (df, meta). df holds the kept feature columns plus:
  __y      binary label (0 normal, 1 attack)
  __type   finest attack-type label (stratification + multiclass)
  __multi  multiclass target (None if the dataset has no attack types)
  __time   timestamp (chronological datasets only; never a feature)
"""
import os

import numpy as np
import pandas as pd

from common import ROOT

TON = "data/extracted/ton_iot/Train_Test_datasets"

R_IP = "IP address: host identifier, lets the model memorise testbed hosts"
R_PORT = "port number: ephemeral/testbed-specific identifier"
R_TIME = "timestamp/date: identifies capture session, not behaviour"
R_SEQ = "per-packet sequence/ack/checksum/stream counter: identifier, not behaviour"
R_TEXT = "raw payload/URI/free text: high-cardinality, memorisable"
R_HOST = "host/domain/certificate/user-agent string: identifies testbed endpoints"
R_SVC = "raw client/server port: replaced by derived service_port (the lower, server-side port)"
R_SUM = "raw string: replaced by length (+ injection-character count for URIs/queries) summaries"

SPECS = {
    "BATADAL": dict(
        kind="process/sensor telemetry (water distribution SCADA, hourly)",
        split="chronological", text_prefix="Water distribution SCADA hourly reading",
        drop={"DATETIME": R_TIME + " (used only to order rows for the chronological split)"},
        categorical=[], multiclass=None),
    "Edge-IIoTset": dict(
        kind="network traffic (packet-level, incl. Modbus/MQTT/HTTP/DNS fields)",
        split="stratified", text_prefix="IIoT network packet",
        path="data/extracted/edge_iiotset/DNN-EdgeIIoT-dataset.csv",
        label="Attack_label", type_col="Attack_type", multiclass="Attack_type",
        drop={
            "frame.time": R_TIME + "; also column-misaligned for all DDoS_UDP/MITM rows",
            "ip.src_host": R_IP, "ip.dst_host": R_IP,
            "arp.src.proto_ipv4": R_IP, "arp.dst.proto_ipv4": R_IP,
            "tcp.srcport": R_SVC, "tcp.dstport": R_SVC,
            "tcp.ack": R_SEQ, "tcp.ack_raw": R_SEQ, "tcp.seq": R_SEQ, "tcp.checksum": R_SEQ,
            "icmp.checksum": R_SEQ, "icmp.seq_le": R_SEQ, "icmp.transmit_timestamp": R_SEQ,
            "udp.stream": R_SEQ, "mbtcp.trans_id": R_SEQ,
            "tcp.options": "raw TCP options bytes: embed per-packet TCP timestamp values (counter)",
            "tcp.payload": R_SUM, "http.file_data": R_SUM,
            "http.request.full_uri": R_SUM, "http.request.uri.query": R_SUM,
            "mqtt.msg": R_SUM, "dns.qry.name": R_SUM,
            "dns.qry.name.len": "header-misaligned in source (holds hostnames); its length is kept as len_dns_name",
        },
        derive="edge",
        categorical=["http.request.method", "http.referer", "http.request.version",
                     "mqtt.protoname", "mqtt.topic", "mqtt.conack.flags"]),
    "X-IIoTID": dict(
        kind="network flow + host telemetry (+ OSSEC/host alerts)",
        split="stratified", text_prefix="IIoT network flow with host statistics",
        path="data/extracted/x_iiotid/X-IIoTID dataset.csv",
        label="class3", type_col="class1", multiclass="class2",
        extra_labels=["class1", "class2", "class3"],
        drop={"Date": R_TIME, "Timestamp": R_TIME, "Scr_IP": R_IP, "Des_IP": R_IP,
              "Scr_port": R_SVC, "Des_port": R_SVC},
        categorical=["Protocol", "Service"], derive="xiiotid"),
    "TON_IoT-Network": dict(
        kind="network traffic (Zeek flows)",
        split="stratified", text_prefix="Network connection log",
        path=f"{TON}/Train_Test_Network_dataset/train_test_network.csv",
        label="label", type_col="type", multiclass="type",
        drop={"src_ip": R_IP, "dst_ip": R_IP, "src_port": R_SVC, "dst_port": R_SVC,
              "dns_query": R_SUM, "ssl_subject": R_SUM, "ssl_issuer": R_SUM,
              "http_uri": R_SUM, "http_user_agent": R_SUM},
        derive="tonnet",
        categorical=["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
                     "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "http_method",
                     "http_version", "http_orig_mime_types", "http_resp_mime_types", "weird_name",
                     "weird_addl", "weird_notice"]),
}

_IOT_DROP = {"date": R_TIME, "time": R_TIME + "; leading-space format differs by label (leak)"}
_IOT_CAT = {"Fridge": ["temp_condition"], "Garage_Door": ["door_state"], "Motion_Light": ["light_status"]}
for dev in ["Modbus", "Fridge", "Garage_Door", "GPS_Tracker", "Motion_Light", "Thermostat", "Weather"]:
    name = "TON_IoT-Modbus" if dev == "Modbus" else f"TON_IoT-IoT_{dev}"
    SPECS[name] = dict(
        kind="IoT/IIoT device telemetry" + (" (Modbus function-code register values)" if dev == "Modbus" else ""),
        split="stratified", text_prefix=f"IIoT {dev.replace('_', ' ')} telemetry",
        path=f"{TON}/Train_Test_IoT_dataset/Train_Test_IoT_{dev}.csv",
        label="label", type_col="type", multiclass="type",
        drop=dict(_IOT_DROP), categorical=_IOT_CAT.get(dev, []))

BOOL_MAP = {"true": 1.0, "false": 0.0, "t": 1.0, "f": 0.0}
PLACEHOLDERS = {"", "-", "nan", "none", "null", "na"}


def clean_chunk(raw, spec, label_cols):
    """Strip, drop identifiers, coerce numerics, normalise categoricals."""
    raw = raw.rename(columns=lambda c: c.strip())
    out = {}
    stats = {"non_numeric_coerced": {}}
    for c in raw.columns:
        if c in spec["drop"] or c in label_cols:
            continue
        s = raw[c].astype(str).str.strip()
        low = s.str.lower()
        if c in spec["categorical"]:
            # lower+strip and canonical numbers remove formatting differences
            # (e.g. ' off' vs 'off', '0' vs '0.0') that could leak the label
            asnum = pd.to_numeric(low, errors="coerce")
            low = low.where(asnum.isna(), asnum.map(lambda x: f"{x:g}"))
            out[c] = low.where(~low.isin(PLACEHOLDERS), "na")
        else:
            v = low.map(BOOL_MAP)
            num = pd.to_numeric(low.where(v.isna(), np.nan), errors="coerce")
            num = num.fillna(v)
            bad = num.isna() & ~low.isin(PLACEHOLDERS)
            if bad.any():
                stats["non_numeric_coerced"][c] = int(bad.sum())
            num = num.replace([np.inf, -np.inf], np.nan)
            out[c] = num.astype("float32")
    return pd.DataFrame(out, index=raw.index), stats


# ---------------------------------------------------------------- derived (non-identifying) fields
INJ_CHARS = r"['\"<>%;]|\.\."
TEXT_MAX = 80  # chars of method+URI kept for the natural-language descriptions only


def _num(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce").fillna(0)


def service_port(a, b):
    """Server-side port of a connection: the lower non-zero of the two ports."""
    a, b = _num(a).to_numpy(), _num(b).to_numpy()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.where(lo > 0, lo, hi).astype("float32")


def _empty(s):
    s = s.astype(str).str.strip()
    return s, s.str.lower().isin(PLACEHOLDERS) | pd.to_numeric(s, errors="coerce").eq(0)


def strlen(s):
    s, e = _empty(s)
    return s.str.len().where(~e, 0).astype("float32").to_numpy()


def injcount(s):
    s, e = _empty(s)
    return s.str.count(INJ_CHARS).where(~e, 0).astype("float32").to_numpy()


def http_text(method, uri):
    m, me = _empty(method)
    u, ue = _empty(uri)
    t = (m.where(~me, "") + " " + u.where(~ue, "").str.slice(0, TEXT_MAX)).str.strip()
    return t.to_numpy()


def derive(kind, raw):
    """Returns (derived feature frame, text-only columns). Text columns start with '__txt'."""
    d = {}
    if kind == "edge":
        tcp_port = service_port(raw["tcp.srcport"], raw["tcp.dstport"])
        # DDoS_UDP/MITM rows are column-misaligned in the source (frame.time holds an IP or
        # number): their tcp port columns hold shifted values, so treat them as non-TCP.
        ft = raw["frame.time"].str.strip()
        misaligned = ~(ft.str.match(r"^\d{4} \d\d:\d\d") | ft.str.match(r"^[A-Za-z]{3} "))
        d["service_port"] = np.where(misaligned.to_numpy(), 0, tcp_port).astype("float32")
        dns_name = raw["dns.qry.name.len"]  # holds the hostname in this release
        d.update({"len_tcp_payload": strlen(raw["tcp.payload"]) / 2,  # hex -> bytes
                  "len_http_file_data": strlen(raw["http.file_data"]),
                  "len_http_uri": strlen(raw["http.request.full_uri"]),
                  "len_http_query": strlen(raw["http.request.uri.query"]),
                  "inj_http_uri": injcount(raw["http.request.full_uri"]),
                  "inj_http_query": injcount(raw["http.request.uri.query"]),
                  "len_mqtt_msg": strlen(raw["mqtt.msg"]) / 2,
                  "len_dns_name": strlen(dns_name.where(pd.to_numeric(dns_name, errors="coerce").isna(), "0"))})
        txt = {"__txt_http": http_text(raw["http.request.method"], raw["http.request.full_uri"])}
    elif kind == "xiiotid":
        d["service_port"] = service_port(raw["Scr_port"], raw["Des_port"])
        txt = {}
    elif kind == "tonnet":
        d.update({"service_port": service_port(raw["src_port"], raw["dst_port"]),
                  "len_dns_query": strlen(raw["dns_query"]), "len_http_uri": strlen(raw["http_uri"]),
                  "inj_http_uri": injcount(raw["http_uri"]),
                  "len_http_user_agent": strlen(raw["http_user_agent"]),
                  "len_ssl_subject": strlen(raw["ssl_subject"]), "len_ssl_issuer": strlen(raw["ssl_issuer"])})
        txt = {"__txt_http": http_text(raw["http_method"], raw["http_uri"])}
    else:
        return pd.DataFrame(index=raw.index), {}
    return pd.DataFrame(d, index=raw.index), txt


DERIVED_DOC = {
    "service_port": "lower non-zero of the two port numbers (server/service side); client ephemeral port discarded",
    "len_*": "character length of the raw string (bytes for hex payloads); 0 when absent",
    "inj_*": "count of injection-typical characters ' \" < > % ; and '..' in the URI/query",
    "__txt_http": "HTTP method + first 80 chars of URI: used ONLY in the text descriptions, never as a tabular feature",
}


def _merge_stats(a, b):
    for k, v in b["non_numeric_coerced"].items():
        a["non_numeric_coerced"][k] = a["non_numeric_coerced"].get(k, 0) + v


def load_csv_dataset(name):
    spec = SPECS[name]
    labels = [spec["label"], spec["type_col"]] + spec.get("extra_labels", [])
    labels = list(dict.fromkeys(labels))
    parts, stats, n_raw = [], {"non_numeric_coerced": {}}, 0
    for raw in pd.read_csv(os.path.join(ROOT, spec["path"]), chunksize=250_000, dtype=str,
                           keep_default_na=False, encoding="utf-8-sig", low_memory=False):
        raw.columns = [c.strip() for c in raw.columns]
        feats, st = clean_chunk(raw, spec, labels)
        _merge_stats(stats, st)
        der, txt = derive(spec.get("derive"), raw)
        feats = pd.concat([feats, der], axis=1)
        for k, v in txt.items():
            feats[k] = v
        y = pd.to_numeric(raw[spec["label"]].str.strip().replace({"Attack": "1", "Normal": "0"}),
                          errors="coerce")
        feats["__y"] = y.astype("int8").values
        feats["__type"] = raw[spec["type_col"]].str.strip().values
        feats["__multi"] = raw[spec["multiclass"]].str.strip().values
        if name == "X-IIoTID":
            feats["__class1"] = raw["class1"].str.strip().values
        parts.append(feats)
        n_raw += len(raw)
    df = pd.concat(parts, ignore_index=True)
    # Garage_Door: sphone_signal uses 'false'/'true' for normal rows and '0'/'1' for
    # attacks; BOOL_MAP has already normalised both to 0/1.
    meta = {"rows_raw": n_raw, "coercion": stats}
    return df, meta


def load_batadal():
    spec = SPECS["BATADAL"]
    frames = []
    for fname, part in [("BATADAL_dataset03.csv", "dataset03"), ("BATADAL_dataset04.csv", "dataset04")]:
        d = pd.read_csv(os.path.join(ROOT, "data", "raw", fname), skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        d["__part"] = part
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    feats = [c for c in d.columns if c not in ("DATETIME", "ATT_FLAG", "__part")]
    # dataset04 is published with 2 decimals; round dataset03 to match so the
    # model cannot identify the source file (and hence the label) by precision.
    out = d[feats].astype("float64").round(2).astype("float32")
    out["__time"] = pd.to_datetime(d["DATETIME"], format="%d/%m/%y %H")
    flag = d["ATT_FLAG"].astype(int)
    out["__y"] = (flag == 1).astype("int8")
    out["__type"] = np.where(flag == 1, "attack", "normal")
    out["__multi"] = None
    out["__part"] = d["__part"]
    meta = {"rows_raw": len(d), "coercion": {"non_numeric_coerced": {}},
            "label_noise": "ATT_FLAG=-999 in dataset04 (%d rows) treated as normal; the BATADAL "
                           "organisers left some dataset04 attacks unlabelled, so some of these rows "
                           "may be attacks." % int((flag == -999).sum())}
    return out, meta


def load(name):
    if name == "BATADAL":
        return load_batadal()
    return load_csv_dataset(name)


# BATADAL chronological split (approved): attack windows kept whole.
BATADAL_SPLIT = {"val_start": "2016-10-20", "test_start": "2016-11-15"}
# Leave-one-attack-window-out: dataset04 cut at midpoints between the 5 labelled windows.
BATADAL_LOWO_CUTS = ["2016-09-27", "2016-10-21", "2016-11-14", "2016-12-02"]
