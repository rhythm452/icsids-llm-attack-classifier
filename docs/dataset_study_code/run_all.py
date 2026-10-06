"""Reproduce the whole study: python dataset_study/run_all.py

Every stage saves its outputs and skips work already done, so a crash can be
resumed by re-running the same command. Delete dataset_study/cache and
dataset_study/results to start from scratch.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STAGES = [
    ("step0_profile.py", [ROOT]),   # Step 0: raw-file profile
    ("prepare.py", []),             # Step 1: clean / dedup / sample / split
    ("train_eval.py", []),          # Steps 1-2: models + metrics
    ("batadal_lowo.py", []),        # BATADAL leave-one-window-out
    ("sanity.py", []),              # Step 3
    ("cross.py", []),               # Step 4 (also caches the embedding samples)
    ("zeroshot.py", []),            # project zero-shot method (needs webapp/)
    ("report.py", []),              # Step 5
]

if __name__ == "__main__":
    prep_args = ["BATADAL", "Edge-IIoTset", "X-IIoTID", "TON_IoT-Network", "TON_IoT-Modbus",
                 "TON_IoT-IoT_Fridge", "TON_IoT-IoT_Garage_Door", "TON_IoT-IoT_GPS_Tracker",
                 "TON_IoT-IoT_Motion_Light", "TON_IoT-IoT_Thermostat", "TON_IoT-IoT_Weather",
                 "Edge-IIoTset@nodedup", "X-IIoTID@nodedup", "TON_IoT-Network@nodedup", "TON_IoT-Modbus@nodedup"]
    for script, args in STAGES:
        args = prep_args if script == "prepare.py" else args
        print(f"=== {script}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args, cwd=HERE)
        if r.returncode:
            sys.exit(f"stage {script} failed (exit {r.returncode}); fix and re-run to resume")
