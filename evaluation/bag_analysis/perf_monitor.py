#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_resources.py — Journalisation CPU / RAM / GPU / VRAM pendant un essai
================================================================================
Indépendant de ROS : à lancer dans un terminal séparé, EN MÊME TEMPS que
`ros2 bag record` (démarrage et arrêt synchronisés à la main -- ce script
n'a pas besoin d'être précis à la milliseconde, juste de couvrir toute
la durée de l'essai). Sert à comparer objectivement le coût de calcul
des 3 architectures (YOLO/RAFT = GPU, Farneback = CPU pur), demande
initiale : "avoir une idée de ce que les codes consomment".

USAGE
-----
    python3 monitor_resources.py
    # (chemin de sortie et interval définis en dur dans CONFIG ci-dessous)
    # ... lancer ros2 bag record dans un autre terminal, faire l'essai ...
    # Ctrl+C ici pour arrêter (juste après avoir arrêté le bag record)

    Override ponctuel possible sans toucher au code :
    python3 monitor_resources.py --out /chemin/autre.csv --interval 1.0

    Puis, lors de l'analyse :
    python3 analyze_bag.py bag_raft_S1.3_02 --scenario S1.3 --trial 2 \\
        --success 1 --resource-log resource_raft_S1.3_02.csv \\
        --master ./results_master.csv

DÉPENDANCES
-----------
    pip install psutil --break-system-packages
    GPU : utilise `nvidia-smi` s'il est présent sur la machine (laptop
    GTX 1070). Si absent, les colonnes GPU restent vides -- pas d'erreur,
    utile pour les essais Farneback (CPU pur, aucune attente GPU requise).
"""

import argparse
import csv
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    print("[ERREUR] psutil manquant : pip install psutil --break-system-packages",
          file=sys.stderr)
    raise


# ============================================================================
# CONFIG EN DUR — à éditer avant chaque essai
# ============================================================================
BASE_DIR = "/media/imr2204/bd37914b-8e04-4d06-b568-4a7cd46f37ab/home/imr/Erwan/clearpath_simulator/src/bag-eval"

NODE = "yolo"          # "yolo" | "raft" | "farneback"
SCENARIO = "S1"         # "S1", "S1.3", etc.
TRIAL = "01"            # "01", "02", ...

DEFAULT_OUT = f"{BASE_DIR}/{NODE}/{SCENARIO}/resource_{NODE}_{SCENARIO}_{TRIAL}.csv"
DEFAULT_INTERVAL = 0.5  # secondes
# ============================================================================


def read_gpu():
    """Retourne (util_%, mem_used_MB, mem_total_MB) ou (None, None, None)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=1.0, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        if not out:
            return None, None, None
        gpu_pct, mem_used, mem_total = [float(x) for x in out[0].split(",")]
        return gpu_pct, mem_used, mem_total
    except Exception:
        return None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT,
                     help=f"Fichier CSV de sortie (défaut : {DEFAULT_OUT})")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                     help=f"Période d'échantillonnage en secondes (défaut {DEFAULT_INTERVAL}s)")
    args = ap.parse_args()

    gpu_ok = read_gpu()[0] is not None
    print(f"[+] Monitoring GPU : {'actif (nvidia-smi détecté)' if gpu_ok else 'indisponible (colonnes GPU vides)'}")
    print(f"[+] Écriture dans {args.out} toutes les {args.interval}s")
    print("[+] Démarrez ros2 bag record MAINTENANT dans un autre terminal.")
    print("[+] Ctrl+C ici pour arrêter (juste après avoir arrêté le bag record).")

    psutil.cpu_percent()  # premier appel = référence (toujours 0.0), à ignorer

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_unix", "cpu_percent", "ram_percent", "ram_used_mb",
                     "gpu_percent", "gpu_mem_used_mb", "gpu_mem_total_mb"])
        n = 0
        try:
            while True:
                loop_start = time.time()
                cpu = psutil.cpu_percent()  # % moyen tous coeurs depuis l'appel précédent
                vm = psutil.virtual_memory()
                gpu_pct, gpu_mem, gpu_total = read_gpu()
                w.writerow([loop_start, cpu, vm.percent, vm.used / 1e6,
                            gpu_pct, gpu_mem, gpu_total])
                f.flush()
                n += 1
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, args.interval - elapsed))
        except KeyboardInterrupt:
            print(f"\n[+] Arrêt — {n} échantillons écrits dans {args.out}")


if __name__ == "__main__":
    main()