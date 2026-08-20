#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  run_benchmark_analysis.sh — Analyse Statistique & Génération de Graphiques
# ════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_homography_benchmark.py"

chmod +x "${ANALYSIS_SCRIPT}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ANALYSE STATISTIQUE DU BENCHMARK D'HOMOGRAPHIE (H_old vs H_new)"
echo "════════════════════════════════════════════════════════════════"
echo "  1) Analyser le fichier CSV le plus récent"
echo "  2) Spécifier un fichier CSV particulier"
echo "════════════════════════════════════════════════════════════════"
echo ""

CHOICE="${1:-}"

if [ -z "$CHOICE" ]; then
    read -p "Sélectionnez une option (1 ou 2) : " CHOICE
fi

if [ "$CHOICE" == "1" ]; then
    echo "[INFO] Analyse du fichier le plus récent..."
    python3 "${ANALYSIS_SCRIPT}"

elif [ "$CHOICE" == "2" ]; then
    read -p "Chemin vers le fichier CSV : " CSV_PATH
    python3 "${ANALYSIS_SCRIPT}" --csv "${CSV_PATH}"

else
    echo "Option invalide."
fi
