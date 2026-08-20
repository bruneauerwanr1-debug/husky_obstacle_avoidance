#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  run_camera_evaluation.sh — Lanceur Interactif du Banc de Test Caméra
# ════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod +x "${SCRIPT_DIR}/distortion_evaluator_node.py"
chmod +x "${SCRIPT_DIR}/homography_distortion_evaluator_node.py"
chmod +x "${SCRIPT_DIR}/camera_analysis.py"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BANC D'ÉVALUATION DISTORSION & CALIBRATION (INTEL REALSENSE)  "
echo "════════════════════════════════════════════════════════════════"
echo "  1) Lancer le Test A & B (Damier RMSE + Rectilinéarité Lignes)"
echo "  2) Lancer le Test D (Homographie Sol & AprilTags en 1280x720)"
echo "  3) Lancer la Synthèse Globale & Rapport (camera_analysis.py)"
echo "  4) Mode Hors-ligne (Images de test homography-imagetest)"
echo "════════════════════════════════════════════════════════════════"
echo ""

CHOICE="${1:-}"

if [ -z "$CHOICE" ]; then
    read -p "Sélectionnez une option (1, 2, 3 ou 4) : " CHOICE
fi

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

if [ "$CHOICE" == "1" ]; then
    echo "[INFO] Lancement des Tests A & B..."
    python3 "${SCRIPT_DIR}/distortion_evaluator_node.py" --image-topic /camera/color/image_raw --info-topic /camera/color/camera_info

elif [ "$CHOICE" == "2" ]; then
    echo "[INFO] Lancement du Test D (Homographie)..."
    echo "[CONSEIL] Pour des résultats optimaux, lancez le flux RealSense en 1280x720x15 :"
    echo "          ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:=1280x720x15"
    python3 "${SCRIPT_DIR}/homography_distortion_evaluator_node.py" --image-topic /camera/color/image_raw --info-topic /camera/color/camera_info

elif [ "$CHOICE" == "3" ]; then
    echo "[INFO] Génération de la Synthèse Globale..."
    python3 "${SCRIPT_DIR}/camera_analysis.py" --results-dir "${SCRIPT_DIR}/results"

elif [ "$CHOICE" == "4" ]; then
    TEST_DIR="${SCRIPT_DIR}/../../homography/homography-imagetest"
    echo "[INFO] Lancement Hors-ligne sur ${TEST_DIR}..."
    python3 "${SCRIPT_DIR}/homography_distortion_evaluator_node.py" --offline --images "${TEST_DIR}"

else
    echo "Option invalide. Choisissez 1, 2, 3 ou 4."
fi
