# Rapport Scientifique Consolidé — Évaluation Distorsion & Calibration Caméra

**Date d'analyse** : 2026-08-20 14:18:41  
**Capteur** : Intel RealSense D435i (Flux Couleur RGB)  
**Périmètre** : Qualité des intrinsèques, rectilinéarité des lignes et homographie au sol  

---

## 1. Synthèse Globale des Résultats

| Test | Indicateur Clé | Résultat Mesuré | Seuil de Tolérance | Verdict Scientifique |
| :--- | :--- | :---: | :---: | :---: |
| **Test A — RMSE Damier** | Erreur de reprojection subpixel | **0.274 px** (39 vues) | < 0.30 px | 🟢 **EXCELLENTE** |
| **Test A — Dérive Focale** | Écart $f_x, f_y$ (Usine vs Calibré) | **< 0.2%** ($f_x=615.6, f_y=616.9$) | < 2.0% | 🟢 **PARFAITE STABILITÉ** |
| **Test B — Rectilinéarité** | Déviation moyenne sur arête brute | **0.61 px** | <= 1.0 px | 🟢 **CONFORME BRUTE** |
| **Test C — Intrinsèques** | Modèle de distorsion firmware | $D = [0, 0, 0, 0, 0]$ | — | 🟢 **CORRECTION USINE ACTIVE** |
| **Test D — Homographie Sol** | Erreur métrique sol brute ($H_{raw}$, 6 tags) | **2.96 mm** (0.296 cm) | < 20.0 mm | 🟢 **ULTRA-PRÉCISE** |
| **Test D — Gain Redressement** | Gain $H_{undist}$ vs $H_{raw}$ | **+2.29%** ($\Delta = 0.07\text{ mm}$) | > 10.0% | ⚪ **GAIN NÉGLIGEABLE** |

---

## 2. Recommandations et Décisions pour le Rapport de Stage

1. **État de la caméra** : La caméra Intel RealSense D435i est en parfait état optique (RMSE < 0.3 px). Aucune recalibration matérielle n'est requise.
2. **Optimisation temps réel** : Ne pas appliquer de correction logicielle de distorsion (`cv2.undistort`) en temps réel est **totalement justifié** (gain métrique < 0.1 mm, économie de CPU/GPU).
3. **Homographie validée** : L'homographie $H_{new}$ atteint une précision sub-centimétrique ($< 3\text{ mm}$) sur l'ensemble du champ de vision utile.
