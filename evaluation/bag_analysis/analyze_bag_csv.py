#!/usr/bin/env python3
"""
analyze_bag_csv.py — Analyse et visualisation des CSV extraits par extract_bag.py

Génère, dans un dossier de sortie, les graphiques clés pour diagnostiquer
le comportement de emergency_stop_node.py à partir d'une session de test :

  01_commands_overlay.png   → linear_x et angular_z des 4 topics Twist
                               superposés, avec les périodes EMERGENCY
                               ombrées en rouge. Vue d'ensemble de la
                               session.
  02_obstacle_distance.png  → distance au plus proche obstacle détecté
                               (/ground_detections) dans le temps, avec
                               les seuils DANGER/WARNING du code, et les
                               périodes EMERGENCY ombrées.
  03_watchdog_latency.png   → écarts temporels (dt) entre messages
                               consécutifs sur /joy_teleop/cmd_vel et
                               /cmd_vel_in. Tout dt > CMD_SOURCE_TIMEOUT
                               (0.3s) signale un silence de commande —
                               donc le watchdog aurait dû agir.
  04_avoidance_episodes.png → chronologie des épisodes d'évitement
                               (/avoidance_cmd_vel), durée de chacun,
                               et nombre de changements de signe de
                               angular_z par épisode (indicateur de
                               saccade malgré l'hystérésis de côté).
  05_state_timeline.png     → vue synthétique : état EMERGENCY (rouge),
                               évitement actif (orange), distance min
                               obstacle, sur un seul axe temporel partagé.
  06_distance_calibration.png → distance RÉELLE (ruban, saisie manuelle
                               dans ground_truth_distances.csv) vs distance
                               CALCULÉE par le pipeline homographie+YOLO
                               (/ground_detections). Cœur de la validation
                               bird-eye view — sans ce fichier de vérité
                               terrain, ce graphique est sauté (une mesure
                               physique ne peut pas être déduite du bag).
  07_vo_side_consistency.png → cohérence du côté d'esquive choisi
                               (compute_vo_side) par rapport à la position
                               latérale (y_m) de l'obstacle déclencheur.
                               Proxy basé sur l'obstacle le plus proche —
                               fiable pour un seul obstacle statique à la
                               fois (voir limite documentée dans le code).
  08_yaw_excursions.png     → excursion de lacet réelle (odométrie) par
                               épisode d'évitement, comparée au plafond
                               MAX_AVOIDANCE_YAW_DEVIATION du code —
                               détecte une éventuelle fuite du plafond.
  09_perf_cpu_gpu.png       → CPU % et GPU % (si perf_monitor.py actif)
                               en fonction du temps, avec périodes EMERGENCY
                               ombrées — visualise la charge système pendant
                               un évitement et compare YOLO vs Farneback.
  10_latency_joy_cmd.png    → latence /joy_teleop/cmd_vel → /cmd_vel mesurée
                               par perf_monitor.py (/perf/joy_to_cmd_latency_ms),
                               et période d'inférence (/perf/inference_period_ms).
                               Indicateur direct de la réactivité du pipeline.
  11_farneback_flow.png     → métriques de flux optique Farneback par zone
                               (/optical_flow_zones) dans le temps — équivalent
                               de la distance YOLO pour la comparaison ; absent
                               si le bag vient d'une session YOLO.

  summary.txt                → résumé chiffré (texte) de tout ce qui
                               précède : nb d'arrêts d'urgence, durée
                               totale, nb de trous watchdog détectés
                               avec horodatage, nb d'épisodes d'évitement,
                               durée moyenne/max, distance minimale
                               atteinte et instant correspondant, biais/
                               RMSE de calibration distance, taux de
                               cohérence VO, dépassements de plafond de lacet.

FICHIER À CRÉER MANUELLEMENT (calibration bird-eye, optionnel) :
    ground_truth_distances.csv  — colonnes : label, real_distance_m,
                                  t_start_s, t_end_s
    Voir le protocole d'évaluation, section "Calibration distance
    réelle/calculée", pour la procédure de mesure au ruban et le gabarit.

USAGE :
    python3 analyze_bag_csv.py [/chemin/vers/dossier_csv] [--out ./analyse]

Le dossier d'entrée utilise par défaut le dossier du stage CVUT.
Les fichiers manquants sont simplement ignorés (le graphique correspondant
est sauté avec un message [info]).
"""

import argparse
import json
import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
#  Chemin cible par défaut (Modifiable ici en cas de nouveau dossier)
# ──────────────────────────────────────────────────────────────────────────
# Nom du dossier contenant les CSV extraits (modifiez ceci pour un nouveau dossier)
CSV_FOLDER_NAME = r"C:\Users\brune\OneDrive - IMT MINES ALES\Documents\Cours\2A Mines Alès\Stage CVUT\évalutation\bag record\test flux optique farneback\bag_extracted_test_full_farneback_20260708_105924"

# Le chemin cible par défaut se construit automatiquement à partir du dossier du script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DIR = os.path.join(BASE_DIR, CSV_FOLDER_NAME)

# ──────────────────────────────────────────────────────────────────────────
#  Constantes du nœud — reprises de emergency_stop_node.py pour que les
#  seuils tracés correspondent exactement au comportement réel attendu.
# ──────────────────────────────────────────────────────────────────────────
DISTANCE_WARN_M     = 1.5
DISTANCE_DANGER_M   = 0.8
CMD_SOURCE_TIMEOUT  = 0.3   # s — au-delà, le watchdog doit zéroter /cmd_vel
TTC_HORIZON         = 3.0   # s
MAX_AVOIDANCE_YAW_DEVIATION = 1.30   # rad ≈ 75° — plafond de déviation
                                       # cumulée pendant un évitement
                                       # (cf. emergency_stop_node.py)
ROBOT_HALF_WIDTH    = 0.40   # m — utilisé pour situer le "côté" d'esquive


# ──────────────────────────────────────────────────────────────────────────
#  Chargement
# ──────────────────────────────────────────────────────────────────────────

def load_twist(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values('t_rel_s').reset_index(drop=True)
    return df


def load_bool(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # Le CSV stocke "True"/"False" en texte
    df['data'] = df['data'].astype(str).str.strip().map({'True': True, 'False': False})
    df = df.sort_values('t_rel_s').reset_index(drop=True)
    return df


def load_ground_detections(path):
    """
    Parse la colonne JSON de /ground_detections et retourne un DataFrame
    avec, pour chaque message reçu : t_rel_s, dist_min_m (plus proche
    obstacle tous types confondus), nb_obstacles, classe de l'obstacle
    le plus proche.
    """
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    rows = []
    for _, r in raw.iterrows():
        data = r.get('data', None)
        if pd.isna(data) or data == '':
            continue
        try:
            dets = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        if not dets:
            rows.append({'t_rel_s': r['t_rel_s'], 'dist_min_m': np.nan,
                         'nb_obstacles': 0, 'closest_class': None})
            continue
        closest = min(dets, key=lambda d: d.get('dist_m', float('inf')))
        rows.append({
            't_rel_s': r['t_rel_s'],
            'dist_min_m': closest.get('dist_m', np.nan),
            'nb_obstacles': len(dets),
            'closest_class': closest.get('class', None),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('t_rel_s').reset_index(drop=True)


def load_float64_topic(path):
    """Charge un CSV à une seule valeur float par message (topics /perf/*)."""
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if df.empty or 'data' not in df.columns:
        return None
    df['data'] = pd.to_numeric(df['data'], errors='coerce')
    df = df.dropna(subset=['data']).sort_values('t_rel_s').reset_index(drop=True)
    return df if not df.empty else None


def load_optical_flow_zones(path):
    """
    Charge optical_flow_zones.csv (Float64MultiArray, 18 valeurs par message).

    Format de la liste : [L_mean, L_max, L_cov, L_div, L_risk, L_spike,
                          C_mean, C_max, C_cov, C_div, C_risk, C_spike,
                          R_mean, R_max, R_cov, R_div, R_risk, R_spike]
    (voir _publish_zone_metrics dans optical_flow_farneback.py)

    Retourne un DataFrame avec colonnes nommées + t_rel_s.
    """
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    if raw.empty or 'data' not in raw.columns:
        return None

    cols = ['L_mean', 'L_max', 'L_cov', 'L_div', 'L_risk', 'L_spike',
            'C_mean', 'C_max', 'C_cov', 'C_div', 'C_risk', 'C_spike',
            'R_mean', 'R_max', 'R_cov', 'R_div', 'R_risk', 'R_spike']
    rows = []
    for _, r in raw.iterrows():
        try:
            vals = json.loads(r['data']) if isinstance(r['data'], str) else list(r['data'])
        except Exception:
            continue
        if len(vals) < 18:
            continue
        row = {'t_rel_s': r['t_rel_s']}
        row.update(dict(zip(cols, vals[:18])))
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values('t_rel_s').reset_index(drop=True)


def load_ground_detections_raw(path):
    """
    Comme load_ground_detections, mais conserve la LISTE COMPLÈTE des
    détections de chaque message (pas seulement la plus proche). Sert
    à la vérification de cohérence du côté d'évitement (compute_vo_side)
    : on a besoin de la position latérale (y_m) de l'obstacle déclencheur,
    pas seulement de sa distance.

    Retourne une liste de tuples (t_rel_s, [dets...]) triée par temps.
    """
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path)
    out = []
    for _, r in raw.iterrows():
        data = r.get('data', None)
        if pd.isna(data) or data == '':
            continue
        try:
            dets = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        out.append((r['t_rel_s'], dets))
    out.sort(key=lambda x: x[0])
    return out if out else None


def load_odometry(path):
    """
    Charge odometry_filtered.csv (x, y, yaw_rad, vx, vy, vyaw).
    Nécessaire pour : (a) la vitesse RÉELLE du robot dans la validation
    TTC — la commande envoyée n'est pas la vitesse atteinte par un
    Husky chargé — et (b) une estimation de l'excursion de lacet pendant
    un évitement (approximation de MAX_AVOIDANCE_YAW_DEVIATION, voir
    check_yaw_excursions ci-dessous).
    """
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values('t_rel_s').reset_index(drop=True)
    return df


def load_ground_truth(path):
    """
    Charge le fichier de vérité terrain rempli MANUELLEMENT pendant le
    test (voir protocole, section "Calibration distance réelle/calculée").
    Colonnes attendues : label, real_distance_m, t_start_s, t_end_s.

    Si le fichier est absent, retourne None silencieusement — la
    calibration distance réelle/calculée est alors simplement sautée
    (elle dépend d'une mesure physique au ruban, impossible à déduire
    du bag seul).
    """
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    required = {'label', 'real_distance_m', 't_start_s', 't_end_s'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f'[avertissement] ground_truth_distances.csv : colonnes manquantes {missing_cols} — fichier ignoré.')
        return None
    return df


# ──────────────────────────────────────────────────────────────────────────
#  Helpers d'analyse
# ──────────────────────────────────────────────────────────────────────────

def emergency_intervals(df_bool):
    """Retourne la liste [(t_start, t_end), ...] des périodes data == True."""
    if df_bool is None or df_bool.empty:
        return []
    intervals = []
    active = False
    t_start = None
    for _, r in df_bool.iterrows():
        if r['data'] and not active:
            active = True
            t_start = r['t_rel_s']
        elif not r['data'] and active:
            active = False
            intervals.append((t_start, r['t_rel_s']))
    if active:
        intervals.append((t_start, df_bool['t_rel_s'].iloc[-1]))
    return intervals


def shade_intervals(ax, intervals, color='red', alpha=0.15, label=None):
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, color=color, alpha=alpha,
                  label=(label if i == 0 else None))


def gaps(df_twist):
    """dt entre messages consécutifs — détecte les silences de publication."""
    if df_twist is None or len(df_twist) < 2:
        return pd.DataFrame(columns=['t_rel_s', 'dt'])
    t = df_twist['t_rel_s'].values
    dt = np.diff(t)
    return pd.DataFrame({'t_rel_s': t[1:], 'dt': dt})


def detect_episodes(df_twist, max_gap=0.5):
    """
    Regroupe les messages de /avoidance_cmd_vel en épisodes contigus
    (le nœud ne publie QUE quand avoid_cmd n'est pas None — donc une
    longue absence de message = pas d'évitement actif).
    Retourne une liste de dicts {start, end, duration, n_msgs, n_side_flips}.
    """
    if df_twist is None or df_twist.empty:
        return []
    t = df_twist['t_rel_s'].values
    az = df_twist['angular_z'].values

    episodes = []
    ep_start_idx = 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > max_gap:
            episodes.append((ep_start_idx, i - 1))
            ep_start_idx = i
    episodes.append((ep_start_idx, len(t) - 1))

    out = []
    for i0, i1 in episodes:
        seg_az = az[i0:i1 + 1]
        signs = np.sign(seg_az)
        signs = signs[signs != 0]
        n_flips = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        out.append({
            'start': t[i0], 'end': t[i1],
            'duration': t[i1] - t[i0],
            'n_msgs': i1 - i0 + 1,
            'n_side_flips': n_flips,
        })
    return out


def compute_distance_calibration(ground_truth, ground):
    """
    Pour chaque plateau de mesure défini dans ground_truth_distances.csv
    (un obstacle posé à une distance RÉELLE connue pendant un intervalle
    [t_start_s, t_end_s], robot et obstacle immobiles), calcule la
    distance CALCULÉE moyenne par le pipeline homographie + YOLO
    (/ground_detections) sur le même intervalle, et l'écart.

    C'est le test de fond de la validation bird-eye view : sans cette
    comparaison, on ne sait jamais si une distance affichée en bird-eye
    correspond à la réalité ou à une homographie dérivée/mal calibrée.
    """
    if ground_truth is None or ground is None or ground.empty:
        return None
    rows = []
    for _, gt in ground_truth.iterrows():
        mask = (ground['t_rel_s'] >= gt['t_start_s']) & (ground['t_rel_s'] <= gt['t_end_s'])
        seg = ground.loc[mask, 'dist_min_m'].dropna()
        if seg.empty:
            rows.append({
                'label': gt['label'], 'real_distance_m': gt['real_distance_m'],
                'calc_mean_m': np.nan, 'calc_std_m': np.nan,
                'error_m': np.nan, 'n_samples': 0,
            })
            continue
        calc_mean = float(seg.mean())
        rows.append({
            'label': gt['label'],
            'real_distance_m': gt['real_distance_m'],
            'calc_mean_m': calc_mean,
            'calc_std_m': float(seg.std()) if len(seg) > 1 else 0.0,
            'error_m': calc_mean - gt['real_distance_m'],
            'n_samples': int(len(seg)),
        })
    return pd.DataFrame(rows)


def check_vo_side_consistency(episodes, ground_raw, lookback=0.3):
    """
    Vérifie, pour chaque épisode d'évitement, que le côté choisi
    (signe de angular_z au déclenchement) est cohérent avec la
    convention documentée dans compute_vo_side() :

        y_m > 0 (obstacle à DROITE du robot, convention homographie)
            → esquive 'left'  → angular_z > 0
        y_m < 0 (obstacle à GAUCHE)
            → esquive 'right' → angular_z < 0

    On retient, parmi les détections juste avant/au début de l'épisode
    (fenêtre [start-lookback, start]), l'obstacle le plus proche comme
    déclencheur le plus probable (cohérent avec la logique de priorité
    du nœud : catégorie 0 = quasi-collision dans le couloir, catégorie 1
    = TTC — dans les deux cas c'est l'obstacle le plus URGENT qui gagne,
    et le plus proche est un bon proxy de "le plus urgent" pour un test
    contrôlé avec un seul obstacle statique à la fois).

    LIMITE CONNUE : ce test est un PROXY, pas une vérification directe.
    Le nœud ne publie pas aujourd'hui quel obstacle a réellement déclenché
    l'esquive ni quelle branche (mobile/statique/fallback) de
    compute_vo_side a été utilisée. Pour des tests à un seul obstacle
    statique (cas Y-1/E/A du protocole), ce proxy est fiable. Pour des
    scénarios multi-obstacles (S-5), il peut se tromper de cible — ne
    pas tirer de conclusion définitive sur ces cas sans l'instrumentation
    proposée en annexe du protocole (publication du côté/obstacle choisi).

    Retourne une liste de dicts par épisode :
        {start, y_m_obstacle, angular_z_trigger, expected_side,
         actual_side, match}
    """
    if not episodes or not ground_raw:
        return []

    out = []
    for ep in episodes:
        # Détections dans la fenêtre de déclenchement
        candidates = [dets for (t, dets) in ground_raw
                      if ep['start'] - lookback <= t <= ep['start'] + 0.05]
        flat = [d for dets in candidates for d in dets]
        if not flat:
            continue
        closest = min(flat, key=lambda d: d.get('dist_m', float('inf')))
        y_m = closest.get('y_m', 0.0)

        expected_side = 'left' if y_m > 0 else 'right'
        # Le signe réel observé au premier message de l'épisode est
        # déjà la commande LISSÉE (EMA) — son signe reste néanmoins
        # fiable dès les premiers cycles car le lissage ne change pas
        # le signe, seulement la magnitude.
        out.append({
            'start': ep['start'],
            'y_m_obstacle': y_m,
            'closest_class': closest.get('class'),
            'dist_m': closest.get('dist_m'),
            'expected_side': expected_side,
        })
    return out


def attach_actual_side(vo_checks, avoid):
    """Complète vo_checks avec le côté RÉEL (signe angular_z) au début de
    chaque épisode, en relisant /avoidance_cmd_vel directement (plus
    fiable qu'une valeur déjà stockée, car on prend le PREMIER signe
    non-nul après le début de l'épisode, pas seulement le tout premier
    message qui peut être à 0 si l'EMA démarre de zéro)."""
    if avoid is None or avoid.empty:
        return vo_checks
    t = avoid['t_rel_s'].values
    az = avoid['angular_z'].values
    for check in vo_checks:
        window = (t >= check['start']) & (t <= check['start'] + 0.5)
        seg = az[window]
        seg_nz = seg[np.abs(seg) > 1e-3]
        if len(seg_nz) == 0:
            check['angular_z_trigger'] = 0.0
            check['actual_side'] = None
            check['match'] = None
            continue
        az0 = float(seg_nz[0])
        check['angular_z_trigger'] = az0
        check['actual_side'] = 'left' if az0 > 0 else 'right'
        check['match'] = (check['actual_side'] == check['expected_side'])
    return vo_checks


def check_yaw_excursions(episodes, odom):
    """
    Estime l'excursion de lacet (yaw) cumulée pendant chaque épisode
    d'évitement, à partir de l'odométrie réelle — PAS de la commande.

    APPROXIMATION ASSUMÉE : le nœud mémorise _target_yaw au début de
    l'esquive (cap voulu par le pilote juste avant déclenchement) et
    plafonne l'écart à MAX_AVOIDANCE_YAW_DEVIATION. Comme _target_yaw
    n'est pas publié, on utilise ici le yaw au DÉBUT de l'épisode comme
    proxy de _target_yaw — valide tant que le pilote n'a pas changé
    activement de cap juste avant (cas normal d'un test contrôlé en
    ligne droite). Le yaw est UNWRAPPÉ (np.unwrap) pour éviter le bug
    historique de remise à zéro à chaque tour complet (cf. notes projet).

    Retourne les épisodes enrichis avec 'max_yaw_excursion_rad' et
    'yaw_cap_violated' (True si l'excursion dépasse
    MAX_AVOIDANCE_YAW_DEVIATION malgré le plafond censé l'empêcher).
    """
    if not episodes or odom is None or odom.empty:
        return episodes
    t_odom = odom['t_rel_s'].values
    yaw_odom = np.unwrap(odom['yaw_rad'].values)

    for ep in episodes:
        mask = (t_odom >= ep['start']) & (t_odom <= ep['end'])
        if not np.any(mask):
            ep['max_yaw_excursion_rad'] = None
            ep['yaw_cap_violated'] = None
            continue
        seg_yaw = yaw_odom[mask]
        yaw0 = seg_yaw[0]
        excursion = np.max(np.abs(seg_yaw - yaw0))
        ep['max_yaw_excursion_rad'] = float(excursion)
        ep['yaw_cap_violated'] = bool(excursion > MAX_AVOIDANCE_YAW_DEVIATION)
    return episodes


# ──────────────────────────────────────────────────────────────────────────
#  Graphiques
# ──────────────────────────────────────────────────────────────────────────

def plot_commands_overlay(out_dir, cmd_vel, cmd_vel_in, joy, avoid, estop_intervals):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    series = [
        ('joy_teleop/cmd_vel', joy,        'tab:gray',   0.8),
        ('cmd_vel_in',         cmd_vel_in, 'tab:blue',   0.8),
        ('avoidance_cmd_vel',  avoid,      'tab:orange', 1.0),
        ('cmd_vel (sortie)',   cmd_vel,    'tab:green',  1.0),
    ]

    for label, df, color, lw in series:
        if df is None or df.empty:
            continue
        axes[0].plot(df['t_rel_s'], df['linear_x'], label=label, color=color, linewidth=lw)
        axes[1].plot(df['t_rel_s'], df['angular_z'], label=label, color=color, linewidth=lw)

    shade_intervals(axes[0], estop_intervals, label='EMERGENCY')
    shade_intervals(axes[1], estop_intervals)

    axes[0].set_ylabel('linear_x (m/s)')
    axes[0].set_title('Commandes — vitesse linéaire')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel('angular_z (rad/s)')
    axes[1].set_xlabel('temps (s)')
    axes[1].set_title('Commandes — vitesse angulaire')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_commands_overlay.png'), dpi=150)
    plt.close(fig)


def plot_obstacle_distance(out_dir, ground, estop_intervals):
    if ground is None or ground.empty:
        print('[info] /ground_detections vide ou absent — graphique 02 sauté.')
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ground['t_rel_s'], ground['dist_min_m'], color='tab:purple',
           linewidth=1.2, label='Distance obstacle le plus proche')

    ax.axhline(DISTANCE_WARN_M, color='orange', linestyle='--', linewidth=1,
              label=f'Seuil WARNING ({DISTANCE_WARN_M} m)')
    ax.axhline(DISTANCE_DANGER_M, color='red', linestyle='--', linewidth=1,
              label=f'Seuil DANGER ({DISTANCE_DANGER_M} m)')

    shade_intervals(ax, estop_intervals, label='EMERGENCY')

    ax.set_xlabel('temps (s)')
    ax.set_ylabel('distance (m)')
    ax.set_title('Distance au plus proche obstacle détecté')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_obstacle_distance.png'), dpi=150)
    plt.close(fig)


def plot_watchdog_latency(out_dir, joy, cmd_vel_in):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, (label, df) in zip(axes, [('joy_teleop/cmd_vel', joy),
                                       ('cmd_vel_in', cmd_vel_in)]):
        g = gaps(df)
        if g.empty:
            ax.set_title(f'{label}  pas de données')
            continue
        colors = np.where(g['dt'] > CMD_SOURCE_TIMEOUT, 'red', 'tab:blue')
        ax.scatter(g['t_rel_s'], g['dt'], c=colors, s=8)
        ax.axhline(CMD_SOURCE_TIMEOUT, color='red', linestyle='--', linewidth=1,
                  label=f'CMD_SOURCE_TIMEOUT ({CMD_SOURCE_TIMEOUT}s)')
        n_violations = int((g['dt'] > CMD_SOURCE_TIMEOUT).sum())
        ax.set_title(f'{label} — écarts entre messages consécutifs '
                     f'({n_violations} silence(s) > {CMD_SOURCE_TIMEOUT}s)')
        ax.set_ylabel('dt (s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('temps (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_watchdog_latency.png'), dpi=150)
    plt.close(fig)


def plot_avoidance_episodes(out_dir, episodes):
    if not episodes:
        print('[info] Aucun épisode /avoidance_cmd_vel détecté — graphique 04 sauté.')
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Chronologie des épisodes (barres horizontales)
    ax0 = axes[0]
    for i, ep in enumerate(episodes):
        color = 'red' if ep['n_side_flips'] >= 2 else 'tab:orange'
        ax0.barh(i, ep['duration'], left=ep['start'], color=color, height=0.6)
        # N'annoter que les épisodes avec au moins 1 flip — sinon, avec
        # des dizaines/centaines d'épisodes sans saccade, le texte "0
        # flip(s)" répété rend le graphique illisible sans rien apporter.
        if ep['n_side_flips'] >= 1:
            ax0.text(ep['start'], i, f"  {ep['n_side_flips']} flip(s)",
                     va='center', fontsize=7, color='red' if ep['n_side_flips'] >= 2 else 'black')
    ax0.set_xlabel('temps (s)')
    ax0.set_ylabel('# épisode')
    ax0.set_title("Épisodes d'évitement — rouge = ≥2 changements de côté (saccade potentielle)")
    ax0.grid(alpha=0.3, axis='x')

    # Histogramme des durées
    ax1 = axes[1]
    durations = [ep['duration'] for ep in episodes]
    ax1.hist(durations, bins=min(30, max(5, len(durations) // 2)), color='tab:orange', edgecolor='black')
    ax1.set_xlabel('durée de l\'épisode (s)')
    ax1.set_ylabel('nombre d\'épisodes')
    ax1.set_title('Distribution des durées d\'évitement')
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_avoidance_episodes.png'), dpi=150)
    plt.close(fig)


def plot_distance_calibration(out_dir, calib):
    if calib is None or calib.empty:
        print('[info] ground_truth_distances.csv absent — graphique 06 sauté '
              '(calibration distance réelle/calculée impossible sans mesure au ruban).')
        return

    valid = calib.dropna(subset=['calc_mean_m'])
    if valid.empty:
        print('[avertissement] ground_truth_distances.csv présent mais aucune mesure '
              'recoupée avec /ground_detections sur les fenêtres de temps données — '
              'vérifiez t_start_s/t_end_s.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax0 = axes[0]
    lim = max(valid['real_distance_m'].max(), valid['calc_mean_m'].max()) * 1.15
    ax0.plot([0, lim], [0, lim], color='gray', linestyle='--', linewidth=1, label='y = x (idéal)')
    ax0.errorbar(valid['real_distance_m'], valid['calc_mean_m'], yerr=valid['calc_std_m'],
                fmt='o', color='tab:purple', ecolor='tab:purple', capsize=3, label='Mesures')
    for _, r in valid.iterrows():
        ax0.annotate(str(r['label']), (r['real_distance_m'], r['calc_mean_m']),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax0.set_xlabel('Distance RÉELLE (ruban) [m]')
    ax0.set_ylabel('Distance CALCULÉE (/ground_detections, moyenne ± écart-type) [m]')
    ax0.set_title('Calibration bird-eye : distance réelle vs calculée')
    ax0.set_xlim(0, lim); ax0.set_ylim(0, lim)
    ax0.legend(loc='upper left', fontsize=8)
    ax0.grid(alpha=0.3)
    ax0.set_aspect('equal')

    ax1 = axes[1]
    colors = ['red' if abs(e) > 0.10 else 'tab:green' for e in valid['error_m']]
    ax1.bar(valid['label'].astype(str), valid['error_m'], color=colors)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axhline(0.10, color='red', linestyle='--', linewidth=1, label='±10 cm')
    ax1.axhline(-0.10, color='red', linestyle='--', linewidth=1)
    ax1.set_ylabel('Erreur = calculée − réelle [m]')
    ax1.set_title('Erreur de calibration par plateau de distance')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_distance_calibration.png'), dpi=150)
    plt.close(fig)


def plot_vo_side_consistency(out_dir, vo_checks):
    valid = [c for c in vo_checks if c.get('match') is not None]
    if not valid:
        print('[info] Aucune vérification de cohérence VO possible '
              '(/ground_detections et /avoidance_cmd_vel requis ensemble) — graphique 07 sauté.')
        return

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for c in valid:
        color = 'tab:green' if c['match'] else 'tab:red'
        marker = 'o' if c['match'] else 'x'
        ax.scatter(c['y_m_obstacle'], c['angular_z_trigger'], color=color, marker=marker, s=60)

    ax.axvline(0, color='gray', linewidth=1)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xlabel('y_m obstacle déclencheur (>0 = droite robot, <0 = gauche)')
    ax.set_ylabel('angular_z au déclenchement (>0 = esquive gauche)')
    n_match = sum(c['match'] for c in valid)
    ax.set_title(f"Cohérence du côté d'esquive (compute_vo_side) — "
                f"{n_match}/{len(valid)} cohérents\n"
                f"Attendu : y_m>0 → angular_z>0 (haut-droite) | y_m<0 → angular_z<0 (bas-gauche)",
                fontsize=10)
    ax.grid(alpha=0.3)

    # pyrefly: ignore [missing-import]
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=8, label='Cohérent'),
              Line2D([0], [0], marker='x', color='tab:red', markersize=8, label='INCOHÉRENT (ou mauvais obstacle identifié — voir note)')]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.text(0.02, 0.02,
           "Note : un point INCOHÉRENT peut signaler soit un vrai défaut de\n"
           "compute_vo_side, soit que ce proxy a identifié le mauvais obstacle\n"
           "déclencheur (scène multi-obstacles). Croiser avec /yolo/image ou\n"
           "/bird_eye_estop au même timestamp avant de conclure à un bug.",
           transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '07_vo_side_consistency.png'), dpi=150)
    plt.close(fig)


def plot_yaw_excursions(out_dir, episodes):
    valid = [ep for ep in episodes if ep.get('max_yaw_excursion_rad') is not None]
    if not valid:
        print('[info] odometry_filtered.csv absent — graphique 08 sauté '
              '(excursion de lacet non vérifiable sans odométrie réelle).')
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    starts = [ep['start'] for ep in valid]
    excursions = [ep['max_yaw_excursion_rad'] for ep in valid]
    colors = ['red' if ep['yaw_cap_violated'] else 'tab:blue' for ep in valid]
    ax.scatter(starts, excursions, c=colors, s=40)
    ax.axhline(MAX_AVOIDANCE_YAW_DEVIATION, color='red', linestyle='--', linewidth=1,
              label=f'Plafond MAX_AVOIDANCE_YAW_DEVIATION ({MAX_AVOIDANCE_YAW_DEVIATION} rad)')
    ax.set_xlabel('temps de déclenchement de l\'épisode (s)')
    ax.set_ylabel('excursion de lacet max pendant l\'épisode (rad, approx.)')
    n_viol = sum(ep['yaw_cap_violated'] for ep in valid)
    ax.set_title(f"Excursion de lacet par épisode d'évitement — {n_viol} dépassement(s) du plafond")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '08_yaw_excursions.png'), dpi=150)
    plt.close(fig)


def plot_perf_cpu_gpu(out_dir, cpu_df, gpu_df, inference_df, estop_intervals):
    """
    09_perf_cpu_gpu.png — CPU % et GPU % dans le temps.

    Permet de comparer directement la charge système YOLO vs Farneback
    en cherchant dans le même graphique :
      - CPU élevé en continu → Farneback sur ARM (pas de GPU dédié)
      - GPU élevé en continu → YOLO sur NVIDIA
      - Pics CPU/GPU coïncidant avec EMERGENCY → charge lors des décisions critiques
    """
    has_cpu = cpu_df is not None and not cpu_df.empty
    has_gpu = gpu_df is not None and not gpu_df.empty and (gpu_df['data'] > 0).any()
    has_inf = inference_df is not None and not inference_df.empty

    if not has_cpu and not has_gpu:
        print('[info] /perf/cpu_percent et /perf/gpu_percent absents — graphique 09 sauté '
              '(perf_monitor.py non lancé pendant ce test).')
        return

    n_rows = 1 + (1 if has_inf else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    if has_cpu:
        ax0.plot(cpu_df['t_rel_s'], cpu_df['data'], color='tab:blue',
                linewidth=1.2, label='CPU global %')
    if has_gpu:
        ax0_r = ax0.twinx()
        valid_gpu = gpu_df[gpu_df['data'] >= 0]
        ax0_r.plot(valid_gpu['t_rel_s'], valid_gpu['data'], color='tab:orange',
                  linewidth=1.2, label='GPU %')
        ax0_r.set_ylabel('GPU %', color='tab:orange')
        ax0_r.tick_params(axis='y', labelcolor='tab:orange')
        ax0_r.set_ylim(0, 105)
        ax0_r.legend(loc='upper right', fontsize=8)

    shade_intervals(ax0, estop_intervals, label='EMERGENCY')
    ax0.set_ylabel('CPU %')
    ax0.set_ylim(0, 105)
    ax0.set_title('Charge CPU / GPU pendant la session')
    ax0.legend(loc='upper left', fontsize=8)
    ax0.grid(alpha=0.3)

    if has_inf:
        ax1 = axes[1]
        # Filtrer les valeurs aberrantes (délai entre deux sessions)
        inf_clean = inference_df[inference_df['data'] < 1000]
        ax1.plot(inf_clean['t_rel_s'], inf_clean['data'],
                color='tab:purple', linewidth=1.0, alpha=0.8)
        ax1.axhline(100.0, color='gray', linestyle='--', linewidth=1,
                   label='TIMER_PERIOD attendu (100 ms)')
        ax1.axhline(200.0, color='red', linestyle=':', linewidth=1,
                   label='2× TIMER_PERIOD (cycle sauté)')
        shade_intervals(ax1, estop_intervals)
        ax1.set_ylabel('Période d\'inférence (ms)')
        ax1.set_xlabel('temps (s)')
        ax1.set_title('Période réelle de run_inference (proxy: /movement_command ou /ground_detections)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '09_perf_cpu_gpu.png'), dpi=150)
    plt.close(fig)


def plot_latency_joy_cmd(out_dir, latency_df, inference_df, estop_intervals):
    """
    10_latency_joy_cmd.png — latence joystick → /cmd_vel.

    Deux sous-graphiques :
      - Latence brute par événement (scatter) + percentiles P50/P95/P99
      - Histogramme de distribution — confirme qu'on est en régime normal
        (pic à 10–30 ms) ou qu'il y a une queue longue (surcharge CPU)

    La latence mesurée par perf_monitor.py inclut :
        ROS2 middleware (sub/pub) + twist_mux + cb_cmd_vel_in callback
    Elle N'INCLUT PAS le délai côté moteurs (encodeurs → odométrie ≈ 20 ms).
    """
    has_lat = latency_df is not None and not latency_df.empty
    has_inf = inference_df is not None and not inference_df.empty

    if not has_lat:
        print('[info] /perf/joy_to_cmd_latency_ms absent — graphique 10 sauté '
              '(perf_monitor.py non lancé ou aucun mouvement enregistré).')
        return

    # Filtrer les outliers extrêmes (> 500 ms → probablement hors sujet)
    lat_clean = latency_df[latency_df['data'] < 500].copy()
    if lat_clean.empty:
        print('[avertissement] Toutes les latences > 500 ms — données suspectes, '
              'graphique 10 sauté.')
        return

    p50 = float(np.percentile(lat_clean['data'], 50))
    p95 = float(np.percentile(lat_clean['data'], 95))
    p99 = float(np.percentile(lat_clean['data'], 99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter temporel
    ax0 = axes[0]
    colors = np.where(lat_clean['data'] > p95, 'red', 'teal').tolist()
    ax0.scatter(lat_clean['t_rel_s'], lat_clean['data'],
               c=colors, s=12, alpha=0.7)
    ax0.axhline(p50, color='tab:green', linestyle='-', linewidth=1.2,
               label=f'P50 = {p50:.1f} ms')
    ax0.axhline(p95, color='tab:orange', linestyle='--', linewidth=1.2,
               label=f'P95 = {p95:.1f} ms')
    ax0.axhline(p99, color='red', linestyle=':', linewidth=1.2,
               label=f'P99 = {p99:.1f} ms')
    shade_intervals(ax0, estop_intervals)
    ax0.set_xlabel('temps (s)')
    ax0.set_ylabel('latence joy → /cmd_vel (ms)')
    ax0.set_title('Latence joystick → moteurs dans le temps')
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax0.set_ylim(bottom=0)

    # Histogramme
    ax1 = axes[1]
    ax1.hist(lat_clean['data'], bins=min(50, max(10, len(lat_clean) // 5)),
            color='teal', edgecolor='black', alpha=0.8)
    ax1.axvline(p50, color='tab:green', linestyle='-', linewidth=1.5,
               label=f'P50={p50:.1f}ms')
    ax1.axvline(p95, color='tab:orange', linestyle='--', linewidth=1.5,
               label=f'P95={p95:.1f}ms')
    ax1.axvline(p99, color='red', linestyle=':', linewidth=1.5,
               label=f'P99={p99:.1f}ms')
    ax1.set_xlabel('latence (ms)')
    ax1.set_ylabel('occurrences')
    ax1.set_title('Distribution des latences joy → /cmd_vel')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'Latence joystick → /cmd_vel  (n={len(lat_clean)})\n'
        f'P50={p50:.1f}ms  P95={p95:.1f}ms  P99={p99:.1f}ms',
        fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '10_latency_joy_cmd.png'), dpi=150)
    plt.close(fig)


def plot_farneback_flow(out_dir, flow_df, estop_intervals):
    """
    11_farneback_flow.png — métriques de flux optique Farneback par zone.

    Équivalent de 02_obstacle_distance.png pour YOLO :
    la "distance" n'existe plus, on la remplace par la magnitude de flux
    zone centrale (proxy direct du risque de collision frontale).
    Utile pour comparer les seuils FLOW_CENTER_WARN/DANGER avec les
    déclenchements EMERGENCY réels.
    """
    if flow_df is None or flow_df.empty:
        print('[info] optical_flow_zones absent — graphique 11 sauté '
              '(bag vient d\'une session YOLO, ou topic non enregistré).')
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Magnitudes moyennes par zone
    ax0 = axes[0]
    for col, label, color in [('L_mean', 'Gauche (mean)', 'tab:blue'),
                               ('C_mean', 'Centre (mean)', 'tab:red'),
                               ('R_mean', 'Droite (mean)', 'tab:green')]:
        if col in flow_df:
            ax0.plot(flow_df['t_rel_s'], flow_df[col], label=label,
                    color=color, linewidth=1.2)

    # Seuils du code (valeurs Farneback — abaissées vs YOLO)
    # Ces constantes sont définies en haut de ce fichier sous les noms
    # FLOW_CENTER_WARN et FLOW_CENTER_DANGER. Farneback utilise des valeurs
    # plus basses que YOLO (3.0/1.5 vs 8.0/4.0) à cause de la résolution
    # réduite (320×240 vs 640×480). Pour superposer les seuils corrects sur
    # ce graphique, on tente de les lire depuis la config du nœud — à défaut,
    # on utilise les valeurs Farneback par défaut.
    farn_warn   = globals().get('FARNEBACK_FLOW_CENTER_WARN',   1.5)
    farn_danger = globals().get('FARNEBACK_FLOW_CENTER_DANGER', 3.0)

    ax0.axhline(farn_warn,   color='orange', linestyle='--', linewidth=1,
               label=f'FLOW_CENTER_WARN ({farn_warn} px)')
    ax0.axhline(farn_danger, color='red',    linestyle='--', linewidth=1,
               label=f'FLOW_CENTER_DANGER ({farn_danger} px)')
    shade_intervals(ax0, estop_intervals, label='EMERGENCY')
    ax0.set_ylabel('Magnitude flux optique (px/frame, lissée)')
    ax0.set_title('Flux optique Farneback — magnitudes moyennes par zone')
    ax0.legend(loc='upper right', fontsize=8)
    ax0.grid(alpha=0.3)
    ax0.set_ylim(bottom=0)

    # Déséquilibre G/D (signal d'évitement)
    ax1 = axes[1]
    if 'L_mean' in flow_df and 'R_mean' in flow_df:
        diff = flow_df['R_mean'] - flow_df['L_mean']
        colors_diff = np.where(diff > 0, 'tab:orange', 'tab:blue')
        ax1.bar(flow_df['t_rel_s'], diff,
               color=colors_diff, width=0.08, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.8)
        # FLOW_DIFF_AVOID n'est pas importé ici — on met la valeur brute
        ax1.axhline(0.6, color='red', linestyle=':', linewidth=1,
                   label='FLOW_DIFF_AVOID = 0.6 px (→ évitement)')
        ax1.axhline(-0.6, color='red', linestyle=':', linewidth=1)
        shade_intervals(ax1, estop_intervals)
        ax1.set_ylabel('Déséquilibre R - L (px)')
        ax1.set_xlabel('temps (s)')
        ax1.set_title('Déséquilibre flux droite − gauche (→ déclenchement évitement réflexe)')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '11_farneback_flow.png'), dpi=150)
    plt.close(fig)


def plot_state_timeline(out_dir, estop_intervals, avoid_episodes, ground):
    fig, ax1 = plt.subplots(figsize=(14, 5))

    shade_intervals(ax1, estop_intervals, color='red', alpha=0.25, label='EMERGENCY')
    for i, ep in enumerate(avoid_episodes):
        ax1.axvspan(ep['start'], ep['end'], color='orange', alpha=0.25,
                  label=('Évitement actif' if i == 0 else None))

    if ground is not None and not ground.empty:
        ax2 = ax1.twinx()
        ax2.plot(ground['t_rel_s'], ground['dist_min_m'], color='tab:purple', linewidth=1)
        ax2.set_ylabel('distance obstacle (m)', color='tab:purple')
        ax2.tick_params(axis='y', labelcolor='tab:purple')
        ax2.set_ylim(bottom=0)

    ax1.set_xlabel('temps (s)')
    ax1.set_yticks([])
    ax1.set_title('Vue synthétique de la session : états + distance obstacle')
    ax1.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_state_timeline.png'), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
#  Résumé texte
# ──────────────────────────────────────────────────────────────────────────

def write_summary(out_dir, estop_intervals, joy, cmd_vel_in, episodes, ground,
                  calib=None, vo_checks=None):
    lines = []
    lines.append('=== RÉSUMÉ DE SESSION ===\n')

    # Emergency stop
    total_estop = sum(t1 - t0 for t0, t1 in estop_intervals)
    lines.append(f'Arrêts d\'urgence (EMERGENCY) : {len(estop_intervals)} occurrence(s), '
                f'durée totale {total_estop:.2f}s')
    for i, (t0, t1) in enumerate(estop_intervals):
        lines.append(f'    #{i+1}: t={t0:.2f}s → t={t1:.2f}s (durée {t1 - t0:.2f}s)')
    lines.append('')

    # Watchdog
    for label, df in [('joy_teleop/cmd_vel', joy), ('cmd_vel_in', cmd_vel_in)]:
        g = gaps(df)
        violations = g[g['dt'] > CMD_SOURCE_TIMEOUT]
        lines.append(f'Silences > {CMD_SOURCE_TIMEOUT}s sur {label} : {len(violations)}')
        for _, r in violations.iterrows():
            lines.append(f'    t≈{r["t_rel_s"]:.2f}s — silence de {r["dt"]:.2f}s')
    lines.append('')

    # Avoidance episodes
    lines.append(f'Épisodes d\'évitement (/avoidance_cmd_vel) : {len(episodes)}')
    if episodes:
        durations = [ep['duration'] for ep in episodes]
        flips = [ep['n_side_flips'] for ep in episodes]
        lines.append(f'    durée moyenne : {np.mean(durations):.2f}s, max : {np.max(durations):.2f}s')
        lines.append(f'    changements de côté moyens/épisode : {np.mean(flips):.2f}, max : {np.max(flips)}')
        saccades = [ep for ep in episodes if ep['n_side_flips'] >= 2]
        lines.append(f'    épisodes avec ≥2 changements de côté (saccade potentielle) : {len(saccades)}')
        for ep in saccades:
            lines.append(f'        t={ep["start"]:.2f}s → t={ep["end"]:.2f}s '
                         f'({ep["n_side_flips"]} flips)')

        # Excursion de lacet (si odométrie disponible)
        yaw_eps = [ep for ep in episodes if ep.get('max_yaw_excursion_rad') is not None]
        if yaw_eps:
            excursions = [ep['max_yaw_excursion_rad'] for ep in yaw_eps]
            n_viol = sum(ep['yaw_cap_violated'] for ep in yaw_eps)
            lines.append(f'    excursion de lacet max observée : {np.max(excursions):.2f} rad '
                        f'(plafond code : {MAX_AVOIDANCE_YAW_DEVIATION} rad)')
            lines.append(f'    épisodes DÉPASSANT le plafond malgré la protection : {n_viol}')
            for ep in yaw_eps:
                if ep['yaw_cap_violated']:
                    lines.append(f'        t={ep["start"]:.2f}s — excursion {ep["max_yaw_excursion_rad"]:.2f} rad'
                                f' (ANOMALIE — à investiguer en priorité)')
        else:
            lines.append('    excursion de lacet : non vérifiable (odometry_filtered.csv absent)')
    lines.append('')

    # Cohérence du côté d'esquive (VO)
    if vo_checks:
        valid = [c for c in vo_checks if c.get('match') is not None]
        if valid:
            n_match = sum(c['match'] for c in valid)
            rate = 100.0 * n_match / len(valid)
            lines.append(f'Cohérence du côté d\'esquive (compute_vo_side, proxy obstacle le plus '
                        f'proche) : {n_match}/{len(valid)} ({rate:.0f}%)')
            mismatches = [c for c in valid if not c['match']]
            for c in mismatches:
                lines.append(f'    INCOHÉRENT à t={c["start"]:.2f}s : obstacle y_m={c["y_m_obstacle"]:.2f}m '
                            f'(attendu {c["expected_side"]}), commande observée {c["actual_side"]} '
                            f'(angular_z={c["angular_z_trigger"]:.2f})')
            lines.append('    (proxy fiable uniquement pour des tests à un seul obstacle statique — '
                        'voir limite documentée dans check_vo_side_consistency)')
        else:
            lines.append('Cohérence du côté d\'esquive : non vérifiable '
                        '(/ground_detections et /avoidance_cmd_vel requis ensemble, '
                        'ou aucun épisode détecté)')
    lines.append('')

    # Calibration distance réelle / calculée (bird-eye view)
    if calib is not None and not calib.empty:
        valid_calib = calib.dropna(subset=['calc_mean_m'])
        if not valid_calib.empty:
            bias = float(valid_calib['error_m'].mean())
            rmse = float(np.sqrt((valid_calib['error_m'] ** 2).mean()))
            max_err = float(valid_calib['error_m'].abs().max())
            lines.append('Calibration distance réelle (ruban) vs calculée (/ground_detections) :')
            lines.append(f'    biais moyen : {bias:+.3f} m, RMSE : {rmse:.3f} m, '
                        f'erreur max : {max_err:.3f} m')
            for _, r in valid_calib.iterrows():
                lines.append(f'    {r["label"]:<14} réel={r["real_distance_m"]:.2f}m  '
                            f'calc={r["calc_mean_m"]:.2f}m±{r["calc_std_m"]:.2f}  '
                            f'erreur={r["error_m"]:+.3f}m  (n={r["n_samples"]})')
        else:
            lines.append('Calibration distance : ground_truth_distances.csv présent mais '
                        'aucune correspondance temporelle trouvée — vérifiez t_start_s/t_end_s.')
    else:
        lines.append('Calibration distance réelle/calculée : SAUTÉE — créez '
                    'ground_truth_distances.csv (voir protocole, section calibration bird-eye) '
                    'pour activer cette validation essentielle.')
    lines.append('')

    # Obstacle distance
    if ground is not None and not ground.empty:
        idx_min = ground['dist_min_m'].idxmin()
        row_min = ground.loc[idx_min]
        lines.append(f'Distance minimale atteinte : {row_min["dist_min_m"]:.2f}m '
                    f'à t={row_min["t_rel_s"]:.2f}s (classe: {row_min["closest_class"]})')
        n_danger = (ground['dist_min_m'] <= DISTANCE_DANGER_M).sum()
        n_warning = ((ground['dist_min_m'] > DISTANCE_DANGER_M)
                    & (ground['dist_min_m'] <= DISTANCE_WARN_M)).sum()
        lines.append(f'Messages avec obstacle en zone DANGER (≤{DISTANCE_DANGER_M}m) : {n_danger}')
        lines.append(f'Messages avec obstacle en zone WARNING (≤{DISTANCE_WARN_M}m) : {n_warning}')
    lines.append('')

    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text)


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def _append_perf_summary(out_dir, cpu_df, gpu_df, inference_df, latency_df):
    """Ajoute une section performance à la fin de summary.txt."""
    path = os.path.join(out_dir, 'summary.txt')
    lines = ['\n=== PERFORMANCE SYSTÈME (perf_monitor.py) ===\n']

    if cpu_df is not None and not cpu_df.empty:
        valid = cpu_df['data'].dropna()
        lines.append(f'CPU global  : moy={valid.mean():.1f}%  '
                    f'max={valid.max():.1f}%  '
                    f'P95={float(np.percentile(valid, 95)):.1f}%')
    else:
        lines.append('CPU : données absentes (perf_monitor.py non lancé)')

    if gpu_df is not None and not gpu_df.empty:
        valid = gpu_df[gpu_df['data'] >= 0]['data']
        if not valid.empty:
            lines.append(f'GPU compute : moy={valid.mean():.1f}%  '
                        f'max={valid.max():.1f}%  '
                        f'P95={float(np.percentile(valid, 95)):.1f}%')
        else:
            lines.append('GPU : non détecté (-1.0) — voir GPU auto-détection dans perf_monitor.py')
    else:
        lines.append('GPU : données absentes')

    if inference_df is not None and not inference_df.empty:
        valid = inference_df[(inference_df['data'] > 20)
                            & (inference_df['data'] < 1000)]['data']
        if not valid.empty:
            n_skipped = int((valid > 200).sum())
            lines.append(f'Période run_inference : moy={valid.mean():.1f}ms  '
                        f'max={valid.max():.1f}ms  '
                        f'P95={float(np.percentile(valid, 95)):.1f}ms')
            lines.append(f'  cycles sautés (période > 200ms) : {n_skipped}')
    else:
        lines.append('Période d\'inférence : données absentes')

    if latency_df is not None and not latency_df.empty:
        valid = latency_df[(latency_df['data'] > 0) & (latency_df['data'] < 500)]['data']
        if not valid.empty:
            lines.append(f'Latence joy→cmd : '
                        f'P50={float(np.percentile(valid, 50)):.1f}ms  '
                        f'P95={float(np.percentile(valid, 95)):.1f}ms  '
                        f'P99={float(np.percentile(valid, 99)):.1f}ms  '
                        f'max={valid.max():.1f}ms  (n={len(valid)})')
    else:
        lines.append('Latence joy→cmd : données absentes')

    lines.append('\nPour comparer YOLO vs Farneback :')
    lines.append('  → CPU moy YOLO vs Farneback (YOLO + GPU >> Farneback CPU seul)')
    lines.append('  → Latence P95 : Farneback sur ARM peut être > 100ms si CPU saturé')
    lines.append('  → Période d\'inférence : Farneback doit rester ≈ 100ms ; '
                'si > 200ms → baisser FARNEBACK_LEVELS ou PROCESS_WIDTH')

    text = '\n'.join(lines) + '\n'
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Remplacement du défaut '.' par votre variable DEFAULT_CSV_DIR
    parser.add_argument('csv_dir', nargs='?', default=DEFAULT_CSV_DIR,
                        help=f'Dossier contenant les CSV produits par extract_bag.py (défaut: {DEFAULT_CSV_DIR})')
    parser.add_argument('--out', default=None,
                        help='Dossier de sortie pour les graphiques (défaut: ./analyse_<nom_du_dossier>)')
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        print(f'[erreur] Dossier introuvable : {args.csv_dir}')
        sys.exit(1)

    out_dir = args.out or f'./analyse_{os.path.basename(os.path.normpath(args.csv_dir))}'
    os.makedirs(out_dir, exist_ok=True)

    def p(name):
        return os.path.join(args.csv_dir, name)

    print(f'[info] Chargement des CSV depuis : {args.csv_dir}')
    cmd_vel      = load_twist(p('cmd_vel.csv'))
    cmd_vel_in   = load_twist(p('cmd_vel_in.csv'))
    joy          = load_twist(p('joy_teleop_cmd_vel.csv'))
    avoid        = load_twist(p('avoidance_cmd_vel.csv'))
    estop        = load_bool(p('emergency_stop.csv'))
    ground       = load_ground_detections(p('ground_detections.csv'))
    ground_raw   = load_ground_detections_raw(p('ground_detections.csv'))
    odom         = load_odometry(p('odometry_filtered.csv'))
    ground_truth = load_ground_truth(p('ground_truth_distances.csv'))

    # Nouveaux topics — perf_monitor.py (présents uniquement si le script
    # était lancé pendant l'enregistrement)
    cpu_df       = load_float64_topic(p('perf_cpu_percent.csv'))
    gpu_df       = load_float64_topic(p('perf_gpu_percent.csv'))
    inference_df = load_float64_topic(p('perf_inference_period_ms.csv'))
    latency_df   = load_float64_topic(p('perf_joy_to_cmd_latency_ms.csv'))

    # Topics Farneback spécifiques (absents pour une session YOLO)
    flow_zones_df = load_optical_flow_zones(p('optical_flow_zones.csv'))

    for label, df in [('cmd_vel', cmd_vel), ('cmd_vel_in', cmd_vel_in),
                      ('joy_teleop_cmd_vel', joy), ('avoidance_cmd_vel', avoid),
                      ('emergency_stop', estop), ('ground_detections', ground),
                      ('odometry_filtered', odom)]:
        if df is None:
            print(f'[info] {label} absent ou vide — graphiques concernés sautés.')
    if ground_truth is None:
        print('[info] ground_truth_distances.csv absent — calibration distance sautée '
              '(voir protocole pour le créer).')
    if all(df is None for df in [cpu_df, gpu_df, inference_df, latency_df]):
        print('[info] Topics /perf/* absents — graphiques 09/10 sautés '
              '(lancez perf_monitor.py avant la prochaine session).')
    if flow_zones_df is None:
        print('[info] optical_flow_zones.csv absent — graphique 11 sauté '
              '(bag YOLO, ou topic non enregistré).')

    estop_intervals = emergency_intervals(estop)
    episodes        = detect_episodes(avoid)
    episodes        = check_yaw_excursions(episodes, odom)

    vo_checks = check_vo_side_consistency(episodes, ground_raw)
    vo_checks = attach_actual_side(vo_checks, avoid)

    calib = compute_distance_calibration(ground_truth, ground)

    print('[info] Génération des graphiques...')
    plot_commands_overlay(out_dir, cmd_vel, cmd_vel_in, joy, avoid, estop_intervals)
    plot_obstacle_distance(out_dir, ground, estop_intervals)
    plot_watchdog_latency(out_dir, joy, cmd_vel_in)
    plot_avoidance_episodes(out_dir, episodes)
    plot_state_timeline(out_dir, estop_intervals, episodes, ground)
    plot_distance_calibration(out_dir, calib)
    plot_vo_side_consistency(out_dir, vo_checks)
    plot_yaw_excursions(out_dir, episodes)
    plot_perf_cpu_gpu(out_dir, cpu_df, gpu_df, inference_df, estop_intervals)
    plot_latency_joy_cmd(out_dir, latency_df, inference_df, estop_intervals)
    plot_farneback_flow(out_dir, flow_zones_df, estop_intervals)

    write_summary(out_dir, estop_intervals, joy, cmd_vel_in, episodes, ground,
                 calib=calib, vo_checks=vo_checks)

    # Résumé perf dans le summary.txt
    _append_perf_summary(out_dir, cpu_df, gpu_df, inference_df, latency_df)

    print(f'\n[ok] Graphiques + résumé écrits dans : {out_dir}')


if __name__ == '__main__':
    main()