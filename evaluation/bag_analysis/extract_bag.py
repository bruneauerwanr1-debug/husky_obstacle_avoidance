#!/usr/bin/env python3
"""
extract_bag.py — Extraction rapide d'un bag rosbag2 (.db3) vers des CSV lisibles.

Lit UNIQUEMENT les topics légers (Twist, Bool, String, Odometry, Point) —
la caméra /camera/color/image_raw et les images debug (/yolo/image,
/bird_eye_estop) sont volontairement exclues pour aller vite sur un bag
de plusieurs Go. Pas de rclpy.spin, pas de republication réseau : lecture
séquentielle directe du fichier SQLite via rosbag2_py.

USAGE :
    python3 extract_bag.py /chemin/vers/dossier_du_bag
    python3 extract_bag.py /chemin/vers/dossier_du_bag --out ./resultats

Le dossier du bag est celui qui contient le metadata.yaml + le .db3
(ex: "test_full_20260623_100100", pas le fichier .db3 lui-même).

SORTIES (dans --out, par défaut ./bag_extracted/) :
    cmd_vel.csv
    cmd_vel_in.csv
    joy_teleop_cmd_vel.csv
    avoidance_cmd_vel.csv
    emergency_stop.csv
    ground_detections.csv          (si présent et non vide)
    odometry_filtered.csv          (x, y, yaw_rad calculé du quaternion,
                                     vx, vy, vyaw — NÉCESSAIRE pour valider
                                     le TTC et pour servir de référence
                                     "pose robot" dans l'analyse distance)
    robot_ground_position.csv      (x, y — référence utilisée par le nœud
                                     comme origine des distances ; reste
                                     généralement (0,0) si /robot_ground_
                                     position n'est jamais republié par un
                                     autre nœud — normal, pas un bug)
    timeline_merged.csv        ← Twist/Bool/String fusionnés, triés par
                                  temps (Odometry et Point ont un schéma
                                  différent et restent dans leur propre
                                  CSV, sinon la fusion casserait le
                                  format colonnes fixes).

Pour repérer un freeze du watchdog ou une saccade :
    - ouvrez timeline_merged.csv
    - triez/filtrez par "topic"
    - cherchez les écarts de "t_rel_s" anormalement grands sur /cmd_vel
      (= silence > CMD_SOURCE_TIMEOUT du joystick, donc watchdog actif)
    - comparez angular_z de /avoidance_cmd_vel à /cmd_vel à la même
      t_rel_s pour voir si twist_mux a bien basculé la priorité
"""

import argparse
import csv
import math
import os
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool, String, Float64
from nav_msgs.msg import Odometry


# ──────────────────────────────────────────────────────────────────────────
#  Topics ciblés (légers) — la caméra et les images debug sont exclues
#  exprès. Odometry/Point ajoutés pour permettre :
#    - la validation TTC (vitesse réelle du robot, pas seulement la
#      commande envoyée — un Husky chargé n'atteint pas instantanément
#      la vitesse demandée)
#    - la validation distance réelle/calculée (pose robot de référence)
# ──────────────────────────────────────────────────────────────────────────
TOPIC_TYPES = {
    # ── Communs YOLO et Farneback ───────────────────────────────────────
    '/cmd_vel':                      Twist,
    '/cmd_vel_in':                   Twist,
    '/joy_teleop/cmd_vel':           Twist,
    '/avoidance_cmd_vel':            Twist,
    '/emergency_stop':               Bool,
    '/ground_detections':            String,    # YOLO uniquement
    '/odometry/filtered':            Odometry,
    '/robot_ground_position':        Point,
    # ── Farneback uniquement ────────────────────────────────────────────
    '/movement_command':             String,    # état texte du nœud
    # ── perf_monitor.py — Float64 désérialisés en String (même chemin CSV)
    '/perf/cpu_percent':             Float64,
    '/perf/gpu_percent':             Float64,
    '/perf/ram_mb':                  Float64,
    '/perf/inference_period_ms':     Float64,
    '/perf/joy_to_cmd_latency_ms':   Float64,
}

# Nom de fichier safe (remplace les '/' par '_')
def safe_name(topic: str) -> str:
    return topic.strip('/').replace('/', '_') + '.csv'


def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader.open(storage_options, converter_options)
    return reader


def get_available_topics(bag_path: str) -> dict:
    """Retourne {topic_name: type_string} réellement présents dans le bag."""
    reader = open_reader(bag_path)
    topics_and_types = reader.get_all_topics_and_types()
    return {t.name: t.type for t in topics_and_types}


def extract(bag_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    present = get_available_topics(bag_path)
    targets = {t: cls for t, cls in TOPIC_TYPES.items() if t in present}
    missing = [t for t in TOPIC_TYPES if t not in present]

    if missing:
        print(f'[info] Topics absents de ce bag (ignorés) : {missing}')
    if not targets:
        print('[erreur] Aucun topic ciblé trouvé dans ce bag. Topics disponibles :')
        for t, ty in present.items():
            print(f'    {t}  ({ty})')
        sys.exit(1)

    print(f'[info] Extraction des topics : {list(targets.keys())}')

    reader = open_reader(bag_path)
    storage_filter = rosbag2_py.StorageFilter(topics=list(targets.keys()))
    reader.set_filter(storage_filter)

    # Un writer CSV par topic + un buffer pour la timeline fusionnée
    # (Odometry et Point ont leur propre schéma de colonnes et ne sont
    # PAS injectés dans timeline_merged.csv, qui reste au format fixe
    # Twist/Bool/String — sinon la fusion casserait les colonnes.)
    writers = {}
    files = {}
    for topic in targets:
        fpath = os.path.join(out_dir, safe_name(topic))
        f = open(fpath, 'w', newline='')
        files[topic] = f
        w = csv.writer(f)
        if targets[topic] is Twist:
            w.writerow(['t_ns', 't_rel_s', 'linear_x', 'linear_y', 'linear_z',
                       'angular_x', 'angular_y', 'angular_z'])
        elif targets[topic] is Bool:
            w.writerow(['t_ns', 't_rel_s', 'data'])
        elif targets[topic] is String:
            w.writerow(['t_ns', 't_rel_s', 'data'])
        elif targets[topic] is Odometry:
            w.writerow(['t_ns', 't_rel_s', 'x', 'y', 'yaw_rad', 'vx', 'vy', 'vyaw'])
        elif targets[topic] is Point:
            w.writerow(['t_ns', 't_rel_s', 'x', 'y', 'z'])
        elif targets[topic] is Float64:
            w.writerow(['t_ns', 't_rel_s', 'data'])
        writers[topic] = w

    merged_path = os.path.join(out_dir, 'timeline_merged.csv')
    merged_file = open(merged_path, 'w', newline='')
    merged_writer = csv.writer(merged_file)
    merged_writer.writerow(
        ['t_ns', 't_rel_s', 'topic', 'linear_x', 'angular_z', 'bool_or_string_data'])

    t_start = None
    count_per_topic = {t: 0 for t in targets}
    merged_rows = []  # on accumule pour trier par temps avant écriture finale

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in targets:
            continue
        if t_start is None:
            t_start = t_ns
        t_rel = (t_ns - t_start) / 1e9

        msg = deserialize_message(data, targets[topic])
        count_per_topic[topic] += 1

        if isinstance(msg, Twist):
            writers[topic].writerow([
                t_ns, f'{t_rel:.4f}',
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z,
            ])
            merged_rows.append((t_ns, t_rel, topic, msg.linear.x, msg.angular.z, ''))

        elif isinstance(msg, Bool):
            writers[topic].writerow([t_ns, f'{t_rel:.4f}', msg.data])
            merged_rows.append((t_ns, t_rel, topic, '', '', msg.data))

        elif isinstance(msg, String):
            writers[topic].writerow([t_ns, f'{t_rel:.4f}', msg.data])
            merged_rows.append((t_ns, t_rel, topic, '', '', msg.data))

        elif isinstance(msg, Odometry):
            # Conversion quaternion → yaw, identique à cb_odom() dans
            # emergency_stop_node.py, pour rester cohérent avec ce que
            # le nœud utilise réellement en interne (TTC, recovery).
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            writers[topic].writerow([
                t_ns, f'{t_rel:.4f}',
                msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                msg.twist.twist.angular.z,
            ])
            # Pas d'ajout à merged_rows : schéma incompatible (pose, pas
            # une commande Twist) — reste uniquement dans odometry_filtered.csv

        elif isinstance(msg, Point):
            writers[topic].writerow([t_ns, f'{t_rel:.4f}', msg.x, msg.y, msg.z])
            # Idem : pas dans la timeline fusionnée.

        elif isinstance(msg, Float64):
            writers[topic].writerow([t_ns, f'{t_rel:.4f}', msg.data])
            # Pas dans la timeline fusionnée (schéma scalaire)

    for f in files.values():
        f.close()

    # Tri chronologique global puis écriture de la timeline fusionnée
    merged_rows.sort(key=lambda r: r[0])
    for row in merged_rows:
        t_ns, t_rel, topic, lx, az, other = row
        merged_writer.writerow([t_ns, f'{t_rel:.4f}', topic, lx, az, other])
    merged_file.close()

    print('\n[résumé]')
    for t, c in count_per_topic.items():
        print(f'    {t:<28} {c} messages')
    print(f'\n[ok] CSV par topic + timeline_merged.csv écrits dans : {out_dir}')


# ──────────────────────────────────────────────────────────────────────────
#  Chemin par défaut — modifiez cette ligne si le bag change d'emplacement.
#  Le script fonctionne aussi sans rien changer ici : passez simplement un
#  autre chemin en argument (cf. docstring en haut du fichier).
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_BAG_PATH = '/home/imr2204/Desktop/Erwan/clearpath_simulator/test_full_20260623_100100'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag_path', nargs='?', default=DEFAULT_BAG_PATH,
                        help=f'Dossier du bag (contient metadata.yaml + .db3). '
                             f'Défaut : {DEFAULT_BAG_PATH}')
    parser.add_argument('--out', default=None,
                        help='Dossier de sortie (défaut: ./bag_extracted_<nom_du_bag>)')
    args = parser.parse_args()

    if not os.path.isdir(args.bag_path):
        print(f'[erreur] Dossier introuvable : {args.bag_path}')
        sys.exit(1)

    out_dir = args.out or f'./bag_extracted_{os.path.basename(args.bag_path.rstrip("/"))}'

    print(f'[info] Bag source : {args.bag_path}')
    print(f'[info] Sortie     : {out_dir}\n')

    extract(args.bag_path, out_dir)


if __name__ == '__main__':
    main()