"""Run the frozen external model locally. Never sends data over a network."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from learned_possession import features, predict, LABELS, FEATURE_VERSION
from barca4ever import metadata, point, write_csv
from games import require, GAMES, GAME, source_frames
import joblib


def load_tracking(meta, raw, n, pitch):
    teams = metadata(meta)
    players = np.full((n, len(teams), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    scale = np.asarray(pitch)
    count = 0
    with open(raw, encoding="utf-8-sig") as stream:
        for count, line in enumerate(stream, 1):
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != count or count > n:
                raise ValueError("Tracking rows are malformed or out of sequence")
            groups = parts[1].split(";")
            if len(groups) != len(teams):
                raise ValueError("Track count disagrees with metadata")
            for slot, group in enumerate(groups):
                xy = point(group)
                if xy is not None:
                    players[count - 1, slot] = (xy - 0.5) * scale
            xy = point(parts[2])
            if xy is not None:
                ball[count - 1] = (xy - 0.5) * scale
    if count != n:
        raise ValueError("Tracking frame count disagrees with submission contract")
    return players, teams, ball


def tracking_blocks(players, fps):
    """Separate only long tracking outages; no game clock or known match anchors.

    Ten seconds without any observed player is an engineering reset rule, not a
    football dead-ball detector. Do not assign D to ordinary occlusion periods.
    """
    empty = ~np.isfinite(players).all(axis=2).any(axis=1)
    changes = np.flatnonzero(np.r_[True, empty[1:] != empty[:-1], True])
    gaps = [(a, b) for a, b in zip(changes[:-1], changes[1:]) if empty[a] and b - a >= fps * 10]
    blocks, cursor = [], 0
    for start, end in gaps:
        if start > cursor:
            blocks.append((cursor, start))
        cursor = end
    if cursor < len(players):
        blocks.append((cursor, len(players)))
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--model", default="models/external_v1.joblib")
    parser.add_argument("--out", default="evaluation/results/Barca4Ever.csv")
    args = parser.parse_args()
    # Only load the team's own locally trained artifact; joblib is not a safe format
    # for untrusted models downloaded from strangers.
    artifact = joblib.load(args.model)
    if artifact["feature_version"] == FEATURE_VERSION:
        feature_fn = features
    elif artifact["feature_version"] == 2:
        from possession_v2 import features as feature_fn
    else:
        raise ValueError("Feature version mismatch")
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    pitch, fps = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
    players, teams, ball = load_tracking(meta, raw, source_frames(), pitch)
    labels = np.full(len(ball), "X", dtype="<U1")
    for start, end in tracking_blocks(players, fps):
        x = feature_fn(players[start:end], teams, ball[start:end], pitch, fps)
        labels[start:end] = LABELS[predict(artifact["model"], x, artifact["penalty"])]
    write_csv(args.out, labels)
    manifest = {"team": "Barca4Ever", "rows": len(labels), "feature_version": artifact['feature_version'],
                "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
                "training": "IDSSE external matches only; no fitting during inference"}
    Path(args.out + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Local CSV and manifest written. No labels, coordinates or private paths printed.")


if __name__ == "__main__":
    main()
