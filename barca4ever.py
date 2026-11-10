"""Untuned tracking-only possession prototype. No ground-truth or video inputs.

Defaults are engineering assumptions, NOT validated estimates of football control.
Only external data may be used to train/calibrate them. Run locally on gated data.
"""
import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, require, source_frames


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 2.0
    team_margin_m: float = 0.5
    confirmation_s: float = 0.4
    missing_hold_s: float = 2.0
    boundary_margin_m: float = 0.5

    def validate(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Configuration values must be finite and nonnegative")
        if self.control_radius_m == 0 or self.confirmation_s == 0:
            raise ValueError("Control radius and confirmation duration must be positive")


def metadata(path):
    root = ET.parse(path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2 or None in teams or len(set(teams)) != 2:
        raise ValueError("Expected exactly two distinct teams in EPTS metadata")
    slots = []
    for player in root.iter("Player"):
        if player.get("teamId") not in teams:
            raise ValueError("Track slot has an unknown team")
        slots.append(teams.index(player.get("teamId")))
    if not slots:
        raise ValueError("No track slots found")
    return np.asarray(slots)


def point(group):
    values = group.strip().split(",")
    if len(values) < 2:
        raise ValueError("Tracking group needs x and y coordinates")
    xy = np.array([float(values[0]), float(values[1])])
    return xy if np.isfinite(xy).all() else None


def observations(path, slots, n_frames, pitch, config):
    """Yield H/A for confident proximity, D for outside, X for no evidence.

    Strict sequential indexing catches missing/duplicate rows instead of silently
    shifting labels. Player slots are tracklets, never treated as player IDs.
    """
    scale = np.asarray(pitch)
    count = 0
    with open(path, encoding="utf-8-sig") as stream:
        for count, line in enumerate(stream, 1):
            parts = line.strip().split(":")
            if len(parts) != 3 or int(parts[0]) != count or count > n_frames:
                raise ValueError("Invalid or nonsequential tracking row {}".format(count))
            groups = parts[1].split(";")
            if len(groups) != len(slots):
                raise ValueError("Track slot count mismatch at row {}".format(count))
            ball = point(parts[2])
            if ball is None:
                yield "X"
                continue
            ball = ball * scale
            margin = config.boundary_margin_m
            if np.any(ball < -margin) or np.any(ball > scale + margin):
                yield "D"
                continue
            nearest = [float("inf"), float("inf")]
            for team, group in zip(slots, groups):
                player = point(group)
                if player is not None:
                    distance = float(np.linalg.norm(player * scale - ball))
                    nearest[team] = min(nearest[team], distance)
            winner = int(nearest[1] < nearest[0])
            if (nearest[winner] <= config.control_radius_m
                    and nearest[1 - winner] - nearest[winner] >= config.team_margin_m):
                yield "HA"[winner]
            else:
                yield "X"
    if count != n_frames:
        raise ValueError("Expected {} tracking rows, received {}".format(n_frames, count))


def decode(evidence, fps, config):
    """Hysteresis with retrospective assignment of confirmed runs.

    X preserves an established team for short gaps, but never invents dead ball.
    D ends ownership immediately. A restart still requires control confirmation.
    No match-specific time intervals, kickoff anchors, or labels are consulted.
    """
    config.validate()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    required = max(1, math.ceil(config.confirmation_s * fps))
    hold = math.ceil(config.missing_hold_s * fps)
    labels = []
    owner = "X"
    candidate = None
    candidate_start = candidate_count = missing = 0
    for i, value in enumerate(evidence):
        if value not in ("H", "A", "D", "X"):
            raise ValueError("Unknown evidence label")
        if value == "D":
            owner, candidate, candidate_count, missing = "D", None, 0, 0
            labels.append("D")
            continue
        if value == "X":
            candidate, candidate_count = None, 0
            missing += 1
            # Dead ball persists until affirmative evidence of a restart.
            labels.append(owner if owner == "D" or missing <= hold else "X")
            continue
        missing = 0
        labels.append(owner)
        if value == owner:
            candidate, candidate_count = None, 0
            continue
        if value != candidate:
            candidate, candidate_start, candidate_count = value, i, 0
        candidate_count += 1
        if candidate_count >= required:
            owner = value
            labels[candidate_start:i + 1] = [owner] * (i + 1 - candidate_start)
            candidate, candidate_count = None, 0
    return labels


def write_csv(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("frame,label\n")
        for frame, label in enumerate(labels):
            stream.write("{},{}\n".format(frame, label))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Root containing tactical/ (local only)")
    parser.add_argument("--config", help="Optional JSON config calibrated on EXTERNAL data only")
    parser.add_argument("--out", default="evaluation/results/Barca4Ever.csv")
    args = parser.parse_args()
    config = Config(**json.loads(Path(args.config).read_text()) if args.config else {})
    config.validate()
    meta, raw = require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    labels = decode(observations(raw, metadata(meta), source_frames(),
                                 GAMES[GAME]["pitch"], config), GAMES[GAME]["fps"], config)
    write_csv(args.out, labels)
    # Reproducibility metadata: no source coordinates, timestamps or private paths.
    manifest = {"team": "Barca4Ever", "config": asdict(config),
                "parameter_provenance": "Unfitted engineering defaults unless externally calibrated",
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
                "rows": len(labels), "training": "None performed by this program"}
    Path(str(args.out) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote submission and manifest locally. No quality evaluation performed.")


if __name__ == "__main__":
    main()
