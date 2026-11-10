"""IDSSE adapter: labels remain separate from positional feature inputs."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "external_data" / "idsse"
CACHE = ROOT / "external_cache"
SPLITS = {"train": ["J03WOH", "J03WOY", "J03WPY", "J03WQQ"],
          "validation": ["J03WR9"], "test": ["J03WMX", "J03WN1"]}


def paths(match):
    meta = list(RAW.glob("*matchinformation*{}*.xml".format(match)))
    raw = list(RAW.glob("*positions_raw*{}*.xml".format(match)))
    if len(meta) != 1 or len(raw) != 1:
        raise FileNotFoundError("External match {} is missing or ambiguous; finish download_external.py first".format(match))
    return meta[0], raw[0]


def parse(match):
    meta_path, raw_path = paths(match)
    meta = ET.parse(meta_path).getroot()
    general = meta.find(".//General")
    home, away = general.get("HomeTeamId"), general.get("GuestTeamId")
    env = meta.find(".//Environment")
    pitch = np.array([float(env.get("PitchX")), float(env.get("PitchY"))])
    teams = {home: 0, away: 1}
    sets = []
    # Clear each frame immediately; retain only numeric tuples for current track.
    ctx = None
    rows = []
    for kind, elem in ET.iterparse(raw_path, events=("start", "end")):
        if kind == "start" and elem.tag == "FrameSet":
            ctx = dict(elem.attrib)
            rows = []
        elif kind == "end" and elem.tag == "Frame":
            if ctx is not None and (ctx.get("TeamId") in teams or "BallStatus" in elem.attrib):
                a = elem.attrib
                rows.append((int(a["N"]), float(a.get("X", "nan")), float(a.get("Y", "nan")),
                             int(a.get("BallStatus", -1)), int(a.get("BallPossession", -1))))
            elem.clear()
        elif kind == "end" and elem.tag == "FrameSet":
            if rows:
                sets.append((ctx, np.asarray(rows)))
            elem.clear()
            ctx = None
    outputs = []
    for section in ["firstHalf", "secondHalf"]:
        current = [(ctx, values) for ctx, values in sets if ctx["GameSection"] == section]
        balls = [(c, a) for c, a in current if np.any(a[:, 3] >= 0)]
        if len(balls) != 1:
            raise ValueError("Expected one ball stream per half, got {}".format(len(balls)))
        ball_rows = balls[0][1]
        ids = ball_rows[:, 0].astype(int)
        if np.any(np.diff(ids) != 1):
            raise ValueError("Noncontiguous external ball frames")
        ball = ball_rows[:, 1:3].astype(np.float32)
        # Sportec labels: inactive->D, active+home->H, active+away->A, unknown->X.
        labels = np.full(len(ball), 3, dtype=np.int8)
        labels[ball_rows[:, 3] == 0] = 2
        for possession, code in [(1, 0), (2, 1)]:
            labels[(ball_rows[:, 3] == 1) & (ball_rows[:, 4] == possession)] = code
        people = [(c, a) for c, a in current if c.get("TeamId") in teams]
        players = np.full((len(ball), len(people), 2), np.nan, dtype=np.float32)
        team = []
        for slot, (ctx, values) in enumerate(people):
            ix = values[:, 0].astype(int) - ids[0]
            valid = (ix >= 0) & (ix < len(ball))
            players[ix[valid], slot] = values[valid, 1:3]
            team.append(teams[ctx["TeamId"]])
        outputs.append(dict(players=players, teams=np.asarray(team), ball=ball,
                            labels=labels, pitch=pitch, frame_ids=ids))
    return outputs


def load(match):
    CACHE.mkdir(exist_ok=True)
    targets = [CACHE / (match + "_{}.npz".format(half)) for half in [1, 2]]
    if not all(t.exists() for t in targets):
        for target, values in zip(targets, parse(match)):
            np.savez_compressed(target, **values)
    return [dict(np.load(target)) for target in targets]
