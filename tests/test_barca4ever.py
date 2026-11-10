"""Synthetic correctness tests only. No provided match, labels or video read."""
import unittest
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np

from barca4ever import Config, decode, metadata, observations, write_csv


@contextmanager
def synthetic_folder():
    # Windows Python's private-temp ACL conflicts with the managed sandbox.
    root = Path(__file__).resolve().parent
    folder = root / ("synthetic_" + uuid4().hex)
    folder.mkdir(mode=0o777)
    try:
        yield folder
    finally:
        assert folder.parent == root
        for item in folder.iterdir():
            item.unlink()
        folder.rmdir()


class PossessionTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(confirmation_s=0.4, missing_hold_s=0.2)

    def test_brief_deflection_does_not_switch(self):
        self.assertEqual(decode("HHHHHAHHHH", 10, self.config), list("HHHHHHHHHH"))

    def test_controlled_turnover_backdates_to_evidence(self):
        self.assertEqual(decode("HHHHHAAAAA", 10, self.config), list("HHHHHAAAAA"))

    def test_missing_is_not_dead_ball(self):
        self.assertEqual(decode("HHHHXXXX", 10, self.config), list("HHHHHHXX"))

    def test_unknown_initial_state(self):
        self.assertEqual(decode("XXHH", 10, self.config), list("XXXX"))

    def test_stoppage_resets_owner_until_restart(self):
        self.assertEqual(decode("HHHHDDXXAAAA", 10, self.config), list("HHHHDDDDAAAA"))

    def test_evidence_gap_breaks_confirmation(self):
        self.assertEqual(decode("HHHHAAAXAAA", 10, self.config), list("HHHHHHHHHHH"))

    def test_negative_or_nan_parameters_rejected(self):
        for value in [-1, float("nan")]:
            with self.assertRaises(ValueError):
                decode([], 25, Config(confirmation_s=value))

    def test_stream_and_serialization(self):
        with synthetic_folder() as folder:
            root = Path(folder)
            meta = root / "meta.xml"
            meta.write_text('<root><Team id="h"/><Team id="a"/>'
                            '<Player teamId="h"/><Player teamId="a"/></root>')
            slots = metadata(meta)
            raw = root / "tracking.txt"
            groups = "0.1,0.5,0;0.9,0.5,0"
            balls = ["0.1,0.5,0"] * 4 + ["NaN,NaN,NaN"] * 3 + ["1.1,0.5,0"]
            raw.write_text("".join("{}:{}:{}\n".format(i + 1, groups, ball)
                                   for i, ball in enumerate(balls)))
            evidence = list(observations(raw, slots, 8, (100, 60), self.config))
            self.assertEqual(evidence, list("HHHHXXXD"))
            labels = decode(evidence, 10, self.config)
            csv = root / "result.csv"
            write_csv(csv, labels)
            self.assertEqual(csv.read_text().splitlines(),
                             ["frame,label"] + ["{},{}".format(i, x) for i, x in enumerate(labels)])
            with self.assertRaises(ValueError):
                list(observations(raw, slots, 9, (100, 60), self.config))
            raw.write_text("2:{}:0.1,0.5,0\n".format(groups))
            with self.assertRaises(ValueError):
                list(observations(raw, slots, 1, (100, 60), self.config))

    def test_distance_uses_metres_and_ambiguity_abstains(self):
        with synthetic_folder() as folder:
            raw = Path(folder) / "tracking.txt"
            # Home is 1 m away, away is 6 m away despite smaller normalized y gap.
            raw.write_text("1:0.51,0.5,0;0.5,0.6,0:0.5,0.5,0\n"
                           "2:0.49,0.5,0;0.51,0.5,0:0.5,0.5,0\n")
            self.assertEqual(list(observations(raw, np.array([0, 1]), 2,
                                               (100, 60), self.config)), ["H", "X"])


if __name__ == "__main__":
    unittest.main()
