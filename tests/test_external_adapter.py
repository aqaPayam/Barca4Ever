import unittest
from unittest.mock import patch
import numpy as np
from external_idsse import parse
from test_barca4ever import synthetic_folder


class ExternalAdapterTests(unittest.TestCase):
    def test_home_away_and_dead_labels_are_not_in_xy(self):
        with synthetic_folder() as root:
            meta = root / "meta.xml"
            meta.write_text('<root><General HomeTeamId="h" GuestTeamId="a"/>'
                            '<Environment PitchX="105" PitchY="68"/></root>')
            raw = root / "positions.xml"
            pieces = ["<root>"]
            for section in ["firstHalf", "secondHalf"]:
                for team in ["a", "h", "ball"]:
                    pieces.append('<FrameSet GameSection="{}" TeamId="{}">'.format(section, team))
                    for i in range(3):
                        extra = (' BallStatus="{}" BallPossession="{}"'.format(
                            0 if i == 2 else 1, i + 1) if team == "ball" else "")
                        pieces.append('<Frame N="{}" X="1" Y="2"{}/>'.format(i + 100, extra))
                    pieces.append("</FrameSet>")
            raw.write_text("".join(pieces) + "</root>")
            with patch("external_idsse.paths", return_value=(meta, raw)):
                halves = parse("SYNTHETIC")
            for half in halves:
                np.testing.assert_array_equal(half["labels"], [0, 1, 2])
                np.testing.assert_array_equal(half["teams"], [1, 0])
                self.assertEqual(half["ball"].shape, (3, 2))
                np.testing.assert_array_equal(half["ball"], [[1, 2]] * 3)


if __name__ == "__main__":
    unittest.main()
