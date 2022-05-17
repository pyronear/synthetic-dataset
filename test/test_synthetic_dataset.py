# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
import random
import tempfile
import glob
from syntheticdataset.make_set import make_one_set


class SyntheticDatasetTester(unittest.TestCase):
    def test_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as root:

            smoke_video_file = "test/videos/test_smoke.mp4"
            background_file = "test/videos/test_bg.mp4"
            fx = random.randint(1, 9) / 10  # random in [0.1, 0.9]
            fy = random.randint(1, 9) / 10  # random in [0.1, 0.9]
            opacity = random.randint(4, 10) / 10  # random in [0.4, 1.0]
            smoke_speed = random.randint(3, 10)  # random in [3, 10]

            make_one_set(
                smoke_video_file,
                background_file,
                root=root + "pyro_dataset",
                set_idx=0,
                fx=fx,
                fy=fy,
                opacity=opacity,
                smoke_speed=smoke_speed,
                train=True,
                save_mask=True,
                save_bbox=True,
            )

            self.assertGreater(len(glob.glob(root + "pyro_dataset/images/train/*")), 20)
            self.assertGreater(len(glob.glob(root + "pyro_dataset/labels/train/*")), 20)
            self.assertGreater(len(glob.glob(root + "pyro_dataset/mask/train/*")), 20)


if __name__ == "__main__":
    unittest.main()
