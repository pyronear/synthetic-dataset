import glob
from make_set import make_one_set
import random


smoke_videos = glob.glob('DS_video/smoke/*')
background_videos = glob.glob('DS_video/background/*')

print(len(smoke_videos), len(background_videos))

set_idx = 0
for smoke_video_file in smoke_videos:
    for background_file in background_videos:
        fx = random.randint(1, 9) / 10  # random in [0.1, 0.9]
        fy = random.randint(1, 9) / 10  # random in [0.1, 0.9]
        opacity = random.randint(4, 10) / 10  # random in [0.4, 1.0]
        smoke_speed = random.randint(3, 10)  # random in [3, 10]

        make_one_set(smoke_video_file, background_file, set_idx, fx=fx,
                     fy=fy, opacity=opacity, smoke_speed=smoke_speed)

        set_idx += 1
