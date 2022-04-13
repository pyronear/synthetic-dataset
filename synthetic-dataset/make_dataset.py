import glob
from make_set import make_one_set
import random


def main(args):

    smoke_videos = glob.glob('videos/smoke/*')
    background_videos = glob.glob('videos/background/*')

    if args.set > 0:
        # Make n set
        set_idx = 0
        for i in range(args.set):
            smoke_video_file = random.sample(smoke_videos, 1)[0]
            background_file = random.sample(background_videos, 1)[0]
            fx = random.randint(1, 9) / 10  # random in [0.1, 0.9]
            fy = random.randint(1, 9) / 10  # random in [0.1, 0.9]
            opacity = random.randint(4, 10) / 10  # random in [0.4, 1.0]
            smoke_speed = random.randint(3, 10)  # random in [3, 10]

            make_one_set(smoke_video_file, background_file, set_idx, fx=fx,
                         fy=fy, opacity=opacity, smoke_speed=smoke_speed)

            set_idx += 1

    else:
        # Make All

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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Make syanthetic dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--set', default=0, type=int, help='number of set to create')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
