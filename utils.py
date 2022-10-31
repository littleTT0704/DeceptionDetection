from typing import Tuple
import imageio
import numpy as np
import cv2
import os
import logging

logger = logging.getLogger("video")
logger.setLevel(logging.DEBUG)

N_VID = 60
DW = 80
DH = 60


def read_video(
    idx: int,
    deceptive: bool = True,
    length: int = 90,
    width: int = DW,
    height: int = DH,
) -> np.ndarray:
    n = N_VID + deceptive
    if idx < 1 or idx > n:
        logger.error(f"index {idx} out of bound [1, {n}]")
        return

    if not os.path.exists(f"./data/{width}_{height}/"):
        os.mkdir(f"./data/{width}_{height}/")

    cache = f"./data/{width}_{height}/{['truth', 'lie'][deceptive]}_{idx:03d}.npy"
    if os.path.exists(cache):
        res = np.load(cache)
        logger.debug(f"Cache loaded from {cache}")
        return res

    filename = f"./data/Clips/{['Truthful', 'Deceptive'][deceptive]}/trial_{['truth', 'lie'][deceptive]}_{idx:03d}.mp4"
    logger.debug(filename)
    vid = imageio.get_reader(filename, "ffmpeg")
    metadata = vid.get_meta_data()
    n_frames = int(metadata["fps"] * metadata["duration"])
    W, H = metadata["size"]
    logger.debug(f"frame size: {W} * {H}, number of frames: {n_frames}")

    res = np.zeros((length, height, width, 3), dtype=np.uint8)
    idxs = np.arange(0, n_frames, n_frames / length, dtype=int)[:length]
    for i, idx in enumerate(idxs):
        res[i] = cv2.resize(vid.get_data(idx), (width, height))
    res = res.transpose((3, 0, 1, 2))
    np.save(cache, res)
    vid.close()
    return res


def load_videos(
    length: int = 90,
    width: int = DW,
    height: int = DH,
) -> Tuple[np.ndarray]:
    truth = np.zeros((N_VID, 3, length, height, width), dtype=np.uint8)
    deception = np.zeros((N_VID + 1, 3, length, height, width), dtype=np.uint8)
    for i in range(N_VID):
        truth[i] = read_video(i + 1, False, length, width, height)
    for i in range(N_VID + 1):
        deception[i] = read_video(i + 1, True, length, width, height)
    return truth / 255, deception / 255


def load_features():
    features = np.genfromtxt(
        "./data/Annotation/All_Gestures_Deceptive and Truthful.csv", delimiter=","
    )[1:, 1:-1]
    return (
        features[N_VID + 1 :],
        features[: N_VID + 1],
    )


if __name__ == "__main__":
    import time

    t = time.time()
    load_videos()
    print(f"time spent {time.time() - t:.2f}s")
