import imageio
import numpy as np
import cv2
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional

N_VID = 60
DW = 80
DH = 60
root_dir = "../data"


def extract_video(
    idx: int,
    deceptive: bool = True,
    length: int = 90,
    width: Optional[int] = DW,
    height: Optional[int] = DH,
):
    n = N_VID + deceptive
    if idx < 1 or idx > n:
        print(f"index {idx} out of bound [1, {n}]")
        return

    filename = os.path.join(
        root_dir,
        f"Clips/{['Truthful', 'Deceptive'][deceptive]}/trial_{['truth', 'lie'][deceptive]}_{idx:03d}.mp4",
    )
    print(filename)
    vid = imageio.get_reader(filename, "ffmpeg")
    metadata = vid.get_meta_data()
    n_frames = int(metadata["fps"] * metadata["duration"])
    W, H = metadata["size"]
    print(f"frame size: {W} * {H}, number of frames: {n_frames}")
    if width is None or height is None:
        width, height = W, H

    output_dir = os.path.join(root_dir, f"{width}_{height}/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    prefix = length * (n_frames // length)
    entire = np.zeros((prefix, height, width, 3), dtype=np.uint8)
    for i, frame in zip(range(prefix), vid):
        entire[i] = cv2.resize(frame, (width, height))

    idxs = np.arange(0, n_frames, n_frames // length, dtype=int)[:length]
    offset = 0
    while idxs[-1] + offset < n_frames and offset < n_frames // length:
        res = entire[idxs]
        res = res.transpose((3, 0, 1, 2))
        cache = os.path.join(
            output_dir, f"{['truth', 'lie'][deceptive]}_{idx:02d}_{offset:02d}.npy"
        )
        np.save(cache, res)
        offset += 1
    vid.close()


def extract_all_videos(
    length: int = 90,
    width: int = DW,
    height: int = DH,
):
    for i in range(N_VID):
        t = time.time()
        extract_video(i + 1, False, length, width, height)
        print(f"time spent {time.time() - t:.2f}s")
    for i in range(N_VID + 1):
        t = time.time()
        extract_video(i + 1, True, length, width, height)
        print(f"time spent {time.time() - t:.2f}s")


class VideoDataset(Dataset):
    def __init__(self, train: bool, device=None):
        self.device = device
        self.root_path = os.path.join(root_dir, f"{DW}_{DH}/")
        if not os.path.exists(self.root_path):
            extract_all_videos()
        self.data = []
        for filename in os.listdir(self.root_path):
            category, idx, end = filename.split("_")
            offset = int(end.split(".")[0])
            idx = int(idx)
            if offset == 0 and (train ^ (idx <= 8)):
                self.data.append(
                    (os.path.join(self.root_path, filename), category, idx)
                )
        self.mapping = {"lie": 1, "truth": 0}

        # self.features = np.genfromtxt(
        #     "./data/Annotation/All_Gestures_Deceptive and Truthful.csv", delimiter=","
        # )[1:, 1:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, category, idx = self.data[index]
        vid = np.load(file_path) / 255
        # feature_idx = N_VID + idx if category == "truth" else idx - 1
        # feature = self.features[feature_idx]
        label = np.zeros((2,))
        label[self.mapping[category]] = 1

        return (
            torch.from_numpy(vid).float().to(self.device),
            # torch.from_numpy(feature).float().to(self.device),
            torch.from_numpy(label).float().to(self.device),
        )


if __name__ == "__main__":
    t = time.time()
    dataset = VideoDataset(False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for vid, lab in dataloader:
        print(vid.shape, vid.dtype)
        print(lab.shape, lab.dtype)
    print(f"time spent {time.time() - t:.2f}s")
