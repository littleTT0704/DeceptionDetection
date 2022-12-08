import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def shape_to_numpy(shape):
    # print(shape.num_parts)
    res = np.zeros((shape.num_parts, 2), dtype=int)
    for i, pt in enumerate(shape.parts()):
        res[i, 0] = pt.x
        res[i, 1] = pt.y
    return res


predictor_path = "shape_predictor_68_face_landmarks.dat"

N_VID = 60
N_MARK = 68
root_dir = "../data"


def extract_landmark(
    idx: int,
    deceptive: bool = True,
):
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    n = N_VID + deceptive
    if idx < 1 or idx > n:
        print(f"index {idx} out of bound [1, {n}]")
        return

    filename = os.path.join(
        root_dir,
        f"face/{['truth', 'lie'][deceptive]}_{idx:02d}.npy",
    )
    print(filename)
    vid = np.load(filename).transpose((1, 2, 3, 0))
    l, h, w, _ = vid.shape
    res = np.zeros((l, N_MARK, 2), dtype=int)
    default = np.ones((N_MARK, 2), dtype=int) * (h // 2)
    c = 0
    for i in range(l):
        dets = detector(vid[i], 1)
        if len(dets) == 0:
            res[i] = default
        else:
            res[i] = shape_to_numpy(predictor(vid[i], dets[0]))
            c += 1
    print(f"{c} / {l} valid faces")

    output_dir = os.path.join(root_dir, f"landmark/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filename = os.path.join(
        output_dir, f"{['truth', 'lie'][deceptive]}_{idx:02d}.npy"
    )
    np.save(output_filename, res)


def extract_all_landmarks():
    for i in range(N_VID):
        t = time.time()
        extract_landmark(i + 1, False)
        print(f"time spent {time.time() - t:.2f}s")
    for i in range(N_VID + 1):
        t = time.time()
        extract_landmark(i + 1, True)
        print(f"time spent {time.time() - t:.2f}s")


class LandmarkDataset(Dataset):
    def __init__(self, train: bool, device=None):
        self.device = device
        self.root_path = os.path.join(root_dir, f"landmark/")
        if not os.path.exists(self.root_path):
            extract_all_landmarks()
        self.data = []
        for filename in os.listdir(self.root_path):
            category, end = filename.split("_")
            idx = int(end.split(".")[0])
            if train ^ (idx <= 8):
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
        vid = np.load(file_path).reshape((90, -1)) / 128
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
    dataset = LandmarkDataset(False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for vid, lab in dataloader:
        print(vid.shape, vid.dtype)
        print(lab.shape, lab.dtype)
        exit(0)
    print(f"time spent {time.time() - t:.2f}s")
