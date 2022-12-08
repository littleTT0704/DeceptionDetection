import time
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

N_VID = 60
root_dir = "../data"


class FaceDataset(Dataset):
    def __init__(self, train: bool, device=None):
        self.device = device
        self.root_path = os.path.join(root_dir, f"face")
        self.data = []
        for filename in os.listdir(self.root_path):
            _, category, end = filename.split("_")
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
    dataset = FaceDataset(False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for vid, lab in dataloader:
        print(vid.shape, vid.dtype)
        print(lab.shape, lab.dtype)
    print(f"time spent {time.time() - t:.2f}s")
