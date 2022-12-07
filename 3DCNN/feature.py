from utils import DW, DH, N_VID, root_dir
from model import CNN3D
import torch
import numpy as np
import time
import os


def extract_video_features():
    device = torch.device("cpu")
    m = CNN3D().to(device)
    m.load_state_dict(torch.load("./model.pt"))
    m.eval()

    data_dir = os.path.join(root_dir, f"{DW}_{DH}/")
    output_dir = os.path.join(root_dir, "Video")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    d = 256
    truth = np.zeros((N_VID, d))
    lie = np.zeros((N_VID + 1, d))

    for filename in os.listdir(data_dir):
        category, idx, end = filename.split("_")
        offset = int(end.split(".")[0])
        idx = int(idx) - 1
        if offset == 0:
            input_file = os.path.join(data_dir, filename)
            vid = (
                torch.from_numpy(np.load(input_file) / 255)
                .float()
                .to(device)
                .unsqueeze(0)
            )
            feature = m(vid, True)
            feature = feature.detach().numpy()[0]
            if category == "truth":
                truth[idx] = feature
            elif category == "lie":
                lie[idx] = feature
            else:
                print("error")
                return
    np.save(os.path.join(output_dir, "truth.npy"), truth)
    np.save(os.path.join(output_dir, "lie.npy"), lie)


if __name__ == "__main__":
    extract_video_features()
