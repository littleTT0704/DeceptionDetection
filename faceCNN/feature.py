from utils import root_dir, N_VID
from model import FaceCNN
import torch
import numpy as np
import time
import os


def extract_face_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = FaceCNN().to(device)
    m.load_state_dict(torch.load("./model.pt"))
    m.eval()

    data_dir = os.path.join(root_dir, f"face/")
    output_dir = os.path.join(root_dir, "Face/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    d = 64
    truth = np.zeros((N_VID, d))
    lie = np.zeros((N_VID + 1, d))

    for filename in os.listdir(data_dir):
        category, end = filename.split("_")
        idx = int(end.split(".")[0]) - 1

        input_file = os.path.join(data_dir, filename)
        vid = (
            torch.from_numpy(np.load(input_file) / 255).float().to(device).unsqueeze(0)
        )
        feature = m(vid, True)
        feature = feature.cpu().detach().numpy()[0]
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
    extract_face_features()
