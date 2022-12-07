from model import FC_LSTM
import torch
import numpy as np
import time
import os

root_dir = "../data"
N_VID = 60


def extract_audio_features():
    device = torch.device("cpu")
    m = FC_LSTM(196).to(device)
    m.load_state_dict(torch.load("./model.pt"))
    m.eval()

    data_dir = os.path.join(root_dir, "TF_DC")
    output_dir = os.path.join(root_dir, "Audio")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    d = 64
    truth = np.zeros((N_VID, d))
    lie = np.zeros((N_VID + 1, d))

    for filename in os.listdir(data_dir):
        category, end = filename.split("_")
        idx = int(end.split(".")[0]) - 1

        input_file = os.path.join(data_dir, filename)
        aud = torch.from_numpy(np.load(input_file)).float().to(device).unsqueeze(0)
        feature = m(aud, True)
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
    extract_audio_features()
