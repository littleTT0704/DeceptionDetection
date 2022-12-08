from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys
import os

sys.path.append("..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output(m, dataset, filename):
    if os.path.exists(f"{filename}.npy"):
        return np.load(f"{filename}.npy")
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    output = np.zeros((len(dataset), 2), dtype=np.double)
    m.eval()
    for i, (fea, lab) in enumerate(dataloader):
        pred = m(fea)
        output[i] = pred.cpu().detach().numpy()[0]
    np.save(f"{filename}.npy", output)
    return output


from landmark.utils import LandmarkDataset
from landmark.model import LandmarkLSTM

m = LandmarkLSTM(136, num_layers=1, device=device).to(device)
m.load_state_dict(torch.load("../landmark/model.pt"))
landmarktrain = LandmarkDataset(True, device)
landmarktrain = output(m, landmarktrain, "landmarktrain")
landmarktest = LandmarkDataset(False, device)
landmarktest = output(m, landmarktest, "landmarktest")

from CNN3D.utils import VideoDataset
from CNN3D.model import CNN3D

m = CNN3D().to(device)
m.load_state_dict(torch.load("../CNN3D/model.pt"))
CNN3Dtrain = VideoDataset(True, device)
CNN3Dtrain = output(m, CNN3Dtrain, "CNN3Dtrain")
CNN3Dtest = VideoDataset(False, device)
CNN3Dtest = output(m, CNN3Dtest, "CNN3Dtest")


from audio.utils import AudioDataset
from audio.model import FC_LSTM

m = model = FC_LSTM(196, num_layers=1, device=device).to(device)
m.load_state_dict(torch.load("../audio/model.pt"))
audiotrain = AudioDataset(True, device)
audiotrain = output(m, audiotrain, "audiotrain")
audiotest = AudioDataset(False, device)
audiotest = output(m, audiotest, "audiotest")


from faceCNN.utils import FaceDataset
from faceCNN.model import FaceCNN

m = FaceCNN().to(device)
m.load_state_dict(torch.load("../faceCNN/model.pt"))
faceCNNtrain = FaceDataset(True, device)
faceCNNtrain = output(m, faceCNNtrain, "faceCNNtrain")
faceCNNtest = FaceDataset(False, device)
faceCNNtest = output(m, faceCNNtest, "faceCNNtest")

train = landmarktrain + CNN3Dtrain + audiotrain + faceCNNtrain
train_dataset = AudioDataset(True, device)
train_dataloader = DataLoader(train_dataset, 1, False, num_workers=0)
train_accu = 0.0
for i, (fea, lab) in enumerate(train_dataloader):
    if torch.argmax(lab, 1).item() == np.argmax(train[i]):
        train_accu += 1
train_accu /= len(train_dataset)
print(f"train accuracy: {train_accu:.3f}")

test = landmarktest + CNN3Dtest + audiotest + faceCNNtest
test_dataset = AudioDataset(False, device)
test_dataloader = DataLoader(test_dataset, 1, False, num_workers=0)
test_accu = 0.0
for i, (fea, lab) in enumerate(test_dataloader):
    if torch.argmax(lab, 1).item() == np.argmax(test[i]):
        test_accu += 1
test_accu /= len(test_dataset)
print(f"test  accuracy: {test_accu:.3f}")
