import torch
import torch.nn as nn
from torch.autograd import Variable


class LandmarkLSTM(nn.Module):
    def __init__(
        self, input_size=136, hidden_size=256, num_layers=1, num_classes=2, device=None
    ):
        super(LandmarkLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(
            input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1
        )
        # self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.fc3 = nn.Linear(int(hidden_size / 4), num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, feature=False):
        h0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()
        ).to(self.device)
        c0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size).float()
        ).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        if feature:
            return self.fc2(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    from utils import LandmarkDataset
    from torch.utils.data import DataLoader

    dataset = LandmarkDataset(False)
    dataloader = DataLoader(dataset)

    audio, lab = next(iter(dataloader))
    input_shape = audio.shape[-1]
    print(input_shape)

    m = LandmarkLSTM(input_shape, 2)
    audio_fin = m.forward(audio)
