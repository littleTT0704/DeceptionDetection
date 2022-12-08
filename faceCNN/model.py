import torch.nn as nn
import torch


class FaceCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video = nn.Sequential(
            nn.Conv3d(3, 32, (1, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((2, 3, 3)),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 16, (3, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((2, 3, 3)),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 8, (3, 5, 5)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.BatchNorm3d(8),
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.end = nn.Sequential(nn.ReLU(), nn.Linear(64, 2), nn.Softmax(1))

    def forward(self, video, feature=False):
        fea = self.video(video)
        if feature:
            return fea
        return self.end(fea)


if __name__ == "__main__":
    from utils import FaceDataset
    from torch.utils.data import DataLoader

    vid = torch.zeros((1, 3, 90, 128, 128))

    m = FaceCNN()
    m.forward(vid)
