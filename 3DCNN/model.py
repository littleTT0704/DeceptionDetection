import torch.nn as nn
import torch


class CNN3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video = nn.Sequential(
            nn.Conv3d(3, 16, (1, 7, 7)),
            nn.ReLU(),
            nn.MaxPool3d((2, 4, 4)),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 8, (3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d((2, 3, 3)),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 4, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.BatchNorm3d(4),
            nn.Flatten(),
            nn.Linear(5148, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
        # self.feature = nn.Sequential(nn.Linear(39, 8), nn.ReLU())
        self.end = nn.Sequential(nn.ReLU(), nn.Linear(256, 2), nn.Softmax(1))

    def forward(self, video, feature=False):
        fea = self.video(video)
        if feature:
            return fea
        return self.end(fea)

    # def forward(self, video, feature):
    #     video = self.video(video)
    #     feature = self.feature(feature)
    #     concat = torch.concat([video, feature], dim=1)
    #     return self.end(concat)


if __name__ == "__main__":
    from utils import VideoDataset
    from torch.utils.data import DataLoader

    dataset = VideoDataset(False)
    dataloader = DataLoader(dataset)

    vid, lab = next(iter(dataloader))

    m = CNN3D()
    # m.load_state_dict(torch.load("./model.pt"))
    m.forward(vid)
