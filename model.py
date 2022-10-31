import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video = nn.Sequential(
            nn.Conv3d(3, 16, (1, 7, 7)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 8, (5, 5, 5), padding=2),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 4, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.BatchNorm3d(4),
            nn.Flatten(),
            nn.Linear(2376, 512),
        )
        self.feature = nn.Linear(39, 8)
        self.end = nn.Sequential(
            nn.Linear(520, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(1)
        )

    def forward(self, x):
        video, feature = x
        video = self.video(video)
        feature = self.feature(feature)
        concat = torch.concat([video, feature], dim=1)
        return self.end(concat)


if __name__ == "__main__":
    from utils import load_videos, load_features

    truth_video, lie_video = load_videos()
    truth_feature, lie_feature = load_features()

    m = Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    m.forward(
        (
            torch.from_numpy(truth_video[0:16]).float(),
            torch.from_numpy(truth_feature[0:16]).float(),
        )
    )
