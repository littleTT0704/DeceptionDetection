from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys
import time
import torch.nn as nn

sys.path.append("..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_feature(t):
    lie = np.load(f"../data/features/{t}_lie.npy")
    truth = np.load(f"../data/features/{t}_truth.npy")
    return np.concatenate([lie, truth], axis=0)


def get_features():
    types = ["video", "audio", "face", "landmark", "behavior"]
    features = np.concatenate([get_feature(t) for t in types], axis=1)
    labels = np.zeros((features.shape[0],), dtype=int)
    labels[:61] = 1
    return features, labels


class FeatureDataset(Dataset):
    def __init__(self, train: bool, device=None):
        self.device = device
        test_idx = np.in1d(range(121), list(range(8)) + list(range(61, 69)))
        features, labels = get_features()
        if train:
            self.data = features[~test_idx], labels[~test_idx]
        else:
            self.data = features[test_idx], labels[test_idx]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        feature = self.data[0][index]
        category = self.data[1][index]
        label = np.zeros((2,))
        label[category] = 1

        return (
            torch.from_numpy(feature).float().to(self.device),
            torch.from_numpy(label).float().to(self.device),
        )


class Decision(nn.Module):
    def __init__(self) -> None:
        super(Decision, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(487, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(1),
        )

    def forward(self, feature):
        return self.layers(feature)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FeatureDataset(True, device)
    test_dataset = FeatureDataset(False, device)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size, True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=0)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print(f"train dataset size {train_size}")
    print(f"test  dataset size {test_size}")

    m = Decision().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(m.parameters())

    print("model loaded")

    n_epochs = 100
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    train_acces = np.zeros(n_epochs)
    test_acces = np.zeros(n_epochs)
    for i in range(n_epochs):
        t = time.time()
        print(f"Epoch {i}")
        train_loss = 0.0
        train_accu = 0.0
        m.train()
        for vid, lab in train_dataloader:
            pred = m(vid)
            loss = criterion(pred, lab)

            train_loss += loss.item()
            train_accu += torch.sum(torch.argmax(lab, 1) == torch.argmax(pred, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= train_size
        train_accu /= train_size
        train_losses[i] = train_loss
        train_acces[i] = train_accu
        print(f"train loss: {train_loss:.3f}    train accuracy: {train_accu:.3f}")

        test_loss = 0.0
        test_accu = 0.0
        m.eval()
        for vid, lab in test_dataloader:
            pred = m(vid)
            loss = criterion(pred, lab)

            test_loss += loss.item()
            test_accu += torch.sum(torch.argmax(lab, 1) == torch.argmax(pred, 1))
        test_loss /= test_size
        test_accu /= test_size
        test_losses[i] = test_loss
        test_acces[i] = test_accu

        print(f"test  loss: {test_loss:.3f}    test  accuracy: {test_accu:.3f}")
        if test_accu > 0.6:
            torch.save(m.state_dict(), "./good.pt")
        torch.save(m.state_dict(), "./model.pt")
        print(f"time used {time.time() - t:.2f}s\n")

    np.savez("res.npz", train_losses, train_acces, test_losses, test_acces)
