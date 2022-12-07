from utils import VideoDataset
from torch.utils.data import DataLoader
from model import CNN3D
import torch
import numpy as np
import time

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = VideoDataset(True, device)
    test_dataset = VideoDataset(False, device)

    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size, True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size, num_workers=0)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print(f"train dataset size {train_size}")
    print(f"test  dataset size {test_size}")

    m = CNN3D().to(device)
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
        torch.save(m.state_dict(), "./model.pt")
        print(f"time used {time.time() - t:.2f}s\n")

    np.savez("res.npz", train_losses, train_acces, test_losses, test_acces)
