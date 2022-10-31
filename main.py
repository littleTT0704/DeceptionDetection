from utils import load_videos, load_features
from model import Model
import torch
import numpy as np
import time

if __name__ == "__main__":
    truth_video, lie_video = load_videos()
    truth_feature, lie_feature = load_features()

    truth_video_train = truth_video[:-8]
    truth_video_test = truth_video[-8:]
    lie_video_train = lie_video[:-8]
    lie_video_test = lie_video[-8:]
    video_train = np.vstack([truth_video_train, lie_video_train])
    video_test = np.vstack([truth_video_test, lie_video_test])
    video_train = torch.from_numpy(video_train).float()
    video_test = torch.from_numpy(video_test).float()

    truth_feature_train = truth_feature[:-8]
    truth_feature_test = truth_feature[-8:]
    lie_feature_train = lie_feature[:-8]
    lie_feature_test = lie_feature[-8:]
    feature_train = np.vstack([truth_feature_train, lie_feature_train])
    feature_test = np.vstack([truth_feature_test, lie_feature_test])
    feature_train = torch.from_numpy(feature_train).float()
    feature_test = torch.from_numpy(feature_test).float()

    train_label = np.zeros((feature_train.shape[0], 2))
    train_label[: truth_feature_train.shape[0], 0] = 1
    train_label[truth_feature_train.shape[0] :, 1] = 1
    train_label = torch.from_numpy(train_label).float()

    test_label = np.zeros((feature_test.shape[0], 2))
    test_label[: truth_feature_test.shape[0], 0] = 1
    test_label[truth_feature_test.shape[0] :, 1] = 1
    test_label = torch.from_numpy(test_label).float()

    train_size = train_label.size()[0]
    test_size = test_label.size()[0]

    print("data loaded")

    m = Model()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(m.parameters())

    print("model loaded")

    n_epochs = 100
    batch_size = 16
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    train_acces = np.zeros(n_epochs)
    test_acces = np.zeros(n_epochs)
    for i in range(n_epochs):
        t = time.time()
        print(f"Epoch {i}")
        idx = torch.randperm(train_size)
        video_train = video_train[idx]
        feature_train = feature_train[idx]
        train_label = train_label[idx]

        train_loss = 0.0
        train_correct = 0.0
        for b in range(0, train_size, batch_size):
            video_batch = video_train[b : b + batch_size]
            feature_batch = feature_train[b : b + batch_size]
            label_batch = train_label[b : b + batch_size]

            pred = m((video_batch, feature_batch))
            loss = criterion(pred, label_batch)

            train_loss += loss.item()
            train_correct += torch.sum(
                torch.argmax(label_batch, 1) == torch.argmax(pred, 1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        video_batch = video_train[b:]
        feature_batch = feature_train[b:]
        label_batch = train_label[b:]

        pred = m((video_batch, feature_batch))
        loss = criterion(pred, label_batch)
        train_loss += loss.item()
        train_correct += torch.sum(
            torch.argmax(label_batch, 1) == torch.argmax(pred, 1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss /= train_size
        train_correct /= train_size
        train_losses[i] = train_loss
        train_acces[i] = train_correct
        print(f"train loss: {train_loss:.3f}    train accuracy: {train_correct:.3f}")

        test_pred = m((video_test, feature_test))
        test_loss = criterion(test_pred, test_label) / test_size
        test_correct = (
            torch.sum(torch.argmax(test_label, 1) == torch.argmax(test_pred, 1))
            / test_size
        )
        test_losses[i] = test_loss
        test_acces[i] = test_correct
        print(f"test  loss: {test_loss:.3f}    test  accuracy: {test_correct:.3f}")
        torch.save(m.state_dict(), "./model.pt")
        print(f"time used {time.time() - t:.2f}s\n")

    np.savez("res.npz", train_losses, train_acces, test_losses, test_acces)
