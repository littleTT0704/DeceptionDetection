import numpy as np
import matplotlib.pyplot as plt
from model import Model
import torch


def loss_acc(name):
    plt.rcParams.update({"font.size": 15})

    def load(filename):
        with np.load(filename) as data:
            res = [data[f"arr_{i}"] for i in range(4)]
        return res

    train_loss, train_acc, test_loss, test_acc = load(f"./res/{name}.npz")
    n_epochs = test_loss.shape[0]
    epochs = np.arange(n_epochs)

    min_train_loss = np.min(train_loss)
    min_test_loss = np.min(test_loss)

    max_train_acc = np.max(train_acc)
    max_test_acc = np.max(test_acc)

    print(
        f"best train loss: {min_train_loss:.2f}, best train accuracy: {max_train_acc:.2f}"
    )
    print(
        f"best test loss:  {min_test_loss:.2f}, best test accuracy:  {max_test_acc:.2f}"
    )

    c = ["#1f77b4", "#ff7f0e"]
    size = (8, 6)

    plt.figure(figsize=size)
    plt.plot(epochs, train_loss, label="train", color=c[0])
    plt.plot(epochs, test_loss, label="test", color=c[1])

    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(f"loss vs epoch")
    plt.legend()
    plt.savefig(f"./res/{name}_loss.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=size)
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, test_acc, label="test")

    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title(f"accuracy vs epoch")
    plt.legend()
    plt.savefig(f"./res/{name}_acc.png", bbox_inches="tight")


def weight(filename):
    d = torch.load(filename, map_location=torch.device("cpu"))
    w = d["end.0.weight"].numpy()
    plt.figure(figsize=(36, 4))
    plt.matshow(w)
    plt.savefig("./res/weight.png")


if __name__ == "__main__":
    loss_acc("200")
