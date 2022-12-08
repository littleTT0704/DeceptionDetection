import numpy as np

features = np.genfromtxt(
    "../data/Annotation/All_Gestures_Deceptive and Truthful.csv", delimiter=","
)[1:, 1:-1]

print(features[61:].shape)
np.save("../data/features/behavior_lie.npy", features[:61])
np.save("../data/features/behavior_truth.npy", features[61:])
