import imageio
import numpy as np
import os
import time
from typing import Optional
from engine import load_model, infer_image_data
import tqdm

N_VID = 60
DW = 128
DH = 128
root_dir = "../data"

m = load_model()


def extract_face(
    idx: int,
    deceptive: bool = True,
    length: int = 90,
    width: Optional[int] = DW,
    height: Optional[int] = DH,
):
    n = N_VID + deceptive
    if idx < 1 or idx > n:
        print(f"index {idx} out of bound [1, {n}]")
        return

    filename = os.path.join(
        root_dir,
        f"Clips/{['Truthful', 'Deceptive'][deceptive]}/trial_{['truth', 'lie'][deceptive]}_{idx:03d}.mp4",
    )
    print(filename)
    vid = imageio.get_reader(filename, "ffmpeg")
    metadata = vid.get_meta_data()
    n_frames = int(metadata["fps"] * metadata["duration"])
    if not deceptive and idx == 25:
        n_frames = int(metadata["fps"] * 25)
    if deceptive and idx == 42:
        n_frames = int(metadata["fps"] * 22)
    if deceptive and idx == 51:
        n_frames = int(metadata["fps"] * 5)
    W, H = metadata["size"]
    print(f"frame size: {W} * {H}, number of frames: {n_frames}")
    if width is None or height is None:
        width, height = W, H

    output_dir = os.path.join(root_dir, f"face/")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filename = os.path.join(
        output_dir, f"{['truth', 'lie'][deceptive]}_{idx:02d}.npy"
    )

    res = np.zeros((length, height, width, 3), dtype=np.uint8)
    for i, frame in enumerate(vid):
        face = infer_image_data(m, frame)
        if face is not None:
            print(f"starting frame {i}")
            res[0] = face
            break

    total_length = n_frames - i
    idxs = np.arange(i, n_frames, total_length // length, dtype=int)[1:length]
    for i, idx in tqdm.tqdm(enumerate(idxs), total=length):
        j = idx
        face = None
        while face is None:
            face = infer_image_data(m, vid.get_data(j))
            j += 1
        res[i + 1] = face
    res = res.transpose((3, 0, 1, 2))
    np.save(output_filename, res)
    vid.close()


def extract_all_faces(
    length: int = 90,
    width: int = DW,
    height: int = DH,
):
    for i in range(N_VID):
        t = time.time()
        extract_face(i + 1, False, length, width, height)
        print(f"time spent {time.time() - t:.2f}s")
    for i in range(N_VID + 1):
        t = time.time()
        extract_face(i + 1, True, length, width, height)
        print(f"time spent {time.time() - t:.2f}s")


if __name__ == "__main__":
    t = time.time()
    extract_all_faces()
    print(f"time spent {time.time() - t:.2f}s")
