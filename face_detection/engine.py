# encoding: utf-8
"""
@author: Ming Cheng
@contact: ming.cheng@dukekunshan.edu.cn
"""

from config import cfg
import os, sys

import cv2
import torch
import numpy as np
from queue import Queue
from threading import Thread
from copy import deepcopy
from time import perf_counter
import numpy as np
import math
from RetinaFace.models.retinaface import RetinaFace, model_cfg
from RetinaFace.models.prior_box import PriorBox
from RetinaFace.utils.box_utils import decode, decode_landm
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
from RetinaFace.face_align import warp_and_crop_face

# from common.utils import init_output_dir, init_mdata_writer
from torchvision import transforms

import warnings

warnings.filterwarnings("ignore")


############ 模型推理部分


def load_model():
    """
    从封装好的模型文件中加载模型结构及其权重，该部分的具体内容视不同工程而定
    Attributes
        device: 将模型加载到对应的设备名称，例如'cuda:0', 'cuda:1'等
    Returns
        model: 已加载到对应设备上的模型
    """
    # weight_path = os.path.join(cfg.root, "weights/retinaface_res50.pth")
    weight_path = os.path.join(cfg.root, "weights/Resnet50_Final.pth")
    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model = RetinaFace(cfg=model_cfg, phase="test")
    model.load_state_dict(state_dict)
    model = model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return model


def load_many_models(workers):
    """
    批处理加载
    """
    single_model = load_model()

    models = []
    for device in workers:
        m = deepcopy(single_model)
        m = m.to(device)
        models.append(m)

    return models


def prepare_batch_data(model, imgs):
    """
    输入加载过的model, imgs = [array]
    返回已经预处理过但是还没有加载到GPU上的一个Batch
    """

    batch_data = []
    for img in imgs:
        img = img - (104.0, 117.0, 123.0)
        img = transforms.ToTensor()(img)
        batch_data.append(img)

    device = next(model.parameters()).device
    batch_data = torch.stack(batch_data, dim=0).float().to(device)

    return batch_data


def compute_batch_data(model, batch_data, resize):

    im_height, im_width = batch_data.size()[-2:]

    device = next(model.parameters()).device
    # 用于还原bounding box
    scale1 = torch.FloatTensor([im_width, im_height, im_width, im_height])
    scale1 = scale1.to(device)
    # 用于还原landmarks
    scale2 = torch.Tensor(
        [
            im_width,
            im_height,
            im_width,
            im_height,
            im_width,
            im_height,
            im_width,
            im_height,
            im_width,
            im_height,
        ]
    )
    scale2 = scale2.to(device)
    # 初始化先验框
    priorbox = PriorBox(model_cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)

    batch_pred = []
    with torch.no_grad():
        batch_locs, batch_confs, batch_landms = model(batch_data)
        # 逐帧处理模型的输出结果
        for i in range(len(batch_locs)):
            # 还原bounding box
            frame_bboxes = decode(
                batch_locs[i].detach(), priors.detach(), model_cfg["variance"]
            )
            frame_bboxes = (frame_bboxes * scale1 * resize).cpu().numpy()
            # 还原confidence scores
            frame_scores = batch_confs[i].detach()[:, 1]
            frame_scores = frame_scores.cpu().numpy()
            # 还原facial landmarks
            frame_coords = decode_landm(
                batch_landms[i].detach(), priors.detach(), model_cfg["variance"]
            )
            frame_coords = (frame_coords * scale2 * resize).cpu().numpy()
            batch_pred.append((frame_bboxes, frame_scores, frame_coords))
    torch.cuda.empty_cache()

    return batch_pred


def extract_thread(model, pred_queue, mdata_writer, video_id, origin_res):
    while True:
        qdata = pred_queue.get()
        if qdata == "stop":
            break
        t1 = perf_counter()
        batch_pred = qdata[0]
        batch_info = qdata[1]
        for i in range(len(batch_pred)):
            frame_meta = extract_frame_meta(
                batch_pred[i], video_id, batch_info[i], origin_res
            )
            mdata_writer.write(frame_meta)
        t2 = perf_counter()
        if cfg.show_runtimes:
            print("\nExtract:", round(t2 - t1, 6))


def compute_thread(model, data_queue, pred_queue, resize):
    while True:
        qdata = data_queue.get()
        if qdata == "stop":
            break
        t1 = perf_counter()
        batch_data = qdata[0]
        batch_info = qdata[1]
        batch_pred = compute_batch_data(model, batch_data, resize)
        pred_queue.put((batch_pred, batch_info))
        t2 = perf_counter()
        if cfg.show_runtimes:
            print("\nCompute:", round(t2 - t1, 6))
    pred_queue.put("stop")

    return


def infer_video_data(model, video_path, save_meta_path, video_id):

    if not os.path.exists(video_path):
        return

    # 初始化队列，用于存储CPU预处理的整个Batch数据
    data_queue = Queue(cfg.queue_size)
    pred_queue = Queue(cfg.queue_size)

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    num = int(cap.get(7))
    fps = int(cap.get(5))

    # 当原始帧过大时，对原始帧进行resize，缩小4备
    if video_id[0] == "h":
        resize = int(cfg.res_resize)
    else:
        resize = 1

    mdata_writer = init_mdata_writer(save_meta_path)

    thread1 = Thread(
        target=compute_thread, args=(model, data_queue, pred_queue, resize), daemon=True
    )
    thread2 = Thread(
        target=extract_thread,
        args=(model, pred_queue, mdata_writer, video_id, (w, h)),
        daemon=True,
    )
    thread1.start()
    thread2.start()

    # 利用列表做buffer，用于积累够一个batchsize
    buff_1 = []
    buff_2 = []
    t1 = perf_counter()
    for idx in range(num):
        ret, img = cap.read()
        if not ret:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        img = cv2.resize(img, (math.ceil(w / resize), math.ceil(h / resize)))
        # 将当前帧添加到buff
        buff_1.append(img)
        buff_2.append(idx)
        # 满足条件时即可构建一个batch的预处理后数据，并清空当前buffer
        if len(buff_1) >= cfg.batch_size:
            batch_data = prepare_batch_data(model, imgs=buff_1)
            batch_info = deepcopy(buff_2)
            data_queue.put((batch_data, batch_info))
            buff_1 = []
            buff_2 = []
            t2 = perf_counter()
            if cfg.show_runtimes:
                print("\nPrepare:", round(t2 - t1, 6))
            t1 = perf_counter()

    if len(buff_1) > 0:
        batch_data = prepare_batch_data(model, imgs=buff_1)
        batch_info = deepcopy(buff_2)
        data_queue.put((batch_data, batch_info))
        buff_1 = []
        buff_2 = []
    # 放入结束标识符，表示当前视频处理完毕
    data_queue.put("stop")
    thread1.join()
    thread2.join()

    # 释放视频文件资源
    cap.release()
    mdata_writer.release()

    return


def infer_many_video_data(model, file_manager, video_ids, counter):
    for video_id in video_ids:
        video_path = file_manager.videos[video_id]
        save_meta_path = os.path.join(
            file_manager.parad_dir, cfg.save_dir, "meta_{}.json".format(video_id)
        )
        infer_video_data(model, video_path, save_meta_path, video_id)
        counter.update(1)

    return


def infer_image_data(model, img):
    h, w = img.shape[:2]
    batch_data = prepare_batch_data(model, [img])
    batch_pred = compute_batch_data(model, batch_data, 1)
    frame_meta = extract_frame_meta(batch_pred[0], 0, 0, (w, h))
    if len(frame_meta) > 0:
        inst_landmark = frame_meta[0]["face_landm"]
        facial5points = [
            [inst_landmark[i], inst_landmark[i + 1]] for i in [0, 2, 4, 6, 8]
        ]
        cropped_image = warp_and_crop_face(
            src_img=img, facial_pts=facial5points, crop_size=(128, 128)
        )
    else:
        cropped_image = None

    return cropped_image


def infer_fbank_data(model, file_manager):
    for person_id in file_manager.fbanks.keys():
        for idx, img_path in enumerate(file_manager.fbanks[person_id]):
            if img_path is None:
                continue
            if not os.path.exists(img_path):
                continue
            img_crop = infer_image_data(model, img_path)
            if img_crop is not None:
                new_path = os.path.join(
                    file_manager.parad_dir,
                    cfg.save_dir,
                    "fbanks",
                    person_id,
                    str(idx) + ".png",
                )
                init_output_dir(new_path)
                cv2.imwrite(new_path, img_crop)


############ 特征提取部分


def correct_coord_outliers(coords, origin_res):
    new_coords = deepcopy(coords)
    for idx, coord in enumerate(coords):
        if idx % 2 == 0:
            new_x = min(max(0, coords[idx]), origin_res[0] - 1)
            new_coords[idx] = new_x
        if idx % 2 == 1:
            new_y = min(max(0, coords[idx]), origin_res[1] - 1)
            new_coords[idx] = new_y

    return new_coords


def extract_frame_meta(frame_pred, video_id, frame_id, origin_res):
    # 从frame-level的模型输出中提取当前帧的meta数据
    frame_bboxes, frame_scores, frame_coords = frame_pred
    # ignore low scores
    inds = np.where(frame_scores > cfg.scr_thresh)[0]
    frame_bboxes = frame_bboxes[inds]
    frame_scores = frame_scores[inds]
    frame_coords = frame_coords[inds]
    # keep the top-K before NMS (1st round)
    topk = frame_scores.argsort()[::-1][: cfg.max_face_num1]
    frame_bboxes = frame_bboxes[topk]
    frame_scores = frame_scores[topk]
    frame_coords = frame_coords[topk]
    # do NMS
    dets = np.hstack((frame_bboxes, frame_scores[:, np.newaxis])).astype(
        np.float32, copy=False
    )
    inds = py_cpu_nms(dets, cfg.nms_thresh)
    dets = dets[inds, :]
    frame_coords = frame_coords[inds]
    # keep the top-K after NMS (2nd round)
    frame_coords = frame_coords[: cfg.max_face_num2, :]
    dets = dets[: cfg.max_face_num2, :]
    dets = np.concatenate((dets, frame_coords), axis=1)

    frame_meta = []
    for det in dets:
        bbox = det[:4].astype("int").tolist()
        bbox = correct_coord_outliers(bbox, origin_res)
        landm = det[5:].astype("int").tolist()
        landm = correct_coord_outliers(landm, origin_res)
        conf = round(float(det[4]), 2)
        # 当人脸小于一定程度时舍弃
        if (bbox[2] - bbox[0]) < cfg.smallest_face or (
            bbox[3] - bbox[1]
        ) < cfg.smallest_face:
            continue
        else:
            frame_meta.append(
                dict(
                    id="unknown",
                    video=video_id,
                    frame=frame_id,
                    face_bbox=bbox,
                    face_landm=landm,
                    facedet_conf=conf,
                )
            )

    return frame_meta
