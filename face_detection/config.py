# encoding: utf-8
"""
@author: Ming Cheng
@contact: ming.cheng@dukekunshan.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")


class Config:
    """
    自定义Config类，用于配置当前模块的所有变量参数
    """

    ############# 基础配置 #############
    # 当前模块路径
    name = "02_face_detection"
    # root = os.path.join('codebase', name)
    root = "./RetinaFace"
    # 待处理范式列表

    # 该模型需要处理的范式列表
    parad_list = [  #'free',
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "c1",
        "c2",
        "c3",
        "c4",
        "d1",
        "d2",
        "d3",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "f1",
        "f2",
        "f3",
        "f4",
    ]
    # parad_list = ['a1']

    # 相机作用域
    valid_cams = [
        "d1",
        "d3",
        "d5",
        "d7",
        "h0",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "h7",
    ]
    res_resize = 4  # 对高清相机有一个缩小的倍数
    fps_preset = 8
    ############# 模型参数 #############
    # 单帧内保留的最大人脸数量
    max_face_num1 = 50
    max_face_num2 = 5
    # 最小的人脸尺寸
    smallest_face = 15
    #

    # 检测的confidence阈值
    scr_thresh = 0.95
    nms_thresh = 0.4
    # 可用设备列表，及单个GPU上的workers数量
    devices = ["cuda:0", "cuda:1"]

    task_num_per_device = 1
    # 单个模型的Batch Size
    batch_size = 32
    queue_size = 3

    ############# 显示设置 #############
    # 是否显示进度
    show_progress = True
    show_runtimes = False
    # 进度提示标语
    prepare_slogan = "正在加载人脸检测模型"
    running_slogan = "正在检测面部活动区域"
    logging_slogan = "人脸检测"
    # 是否输出渲染结果
    render = True
    demo_w = 720
    demo_h = 480

    ############# 文件配置 #############
    # 计算结果保存的相对路径（/parad_dir）
    mask_dir = os.path.join("results", "01_segment")
    save_dir = os.path.join("results", "02_facedet")

    # 本地日志保存路径
    local_logger = os.path.join(os.path.split(root)[0], "logging.log")
    # 是否输出logging到终端
    print_logger = True


cfg = Config()
