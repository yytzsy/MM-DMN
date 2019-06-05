"""Preprocess the data for model."""
import os
import inspect
import csv

import numpy as np
from PIL import Image
import skvideo.io
import scipy
import pandas as pd
import json


def transferid():
    f = open('/mnt/data/yuanyitian/videoQA/youtube_mapping.txt','r')
    ls = f.readlines()
    id_dict = {}
    for line in ls:
        [origin,transfer] = line.split()
        id_dict[transfer] = origin
    return id_dict


def check_extract_frames(name,video_directory,num):

    extract_frames_dict = {}
    for i in range(0, num):
        print i
        if name == 'msvd_extract_frames':
            id_dict = transferid()
            video_name = id_dict['vid' + str(i+1)]
            video_path = os.path.join(video_directory, video_name+'.avi')
        elif name == 'msrvtt_extract_frames':
            video_path = os.path.join(video_directory, 'video' + str(i) + '.mp4')
        try:
            video_data = skvideo.io.vread(video_path)
            total_frames = video_data.shape[0]
            pre_video_data = video_data
        except:
            video_data = pre_video_data
            total_frames = 0
        frame_list = []
        for j in np.linspace(0, total_frames, 20 + 2)[1:20 + 1]:
            frame_list.append(int(j))
        if name == 'msvd_extract_frames':
            extract_frames_dict[i+1] = frame_list
        elif name == 'msrvtt_extract_frames':
            extract_frames_dict[i] = frame_list

    with open("./"+name+'.json',"w") as f:
        json.dump(extract_frames_dict,f)

check_extract_frames('msvd_extract_frames', '/mnt/data/yuanyitian/MSVD/Youtube2Text/YouTubeClips', 1970)
check_extract_frames('msrvtt_extract_frames', '/mnt/data/yuanyitian/videoQA/MSRVTT-QA/video/train-video', 10000)




