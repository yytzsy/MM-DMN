import os
import inspect
import csv

import numpy as np
from PIL import Image
import skvideo.io
import scipy
import tensorflow as tf
import pandas as pd
import cPickle as pkl



def get_msrvttMFCC_20():

    MFCC_all = pkl.load(open('/mnt/data/yuanyitian/videoQA/dmn/data/msrvtt_qa/data/video_audio_feature_1024.pkl'))
    MFCC_20 = np.zeros([10000,20,39],dtype=np.float32)
    for i in range(10000):
        [total_frames,_] = np.shape(MFCC_all[i])
        print i
        print total_frames
        print '**********************************************'
        if total_frames > 20:
            k = 0
            for j in np.linspace(0, total_frames, 20 + 2)[1:20 + 1]:
                print np.shape(MFCC_20[i][k])
                print np.shape(MFCC_all[i][int(j)])
                MFCC_20[i][k] = MFCC_all[i][int(j)]
                k+=1
    mean_MFCC = np.mean(MFCC_20,axis = 0)
    print np.shape(mean_MFCC)
    for i in range(10000):
        [total_frames,_] = np.shape(MFCC_all[i])
        if total_frames == 20:
            MFCC_20[i] = mean_MFCC

    f = open('/mnt/data/yuanyitian/videoQA/dmn/data/msrvtt_qa/data/video_audio_feature_1024_20.pkl','wb')
    pkl.dump(MFCC_20,f)


get_msrvttMFCC_20()




