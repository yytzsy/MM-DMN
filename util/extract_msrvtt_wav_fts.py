from python_speech_features import calcMFCC_delta_delta
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
import cPickle as pkl


def extract_wav_feature():

    video_wav_dict = {}
    video_wav_dir = '/mnt/data/yuanyitian/videoQA/MSRVTT-QA/video/train-video-wav/'
    for i in range(10000):
        print i
        wav_name = video_wav_dir+'video'+str(i)+'.wav'
        if os.path.exists(wav_name):
            (rate,sig) = wav.read(wav_name)
            mfcc_feat = calcMFCC_delta_delta(sig,rate)
            video_wav_dict[i] = mfcc_feat
            print np.shape(mfcc_feat)
        else:
            video_wav_dict[i] = np.zeros([20,39],dtype=np.float32)
            print 'None'
        print '********************************'

    f = open('/mnt/data/yuanyitian/videoQA/dmn/data/msrvtt_qa/data/video_audio_feature_1024.pkl','wb')
    pkl.dump(video_wav_dict,f)


extract_wav_feature()




