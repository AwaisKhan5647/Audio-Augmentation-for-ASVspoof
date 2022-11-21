
from __future__ import print_function
import pandas as pd
import numpy as np
import os, shlex,pandas as pd, subprocess
import json 
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
from os import path
from pydub import AudioSegment
from audiomentations import HighPassFilter 
import acoustics
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm


#  High Pass Filtering 


data_type='PA'
augment_type='HighPassFilter'
train_dataset_path = r'/scratch/projects/smiles/Multi_Features/Datasets/ASVSpoof2019/'+data_type+'/ASVspoof2019_'+data_type+'_train/flac/'
audio_files=glob(train_dataset_path + "/*.flac")

path = r"/scratch/projects/smiles/Multi_Features/Datasets/Augmented_Asvspoof2019/"+data_type+"/Bonafide/"
os.chdir(path)

if not os.path.exists(augment_type):
    os.mkdir(augment_type)
    print("Directory " , augment_type ,  " Created ")
else:    
    print("Directory " , augment_type ,  " already exists")
    
path=os.path.join(path,augment_type)

train_labels= []
train_audio=[] 

print("High pass filtering")
with open('/scratch/projects/smiles/Multi_Features/Datasets/ASVSpoof2019/'+data_type+'/ASVspoof2019_'+data_type+'_cm_protocols/ASVspoof2019.'+data_type+'.cm.train.trn.txt', 'r') as f:
    for i,line in enumerate(tqdm(f)): #train_raw_labels
        t_label = line.split(' ')
        t_name = t_label[1]
        t_lab = t_label[4].strip()
        test_audio, sample_rate_test = lr.load(train_dataset_path+t_name+'.flac',sr=16000)
        train_audio.append(test_audio)
        if str(t_lab)=='spoof':
            train_labels.append(1)
        elif str(t_lab)=='bonafide':
            train_labels.append(0) 
            augment = Compose([HighPassFilter()])
            augmented_samples = augment(samples=test_audio, sample_rate=sample_rate_test)
            print(t_name+'.flac')
            sf.write(os.path.join(path,t_name+'.flac'), augmented_samples, sample_rate_test)
print("Done")
