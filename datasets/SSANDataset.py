import os
import math
import random
from glob import glob

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import imgaug.augmenters as iaa

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])


class SSANDataset(Dataset):

    def __init__(self, data_dir, csv_file, transform=None, UUID=-1):
        super().__init__()
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, csv_file), delimiter=',', header=None)
        
        self.transform = transform
        self.UUID = UUID

    def __getitem__(self, index):
        file_name = self.data.iloc[index, 0]
        frame_path = os.path.join(self.data_dir, file_name)

        frame = self.get_frame_depth_pair(frame_path)
        label = self.data.iloc[index, 1]
        label = 1 if label == 1 else 0
        # label = np.expand_dims(label, axis=0)

        # if label == 1:
        #     mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * self.label_weight
        # else:
        #     mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1.0 - self.label_weight)

        if self.transform:
            # frame = self.transform(image=frame)['image']
            frame = self.transform(frame)

        return frame, label, self.UUID

    def get_frame_depth_pair(self, frame_path):
        frames_total = glob(os.path.join(frame_path, "*.jpg"))
        # depth_list = os.listdir(frame_path)
        frame_sample_name = random.choice(frames_total)
        
        file_name = frame_sample_name.split("_frame")[0]
        dat_sample_name = file_name + "_frame.dat"

        # frame_sample_path = os.path.join(frame_path, frame_sample_name)
        # dat_sample_path = os.path.join(frame_path, dat_sample_name)  
        
        face_scale = np.random.randint(10, 15)
        face_scale = face_scale / 10.0

        # BGR
        # frame_sample_bgr = cv2.imread(frame_sample_name)
        frame_sample_rgb = np.array(Image.open(frame_sample_name))
         
        face_frame = self.get_face_from_frame(frame_sample_rgb, dat_sample_name, face_scale)
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        face_frame_aug = seq.augment_image(face_frame)         
   
        return Image.fromarray(face_frame_aug)

    def get_face_from_frame(self, frame_bgr, dat_sample_path, face_scale):
        with open(dat_sample_path, "r") as f:
            line=f.readlines()[0]
            coords = line.split(",")
            y1,x1,w,h=[float(coord) for coord in coords]
            f.close()  

        y2=y1+h
        x2=x1+w

        y_center=(y1+y2)/2.0
        x_center=(x1+x2)/2.0

        h_frame, w_frame = frame_bgr.shape[0], frame_bgr.shape[1]

        w_scale=face_scale*w
        h_scale=face_scale*h
        y1=y_center-h_scale/2.0
        x1=x_center-w_scale/2.0
        y2=y_center+h_scale/2.0
        x2=x_center+w_scale/2.0

        # Border cases.
        y1=max(math.floor(y1),0)
        x1=max(math.floor(x1),0)
        y2=min(math.floor(y2),h_frame)
        x2=min(math.floor(x2),w_frame)

        face_roi_bgr=frame_bgr[y1:y2, x1:x2]
        
        # face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)

        return cv2.resize(face_roi_bgr, (256, 256))

    def __len__(self):
        return len(self.data)

class SSANDataset_test(Dataset):
    
    def __init__(self, data_dir, csv_file, transform=None, UUID=-1):
        super().__init__()
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, csv_file), delimiter=',', header=None)
        
        self.transform = transform
        self.UUID = UUID

    def __getitem__(self, index):
        file_name = self.data.iloc[index, 0]
        frame_path = os.path.join(self.data_dir, file_name)

        frames = self.get_frame_depth_pair(frame_path)
        label = self.data.iloc[index, 0]
        label = 1 if label == 1 else 0
        
        sample = {'image_x': frames, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_frame_depth_pair(self, frame_path):
        frames_total = glob(os.path.join(frame_path, "*.jpg"))
        face_frames = np.zeros((len(frames_total), 256, 256, 3))

        for i, frame_sample_name in enumerate(frames_total):
            file_name = frame_sample_name.split("_frame")[0]
            dat_sample_name = file_name + "_frame.dat"
            
            # BGR
            frame_sample_bgr = cv2.imread(frame_sample_name)
            
            face_frames[i, :, :, :] = self.get_face_from_frame(frame_sample_bgr, dat_sample_name, face_scale=1.3)

        return face_frames

    def get_face_from_frame(self, frame_bgr, dat_sample_path, face_scale):
        with open(dat_sample_path, "r") as f:
            line=f.readlines()[0]
            coords = line.split(",")
            y1,x1,w,h=[float(coord) for coord in coords]
            f.close()  

        y2=y1+h
        x2=x1+w

        y_center=(y1+y2)/2.0
        x_center=(x1+x2)/2.0

        h_frame, w_frame = frame_bgr.shape[0], frame_bgr.shape[1]

        w_scale=face_scale*w
        h_scale=face_scale*h
        y1=y_center-h_scale/2.0
        x1=x_center-w_scale/2.0
        y2=y_center+h_scale/2.0
        x2=x_center+w_scale/2.0

        # Border cases.
        y1=max(math.floor(y1),0)
        x1=max(math.floor(x1),0)
        y2=min(math.floor(y2),h_frame)
        x2=min(math.floor(x2),w_frame)

        face_roi_bgr=frame_bgr[y1:y2, x1:x2]
        
        # face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)

        return cv2.resize(face_roi_bgr, (256, 256))

    def __len__(self):
        return len(self.data)