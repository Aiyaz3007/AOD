import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2
from tqdm import tqdm
from os.path import join


dehaze_net = net.dehaze_net().cuda()
dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth',map_location="cpu"))


def dehaze_image(image_data):
    # print(image_path)
    # data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(image_data)/255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2,0,1)
    # data_hazy = data_hazy.cpu().unsqueeze(0)
    data_hazy = data_hazy.cuda().unsqueeze(0)


    clean_image = dehaze_net(data_hazy)
    # clean_image = clean_image  

    return clean_image




video_path = "/content/fog_video.mp4"

camera = cv2.VideoCapture(video_path)
output_path = "output.mp4"
fps = int(camera.get(5))

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

prev_time = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
bar = tqdm(total = length)

while camera.isOpened():
    success, frame = camera.read()
    if success:
        current_time = time.time()

        res = dehaze_image(frame)
        res = res.squeeze().permute(1,2,0).cpu().detach().numpy()*255
        if res.dtype != 'uint8':
          res = cv2.convertScaleAbs(res)
        fps_text = f"FPS: {1 / (current_time - prev_time):.2f}"
        prev_time = current_time
        cv2.putText(res, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(res)
        bar.update(1)
    else:
        break
camera.release()
out.release()
    
