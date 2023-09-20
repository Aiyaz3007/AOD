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
dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth',map_location="cuda"))


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


def cv2_dehaze_filter(frame):
    frame = frame.astype(np.float32) / 255.0
    dark_channel = np.min(frame, axis=2)
    atmospheric_light = np.percentile(dark_channel, 95)
    t = 1 - 0.95 * dark_channel / atmospheric_light
    radius = 40 # Adjust the radius as needed
    transmission_map = cv2.GaussianBlur(t, (2 * radius + 1, 2 * radius + 1), 0)
    transmission_map = np.clip(transmission_map, 0, 1)
    scene_radiance = np.zeros_like(frame)
    for i in range(3):
      scene_radiance[:, :, i] = (frame[:, :, i] - atmospheric_light) / transmission_map + atmospheric_light
    scene_radiance = np.clip(scene_radiance, 0, 1)
    dehazed_frame = (scene_radiance * 255).astype(np.uint8)
    return dehazed_frame


def main(video_path,output_path:str,dehaze_filter:bool=False,display:bool=False):
    video_path = video_path

    camera = cv2.VideoCapture(video_path)
    output_path = output_path
    fps = int(camera.get(5))

    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    prev_time = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = tqdm(total = length)
    print(camera.isOpened())
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            current_time = time.time()
            if dehaze_filter:
              frame = cv2_dehaze_filter(frame)

            res = dehaze_image(frame)
            res = res.squeeze().permute(1,2,0).cpu().detach().numpy()*255
            if res.dtype != 'uint8':
              res = cv2.convertScaleAbs(res)
            fps_text = f"FPS: {1 / (current_time - prev_time):.2f}"
            prev_time = current_time
            cv2.putText(res, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if display:
              cv2.imshow("frame",res)
            out.write(res)
            bar.update(1)
        else:
            break
    camera.release()
    out.release()
    cv2.destroyAllWindows()


main(video_path="/content/fog_indoor.mp4",
    output_path="/content/AOD/output.mp4",
    dehaze_filter=True,
    display=False)
    
