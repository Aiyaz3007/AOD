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
from utils import updateLoss
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings("ignore")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.apply(weights_init)

	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path)		
	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	dehaze_net.train()
	epochsList = []
	trainLossList = []
	valLossList = []
	

	bar = tqdm(total=config.num_epochs)

	for epoch in range(config.num_epochs):
		epochsList.append(epoch)
		train_loss = 0
		val_loss = 0
		for iteration, (img_orig, img_haze) in enumerate(train_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean_image = dehaze_net(img_haze)

			t_loss = criterion(clean_image, img_orig)

			optimizer.zero_grad()
			t_loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
			optimizer.step()
			train_loss+=t_loss

		total_train_loss = round(train_loss.item()/len(train_loader),2)
		trainLossList.append(total_train_loss)
		print(f"\nTrain Loss: {total_train_loss}")

				
				

		# Validation Stage
		
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()
			clean_image = dehaze_net(img_haze)	
			with torch.no_grad():
				v_loss = criterion(clean_image, img_orig)
				val_loss += v_loss

			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(epoch)+".jpg")

		# torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth") 
		total_val_loss = round(val_loss.item()/len(val_loader),2)
		valLossList.append(total_val_loss)
		print(f"\nVal Loss: {total_val_loss}")
		torch.save(dehaze_net.state_dict(), config.snapshots_folder + f"Epoch_{str(epoch)}_val_{val_loss/len(val_loader)}.pth") 
		updateLoss(epochs=epochsList,trainloss=trainLossList,valloss=valLossList)
		bar.update(1)





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="data/images/")
	parser.add_argument('--hazy_images_path', type=str, default="data/data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--train_batch_size', type=int, default=16)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=200)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)








	
