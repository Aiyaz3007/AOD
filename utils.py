import matplotlib.pyplot as plt
import numpy as np

figure, axis = plt.subplots(2, 1)

def updateLoss(epochs:list,trainloss:list,valloss:list):
    axis[0].plot(epochs,trainloss)
    axis[0].set_title("Train Loss")
    axis[1].plot(epochs,valloss)
    axis[1].set_title("Val Loss")

    figure.savefig("Loss.png")