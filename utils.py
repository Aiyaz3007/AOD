import matplotlib.pyplot as plt
import numpy as np

figure, axis = plt.subplots(1, 2)

def updateLoss(epochs:list,trainloss:list,valloss:list):
    plt.title("train loss")
    axis[0,0].plot(epochs,trainloss)
    axis[0,1].plot(epochs,valloss)
    figure.savefig("Loss.png")