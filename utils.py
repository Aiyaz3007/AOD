import matplotlib.pyplot as plt
import numpy as np

def updateTrainLoss(epochs:list,loss:list):
    plt.plot(epochs,loss)
    plt.savefig("TrainLoss.png")

def updateValLoss(epochs:list,loss:list):
    plt.plot(epochs,loss)
    plt.savefig("ValLoss.png")