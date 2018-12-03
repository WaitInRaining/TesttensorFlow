#!usr/bin/python
# coding:utf-8

import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
x = [10,20,30,40,50,60,70,80,90,100]
pearson = [0.94578,0.92046,0.91075,0.89469,0.88954,0.88085,0.88125,0.87924,0.87648,0.87868]
jaccard =[0.86526,0.83521,0.80915,0.78945,0.77435,0.77055,0.76987,0.770236,0.770146,0.771847]
ajcard = [0.84569,0.80525,0.77551,0.75556,0.75086,0.74925,0.74989,0.75012,0.74857,0.75013]
my = [0.84049,0.78524,0.75516,0.73356,0.72525,0.72136,0.72068,0.72105,0.72139,0.72256]



fig = plt.figure()

plt.plot(x, pearson, marker="o", label="Pearson")
plt.plot(x, jaccard, marker="*", label="Jaccard-Pearson")
plt.plot(x, ajcard, marker="+", label="文献7算法")
plt.plot(x, my, marker="x", label="本文算法")

plt.legend()
plt.xlabel("number of choose simUser")
plt.ylabel("MAE")
plt.title("MAE of Top-N in MovieLens 100K")
plt.show()