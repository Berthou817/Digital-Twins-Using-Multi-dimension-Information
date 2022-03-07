import numpy as np
import matplotlib.pyplot as plt
data = np.fromfile("res_without_20.raw",dtype=np.uint8).reshape((-1,1024,1024))
# data = np.where(data==0,np.ones_like(data),np.zeros_like(data))
# # np.save("trian.npy",data.flatten())
# print(len(data))
porosity = 1-np.mean(data,axis=-1,keepdims=True).mean(axis=-2,keepdims=True).reshape((-1,1))
np.save("ptestwithout1024.npy",porosity)
# print(porosity)
# plt.plot()
# plt.show()
test = np.load("ptestwithout1024.npy")
target = np.load("pore1024.npy")

print(np.mean(test),np.mean(target))
plt.plot(np.arange(len(test)),test,label="test")
plt.plot(np.arange(len(test)),target,label="target")
plt.legend()
plt.show()
