import numpy as np
import cv2
import random
from tqdm import *
import os
 
# calculate means and std
train_txt_path = '../dataset/train_for_mean_std.txt'
 
CNum = 20   # number of images selected for calculating mean and std dev
 
img_h, img_w = 709, 546
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []
 
with open(train_txt_path, 'r') as f:
  lines = f.readlines()
  random.shuffle(lines)  # shuffle , select images randomly
 
  for i in tqdm_notebook(range(CNum)):
    #print("i=",i)
    img_path = os.path.join('../dataset/labelled/1440_whitebuoy/', lines[i].rstrip().split()[0])
    #print("img_path",img_path)
    img = cv2.imread(img_path)
 
    img = cv2.resize(img, (img_h, img_w))

    #cv2.imshow('e', img)
    #cv2.waitKey(0)
    img = img[:, :, :, np.newaxis]
 
    imgs = np.concatenate((imgs, img), axis=3)
 
 
imgs = imgs.astype(np.float32)/255.
 
for i in tqdm_notebook(range(3)):
  pixels = imgs[:,:,i,:].ravel()
  means.append(np.mean(pixels))
  stdevs.append(np.std(pixels))
 
means.reverse() # BGR --> RGB
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))