#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os, random, sys
from tqdm.auto import tqdm
import cv2
import numpy as np


# In[51]:


if len(sys.argv) > 1:
    train_data_folder = sys.argv[1]
else:
    train_data_folder = '/home/zoro/NTIRE/NAFNet_v2/TRAIN_cvlai/Train' 

hr_folder = os.path.join(train_data_folder, 'HR')
lr_folder = os.path.join(train_data_folder, 'LR_x4') 

N_total = len(os.listdir(hr_folder))//2 

indexes = np.arange(1,N_total+1)
random.shuffle(indexes)

train_idxs = indexes[:700] 
val_idxs = indexes[700:]


# In[45]:


def GenerateTrainingPatches(hr_path, lr_path, idxs): 
    Datasets = ['Flickr1024']
    idx_patch = 1
    scale= 4
    img_fmt = '0000{}_{}.png'
    save_folder_fmt = '000000{}'
    
    for idx_dataset in range(len(Datasets)):
        dataset = Datasets[idx_dataset]
        img_list_hr = sorted(os.listdir(hr_path))
        img_list_lr = sorted(os.listdir(lr_path))
    
        # print(len(img_list_hr), len(img_list_lr))
        
        for idx_file in tqdm(idxs): 
            img_hr_0 = os.path.join(hr_path,img_fmt.format(idx_file,'L')[-10:])
            img_hr_1 = os.path.join(hr_path,img_fmt.format(idx_file,'R')[-10:])
            img_lr_0 = os.path.join(lr_path,img_fmt.format(idx_file,'L')[-10:])
            img_lr_1 = os.path.join(lr_path,img_fmt.format(idx_file,'R')[-10:])
            
            # print(img_hr_0,'\n', img_hr_1)
            # print(img_lr_0,'\n', img_lr_1)
            
            img_hr_0 = cv2.imread(img_hr_0)
            img_hr_1 = cv2.imread(img_hr_1)
            img_lr_0 = cv2.imread(img_lr_0)
            img_lr_1 = cv2.imread(img_lr_1)
            
            for x_lr in range(3, img_lr_0.shape[0] - 33, 20):
                for y_lr in range(3, img_lr_0.shape[1] - 93, 20):
                    x_hr = (x_lr - 1) * scale + 1
                    y_hr = (y_lr - 1) * scale + 1
                    hr_patch_0 = img_hr_0[x_hr : (x_lr+29)*scale+1, y_hr : (y_lr+89)*scale+1, :]
                    hr_patch_1 = img_hr_1[x_hr : (x_lr+29)*scale+1, y_hr : (y_lr+89)*scale+1, :]
                    lr_patch_0 = img_lr_0[x_lr : x_lr+30, y_lr : y_lr+90, :]
                    lr_patch_1 = img_lr_1[x_lr : x_lr+30, y_lr : y_lr+90, :]
                    
                    
                    folder_name = save_folder_fmt.format(idx_patch)[-6:]
                    
                    os.makedirs(f'./datasets/StereoSR/patches_x{scale}/{folder_name}')
                    cv2.imwrite(f'./datasets/StereoSR/patches_x{scale}/{folder_name}/hr0.png', hr_patch_0)
                    cv2.imwrite(f'./datasets/StereoSR/patches_x{scale}/{folder_name}/hr1.png', hr_patch_1)
                    cv2.imwrite(f'./datasets/StereoSR/patches_x{scale}/{folder_name}/lr0.png', lr_patch_0)
                    cv2.imwrite(f'./datasets/StereoSR/patches_x{scale}/{folder_name}/lr1.png', lr_patch_1)
                    # print(f'{folder_name} training samples have been generated...')
                    idx_patch += 1


# In[47]:


def GenerateValidationPatches(hr_path, lr_path, idxs): 
    Datasets = ['Flickr1024']
    idx_patch = 1
    scale= 4
    img_fmt = '0000{}_{}.png'
    save_folder_fmt = '000000{}'
    
    for idx_dataset in range(len(Datasets)):
        dataset = Datasets[idx_dataset]
        img_list_hr = sorted(os.listdir(hr_path))
        img_list_lr = sorted(os.listdir(lr_path))
    
        # print(len(img_list_hr), len(img_list_lr))
        
        for idx_file in tqdm(idxs): 
            img_hr_0 = os.path.join(hr_path,img_fmt.format(idx_file,'L')[-10:])
            img_hr_1 = os.path.join(hr_path,img_fmt.format(idx_file,'R')[-10:])
            img_lr_0 = os.path.join(lr_path,img_fmt.format(idx_file,'L')[-10:])
            img_lr_1 = os.path.join(lr_path,img_fmt.format(idx_file,'R')[-10:])
            
            # print(img_hr_0,'\n', img_hr_1)
            # print(img_lr_0,'\n', img_lr_1)
            
            img_hr_0 = cv2.imread(img_hr_0)
            img_hr_1 = cv2.imread(img_hr_1)
            img_lr_0 = cv2.imread(img_lr_0)
            img_lr_1 = cv2.imread(img_lr_1)
            
            for x_lr in range(3, img_lr_0.shape[0] - 33, 20):
                for y_lr in range(3, img_lr_0.shape[1] - 93, 20):
                    x_hr = (x_lr - 1) * scale + 1
                    y_hr = (y_lr - 1) * scale + 1
                    hr_patch_0 = img_hr_0[x_hr : (x_lr+29)*scale+1, y_hr : (y_lr+89)*scale+1, :]
                    hr_patch_1 = img_hr_1[x_hr : (x_lr+29)*scale+1, y_hr : (y_lr+89)*scale+1, :]
                    lr_patch_0 = img_lr_0[x_lr : x_lr+30, y_lr : y_lr+90, :]
                    lr_patch_1 = img_lr_1[x_lr : x_lr+30, y_lr : y_lr+90, :]
                    
                    
                    folder_name = save_folder_fmt.format(idx_patch)[-6:]
                    
                    os.makedirs(f'./datasets/StereoSR/test/cvpr2023/hr/{folder_name}')
                    os.makedirs(f'./datasets/StereoSR/test/cvpr2023/lr_x{scale}/{folder_name}')
                    cv2.imwrite(f'./datasets/StereoSR/test/cvpr2023/hr/{folder_name}/hr0.png', hr_patch_0)
                    cv2.imwrite(f'./datasets/StereoSR/test/cvpr2023/hr/{folder_name}/hr1.png', hr_patch_1)
                    cv2.imwrite(f'./datasets/StereoSR/test/cvpr2023/lr_x{scale}/{folder_name}/lr0.png', lr_patch_0)
                    cv2.imwrite(f'./datasets/StereoSR/test/cvpr2023/lr_x{scale}/{folder_name}/lr1.png', lr_patch_1)
                    # print(f'{folder_name} training samples have been generated...')
                    idx_patch += 1
                    


# In[48]:


# try:
GenerateTrainingPatches(hr_folder, lr_folder, train_idxs)
GenerateValidationPatches(hr_folder, lr_folder, val_idxs)
# except: 
#     print("COULD NOT GENERATE. DEBUG")
# else: 
#     print("DATASET GENERATED SUCCESSFULLY")


# In[ ]:




