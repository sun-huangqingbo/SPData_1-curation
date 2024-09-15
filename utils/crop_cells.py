# -*- coding: utf-8 -*-
"""crop_cells.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N67qJnKasCJHsKQ4qOkt7tEyq6WJ_7Ih
"""

from tqdm import tqdm
from tifffile import imread,imwrite
import numpy as np
import os
from skimage.filters import gaussian
from skimage import filters, measure, morphology, segmentation

from skimage.morphology import dilation, disk
import torch
# This function crops cell with cell_id:position_index dictionary
def crop_cell(dict,
              img, 
              mask,
              save_path,
              file_name,
              patch_size=40,
              cell_index = None):
  
  csv_path = os.path.join(save_path, 'csv_file')
  csv_file = file_name + ".csv"
  csv_file_path = os.path.join(csv_path, csv_file)
  if not os.path.exists(csv_path):
      os.makedirs(csv_path)
  
  min_val = np.min(img,axis=(1,2),keepdims=True)
  img_zero = img-min_val
  
  with open(csv_file_path, 'w', encoding='UTF8') as csvfile:
    csvfile.write("marker_path")
    
    csvfile.write("\n")

   
    if cell_index is None:
        cell_index = dict.keys()

    for c in tqdm(cell_index):
      
      x_mean = (min(dict[c][0])+max(dict[c][0]))//2
      xmin = x_mean-patch_size/2
      xmin = int(max(xmin,0))
      xmax = int(min(x_mean+patch_size/2,img.shape[1]))

      y_mean = (min(dict[c][1])+max(dict[c][1]))//2
      ymin = y_mean-patch_size/2
      ymin = int(max(ymin,0))
      ymax = int(min(y_mean+patch_size/2,img.shape[2]))

      img_patch = np.zeros((img.shape[0],patch_size,patch_size))
      mask_patch = np.zeros((mask.shape[0],patch_size,patch_size))
      img_zero_patch = np.zeros((img.shape[0],patch_size,patch_size))
      img_patch[:,:(xmax-xmin),:(ymax-ymin)] = img[:,xmin:xmax,ymin:ymax]
      img_zero_patch[:,:(xmax-xmin),:(ymax-ymin)] = img_zero[:,xmin:xmax,ymin:ymax]
      mask_patch[:,:(xmax-xmin),:(ymax-ymin)] = mask[:,xmin:xmax,ymin:ymax]

      mask_smooth = smooth(mask_patch[0],c)
      
     
      marker_a = img_zero_patch*mask_smooth
      marker_a = marker_a+min_val

      
      marker_path = os.path.join(save_path, 'marker_patch', file_name)
      if not os.path.exists(marker_path):
        os.makedirs(marker_path)

  
      f = os.path.join(marker_path, r"{}.pt".format(c))
      tensor_image = torch.tensor(marker_a, dtype=torch.float32)
      torch.save(tensor_image, f)
      a = tensor_image.numpy()
      diff = np.sum(a - marker_a)

      assert diff < 1
              
      

      csvfile.write(f)
      
      csvfile.write("\n")
     

def smooth(mask,c):
    mask = mask == c
    smooth = mask.astype("f")
    count = 1
    for j in range (1, 5, 1):
        mask_dilated = dilation(mask, disk(j))

        smooth += mask_dilated.astype("f")
        count += 1
        for i in np.arange(0, j-1, 1):
            smooth += filters.gaussian(mask_dilated, sigma=1+i)
            count += 1
    smooth /=  count
    smooth /= np.max(smooth+1e-6)

    return smooth

