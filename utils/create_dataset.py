# -*- coding: utf-8 -*-
"""create_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W0jHB8hywE3DMdLoY-kYWs4qEv5o9aIu
"""

from tifffile import imread
import pickle
import os
import pandas as pd
import numpy as np
from crop_cells import crop_cell
from create_dictionary import *
from skimage.filters import gaussian

from crop_cells_csv import align, minmax_normalization


  
def create_dataset_segmentation(data_path,
                        save_path,
                        patch_size = 25,
                        skip_norm = False):
    

  # This create_dataset function create all the data into one folder called 'dataset'
  # will not generate train or val dataset
  # skip normalization if the image is already normalized
    from create_dictionary import read_mask

    if not os.path.exists(save_path):
          os.mkdir(save_path)
    
    mask_path = os.listdir(os.path.join(data_path, 'mask'))
    marker_path = os.listdir(os.path.join(data_path, 'marker'))
  
    marker_path,mask_path = align(marker_path,mask_path)
  
    print(mask_path)
    print(marker_path)

    for i in range(len(mask_path)):

      mask_file = mask_path[i]
      marker_file = marker_path[i]
  
      print(mask_file)
      print(marker_file)
      assert mask_file.split("_")[0] == marker_file.split("_")[0]

     
      cell_dict_path = os.path.join(data_path, 'cell_position_dictionary', 'cell_pos_' + mask_file + ".pkl")
      cell_dict = pickle.load(open(cell_dict_path, "rb"))

      mask = read_mask(os.path.join(data_path, 'mask/',mask_file))[np.newaxis,:,:]
      print("Mask reading finished")
      marker = imread(os.path.join(data_path, 'marker', marker_file)).astype(np.float32)
      print("Marker reading finished")
      if not skip_norm:
        marker = minmax_normalization(marker)
        os.makedirs(os.path.join(data_path,'normalized_marker'),exist_ok=True)
        np.save(os.path.join(data_path,'normalized_marker',f'normalized_{marker_file}'),marker)
    
      dataset_save_path = os.path.join(save_path, 'dataset')
      os.makedirs(dataset_save_path,exist_ok=True)

     
      

      crop_cell(cell_dict, 
                marker, 
                mask, 
                dataset_save_path,
                mask_file,
                patch_size=patch_size)
      
      
      marker_files = os.listdir(os.path.join(dataset_save_path, 'marker_patch', mask_file))
     
    
      csv_file = pd.read_csv(os.path.join(dataset_save_path, 'csv_file', mask_file +'.csv'))
      row_num = len(csv_file.index)
  
      print("Marker file numbers:",len(marker_files))
      print("csv_file length:",row_num)
      del marker
      del mask
    
      assert len(marker_files) == row_num
      




if __name__ == "__main__":
    import os
    import pandas as pd
    from create_dictionary import create_cell_pos_dict
    
    ### lung dataset
    data_path = "/projects/murphylab/cell_phenotyping/data/lung/"
    cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
    mask_path = os.path.join(data_path, "mask")
    save_path = data_path
    
    create_cell_pos_dict(cell_position_path,mask_path)

    create_dataset_segmentation(data_path,save_path,patch_size=40,skip_norm=False)
   