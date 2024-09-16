import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread, imwrite
import pandas as pd
import pickle as pkl
import copy
from skimage.filters import gaussian
from skimage import filters, measure, morphology, segmentation
from skimage.morphology import dilation, disk
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle
import imageio
import torch




def create_dataset_csv(df, 
                       data_path, 
                       save_path, 
                       patch_size=40,
                       minmax_norm=True,
                       filter_channel=True,
                       channel=None,
                       has_label = False):
    # df is the dataframe 
    # This function create the dataset using the center in the csv file
    # If the cell in the csv file doesn't exist in the mask file or region of mask < 300,generate a fake mask
   
    
    # Fine the x,y coordianate
    from create_dictionary import read_mask

    marker_patch_folder = os.path.join(save_path, "marker_patch")
    csv_folder = os.path.join(save_path, "csv_file")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(marker_patch_folder))
        os.mkdir(os.path.join(csv_folder))
    

    # load image
    mask_path = os.path.join(data_path, 'mask')
    marker_path = os.path.join(data_path,'marker')

    mask_folder = os.listdir(mask_path)
    marker_folder = os.listdir(marker_path)
    marker_folder,mask_folder = align(marker_folder,mask_folder)
    print(mask_folder)
    print(marker_folder)
    fake_counts ={}

   
    for i in range(len(mask_folder)):
        # index the df for each image
        fake_count = 0
        mask_file = mask_folder[i]
        marker_file = marker_folder[i]

        file = mask_file.split("_mask")[0]
        print("Processing mask file:", mask_file)
        print("Processing marker file:", marker_file)


        csv_file_path = os.path.join(csv_folder, f"{file}.csv")

        marker_patch = os.path.join(marker_patch_folder, file)
        if not os.path.exists(marker_patch):
            os.mkdir(marker_patch)
        assert mask_file.split("_mask")[0] == marker_file.split("_marker")[0]
        
        file_prefix = mask_file.split("_mask")[0]
        df_ = df[df['file_prefix'] == file_prefix].reset_index(drop=True)
        x = df_['x'].astype(int)
        y = df_['y'].astype(int)

        if has_label == True:
            labels = df_['cell_type']
            label_string = []
        
    
        marker = imread(os.path.join(marker_path,marker_file)).astype(np.float32)
        mask = read_mask(os.path.join(mask_path,mask_file))
        
        print("mask shape:",mask.shape)
        print("marker shape:",marker.shape)
        if filter_channel:
            assert channel is not None
            
            marker = marker[channel, :, :]
            print("After reshape:",marker.shape)
        # Normalize the marker 
        if minmax_norm:
            marker = minmax_normalization(marker)
        

        with open(csv_file_path, 'w', encoding='UTF8') as csvfile:
            csvfile.write("marker_path")
            
        
            csvfile.write("\n")
            for ii in tqdm(range(len(x))):
                label_string.append(labels[ii])
                center = (y[ii], x[ii])
                mask_index = mask[center]
                y_min = max(0, y[ii]-patch_size//2)
                y_max = min(marker.shape[1], y_min+patch_size)
                if y_max - y_min < patch_size and y_max == marker.shape[1]:
                    y_min = y_max - patch_size
                if y_max - y_min < patch_size and y_min == 0:
                    y_max = y_min + patch_size
                
                x_min = max(0, x[ii]-patch_size//2)
                x_max = min(marker.shape[2], x_min+patch_size)
                if x_max - x_min < patch_size and x_max == marker.shape[2]:
                    x_min = x_max - patch_size
                if x_max - x_min < patch_size and x_min == 0:
                    x_max = x_min + patch_size
           
                im_temp = copy.deepcopy(marker[:, y_min:y_max, x_min:x_max])
                mask_temp_ = mask[y_min:y_max, x_min:x_max] == mask_index
                a = np.sum(mask_temp_)
                if mask_index == 0 or a == 0:

                    mask_temp = generate_fake_mask(patch_size)
                    fake_count += 1

                else:
                    mask_temp = mask_temp_
                

                # apply soft mask

                mask_smooth = smooth(mask_temp)
                im_temp = apply_softmask(mask_smooth,im_temp)
                
                
                
        
                f = os.path.join(marker_path, r"{}.tiff".format(ii))
            

            
                tensor_image = torch.tensor(im_temp, dtype=torch.float16)
                torch.save(tensor_image, f"{save_path}/marker_patch/{file}/{ii}.pt")
            
                csvfile.write(f)
                csvfile.write("\n")
            
        
        print("Fake mask ratio:",fake_count/float(len(x)))
        fake_counts[file] = fake_count/float(len(x))

        if has_label == True:
            label_folder = os.path.join(save_path, "label")
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            # save the label (a list of string) as a txt file
            file_path = os.path.join(label_folder, f"{file}_label.txt")
            with open(file_path, 'w') as fi:
                for string in label_string:
                    fi.write(f"{string}\n")   

    # save the fake_mask_ratio dictionary
    with open(os.path.join(save_path, 'fake_counts.pkl'), 'wb') as pickle_file:
        pickle.dump(fake_counts, pickle_file)

def apply_softmask(mask_smooth,im_temp):
    for j in range(im_temp.shape[0]):
        channel_min = np.min(im_temp[j, :, :])
        im_temp[j, :, :] -= channel_min
        im_temp[j, :, :] = mask_smooth * im_temp[j, :, :]
        im_temp[j, :, :] += channel_min
    return im_temp

def generate_fake_mask(patch_size):
    fake_mask = np.zeros((patch_size, patch_size))
    for i in range(patch_size):
        for j in range(patch_size):
            if (i-patch_size//2)**2 + (j-patch_size//2)**2 < (patch_size//2 - 7)**2:
                fake_mask[i, j] = 1
    return fake_mask


def align(marker_list,mask_list):
    # a function to align two list make sure reading the matched marker and mask files
    #marker_indices = {marker.split('_marker')[0]: i for i, marker in enumerate(marker_list)}
    mask_indices = {mask.split('_mask')[0]: i for i, mask in enumerate(mask_list)}
    
    matching_marker = []
    matching_mask = []
    
    for marker in marker_list:
        marker_prefix = marker.split('_marker')[0]
        mask_index = mask_indices.get(marker_prefix)
        if mask_index is not None:
            matching_marker.append(marker)
            matching_mask.append(mask_list[mask_index])
    return matching_marker, matching_mask



def minmax_normalization(im):
    """a function to perform min-max normalization the whole image for each channel 

    Args:
        im (np.array): original image

    Returns:
        im (np.array): image after min max normalization; value range -1 to 1
    """

    for i in range(im.shape[0]):
        idxx = np.where(im[i,:,:]>0)
        if len(idxx[0]) == 0:
            continue
        thresh = np.percentile(im[i,:,:][idxx],98)
        if thresh > 0:
            im[i,:,:] = np.clip(im[i,:,:],0,thresh)
        im[i,:,:] = im[i,:,:]/np.max(im[i,:,:])
        im[i,:,:] = 2*im[i,:,:]-1
    return im

def smooth(mask):
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
    smooth /= np.max(smooth)

    return smooth



if __name__ == "__main__":
    import pandas as pd
    df_path = "../../data/tonsil_labeled/BE_Tonsil_l3_dryad.csv"    
    df = pd.read_csv(df_path)
    df = df[(df['sample_name'] == 'tonsil')]
    data_path = "../../data/tonsil_labeled/"   
    save_path = data_path + "dataset/"
    create_dataset_csv(df, data_path, save_path, patch_size=40, minmax_norm = True, filter_channel = False, channel = None)
  

    