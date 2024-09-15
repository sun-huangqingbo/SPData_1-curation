### Datasets performed by csv
import pandas as pd
import sys 
import os
sys.path.append("utils")
from crop_cells_csv import create_dataset_csv
from create_dictionary import create_cell_pos_dict
from create_dataset import create_dataset_segmentation

### Dataset generated using 'csv version':
### All the csv has been processed to have the correct format
### All the data_path has the folder "marker" and "mask" containing image files we need to be processed 

### 1. Human colorectal cancer CODEX dataset 
df_path = "projects/murphylab/cell_phenotyping//data/CRC/CRC_clusters_neighborhoods_markers.csv"    
df = pd.read_csv(df_path)
data_path = "projects/murphylab/cell_phenotyping//data/CRC/A/" 
#data_path = "../../data/CRC/B/" 
save_path = os.path.join(data_path, "dataset")
crc_filter_channel = [(1, 0),(1, 1),(1, 2),(1, 3),(2, 1),(2, 2),(2, 3),(3, 1),(3, 2),(3, 3),(4, 1),(4, 2),(4, 3),(5, 1),(5, 2),(5, 3),
            (6, 1),(6, 2),(6, 3),(7, 1),(7, 2),(7, 3),(8, 1),(8, 2),(8, 3),(9, 1),(9, 2),(9, 3),(10, 1),(10, 2),(10, 3),(11, 1),(11, 2),(11, 3),
            (12, 1),(12, 2),(12, 3),(13, 1),(13, 2),(13, 3),(14, 1),(14, 2),(14, 3),(15, 1),(15, 2),(15, 3),(16, 1),(16, 2),(16, 3),(17, 1),(17, 2),
            (17, 3),(18, 1),(18, 2),(18, 3),(19, 1),(19, 2),(19, 3),(20, 2),(20, 3),(22, 3)]

create_dataset_csv(df, 
                   data_path, 
                   save_path,               
                   patch_size = 40,        
                   minmax_norm = True,     
                   filter_channel = True,  
                   channel = crc_filter_channel,          
                   has_label = True) 

### 2. Intestine datatset
df_path = "/projects/murphylab/cell_phenotyping/data/Intestine_dryad/23_09_CODEX_HuBMAP_alldata_Dryad_merged_s.csv"    
df = pd.read_csv(df_path)
data_path = "/projects/murphylab/cell_phenotyping/data/Intestine_dryad/" 
save_path = os.path.join(data_path, "dataset")

intestine_filter_channel = [(1, 0),(1, 1),(1, 2),(1, 3),(2, 1),(2, 2),(2, 3),(3, 1),(3, 2),(3, 3),(4, 1),(4, 2),(4, 3),(5, 1),(5, 2),(5, 3),
            (6, 1),(6, 2),(6, 3),(7, 1),(7, 2),(7, 3),(8, 1),(8, 2),(8, 3),(9, 1),(9, 2),(9, 3),(10, 1),(10, 2),(10, 3),(11, 1),(11, 2),(11, 3),
            (12, 2),(12, 3),(13, 1),(13, 2),(13, 3),(14, 2),(14, 3),(15, 2),(15, 3),(16, 2),(16, 3),(17, 2),
            (17, 3),(18, 2),(18, 3),(19, 2),(19, 3),(20, 2),(20, 3),(21, 3),(23,3)]

create_dataset_csv(df, 
                   data_path, 
                   save_path,               
                   patch_size = 40,        
                   minmax_norm = True,     
                   filter_channel = True,  
                   channel = intestine_filter_channel,         
                   has_label = True)     


### 3. Lymphoma CODEX dataset 
df_path = "/projects/murphylab/cell_phenotyping//data/lymphoma lymph nodes/cells_annotation.csv"    
df = pd.read_csv(df_path)
data_path = "/projects/murphylab/cell_phenotyping//data/lymphoma lymph nodes/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size = 40, 
                   minmax_norm = True, 
                   filter_channel = False,
                   channel = None,
                   has_label = True)

### 4. Triple negative breast cancer MIBI-TOF 
df_path = "/projects/murphylab/cell_phenotyping/data/breast_cancer_new/Single_Cell_Data.csv"    
df = pd.read_csv(df_path)
data_path = "/projects/murphylab/cell_phenotyping/data/breast_cancer_new/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size = 40, 
                   minmax_norm = True, 
                   filter_channel = False,
                   channel = None,
                   has_label = True)


### 5. Classic Hodgkin Lymphoma CODEX dataset
df_path = "/projects/murphylab/cell_phenotyping/data/Classic Hodgkin Lymphoma CODEX dataset/cHL_CODEX_annotation.csv“    
df = pd.read_csv(df_path)
data_path = "projects/murphylab/cell_phenotyping/data/Classic Hodgkin Lymphoma CODEX dataset/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size = 40, 
                   minmax_norm = True, 
                   filter_channel = False,
                   channel = None,
                   has_label = True)


### 6. Classic Hodgkin Lymphoma MIBI dataset
df_path = "/projects/murphylab/cell_phenotyping/data/cHL_2_MIBI/cell_table.csv“    
df = pd.read_csv(df_path)
data_path = "projects/murphylab/cell_phenotyping/data/cHL_2_MIBI/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size = 40, 
                   minmax_norm = True, 
                   filter_channel = False,
                   channel = None,
                   has_label = True)

### 7. Tonsil CODEX Image II dataset 
df_path = "/projects/murphylab/cell_phenotyping/data/tonsil_labeled/BE_Tonsil_l3_dryad.csv"    
df = pd.read_csv(df_path)
df = df[(df['sample_name'] == 'tonsil')]
data_path = "projects/murphylab/cell_phenotyping/data/tonsil_labeled/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size=40, 
                   minmax_norm=True, 
                   filter_channel=False,
                   channel = None,
                   has_label = True)


### 8. Fetally derived extravillous trophoblast MIBI dataset
df_path = "/projects/murphylab/cell_phenotyping/data/EVT/Supplementary_table_3_single_cells_updated.csv"    
df = pd.read_csv(df_path)

data_path = "/projects/murphylab/cell_phenotyping/data/tonsil_labeled/"   
save_path = os.path.join(data_path, "dataset")
create_dataset_csv(df, 
                   data_path, 
                   save_path, 
                   patch_size=40, 
                   minmax_norm=True, 
                   filter_channel=False,
                   channel = None,
                   has_label = True)



###############################################



### Dataset generated by 'segmentation version'
### 1. HuBMAP 29-marker CODEX dataset  
df = pd.read_csv("/projects/murphylab/cell_phenotyping/data/HuBMAP_data/SPLEEN_29C_UL.csv",sep=",")
marker_paths = df['data_path']
mask_paths = df['mask_path']
data_path = "/projects/murphylab/marker_opt_proj_data/HuBMAP/" 
save_path = "/projects/murphylab/cell_phenotyping/data/HuBMAP_data/"   
cell_position_path = os.path.join(save_path, "cell_position_dictionary")
print(df)
for i in range(len(marker_paths)):
    marker_file = marker_paths.iloc[i].split("HuBMAP/")[-1]
    mask_file = mask_paths.iloc[i].split("HuBMAP/")[-1]
    mask_path = mask_paths.iloc[i].split("HBM")[0] 

    create_cell_pos_dict(base_dir,mask_path)

    create_dataset_segmentation(data_path,
                    save_path, 
                    patch_size = 40,  
                    skip_norm = False
                    )

### 2. Multi-tumor CODEX image dataset 

data_path = "/projects/murphylab/cell_phenotyping/data/multi-tumor/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary/")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path

create_cell_pos_dict(base_dir,mask_path)
create_dataset_segmentation(data_path,
                    save_path, 
                    patch_size = 40,  
                    skip_norm = False
                    )

### 3. Breast cancer progression MIBI dataset 
data_path = "/projects/murphylab/cell_phenotyping/data/breast_II/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path
create_cell_pos_dict(data_path,mask_path)
create_dataset_segmentation(data_path,
                    save_path, 
                    patch_size = 40,  
                    skip_norm = True
                    )

### 4. Lung dataset
data_path = "/projects/murphylab/cell_phenotyping/data/lung/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path
create_cell_pos_dict(data_path,mask_path)
create_dataset_segmentation(data_path,
                    save_path, 
                    patch_size = 40,  
                    skip_norm = True
                    )


### 5. TB dataset
### This dataset requires a cell index csv
data_path = "/projects/murphylab/cell_phenotyping/data/TB/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path
cell_index_csv = os.path.join(data_path, "allTB-sarcoid-scdata.csv")
create_cell_pos_dict(data_path, mask_path)
create_dataset_segmentation(data_path,
                    save_path, 
                    patch_size = 40,  
                    skip_norm = True,
                    cell_index_csv = cell_index_csv
                    )

### 6. Bone marrow and Acute Myeloid Leukemia (AML) CODEX dataset 
### This dataset requires a cell index csv
data_path = "/projects/murphylab/cell_phenotyping/data/bone_marrow/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path
cell_index_csv = os.path.join(data_path, "annotation_BM.csv")
#cell_index_csv = os.path.join(data_path, "annotation_AML.csv"
create_cell_pos_dict(data_path,mask_path)
create_dataset_segmentation(data_path,
                        save_path,
                        patch_size = 40,
                        skip_norm = True,
                        cell_index_csv = cell_index_csv)

### 7. Human kidney CODEX image datase
data_path = "/projects/murphylab/cell_phenotyping/data/kidney/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary")      
mask_path = os.path.join(data_path, "mask")
save_path = data_path
create_cell_pos_dict(data_path,mask_path)
create_dataset_segmentation(data_path,
                            save_path,
                            patch_size = 40,
                            skip_norm = True
                           )

### 8. Tonsil CODEX Image I dataset
data_path = "/projects/murphylab/cell_phenotyping/data/tonsil/"
cell_position_path = os.path.join(data_path, "cell_position_dictionary") 
mask_path = os.path.join(data_path, "mask")
save_path = data_path
create_cell_pos_dict(data_path,mask_path)
create_dataset_segmentation(data_path,
                            save_path,
                            patch_size = 40,
                            skip_norm = True
                           )


