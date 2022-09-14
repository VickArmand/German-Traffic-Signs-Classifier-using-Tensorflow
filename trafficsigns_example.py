# Dataset used is the GTSRB - German Traffic Sign Recognition Benchmark find it on kaggle
import os
import glob
from posixpath import split
import shutil
from sklearn.model_selection import train_test_split
def split_data(path_to_data,path_to_train,path_to_val,split_size=0.1):
    folders=os.listdir(path_to_data)
    for folder in folders:
        full_path=os.path.join(path_to_data,folder)#concatenating name of path with the image folder
        images_path=glob.glob(os.path.join(full_path,'*.png'))#load all files with extension png 
        x_train,x_val=train_test_split(images_path,test_size=split_size)#split the dataset into train and validation sets
        for x in x_train:
            path_to_folder=os.path.join(path_to_train,folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x,path_to_folder)#saving the trained images in the trained folder
        for x in x_val:
            path_to_folder=os.path.join(path_to_val,folder)
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
            shutil.copy(x,path_to_folder)#saving the validation images in the validation folder 
            
if __name__=='__main___':
    path_to_data = "C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Train"
    path_to_train="C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Model Training Data\\Train"
    path_to_val="C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\Datasets\\Model Training Data\\Validation"
    split_data(path_to_data,path_to_train=path_to_train,path_to_val=path_to_val)