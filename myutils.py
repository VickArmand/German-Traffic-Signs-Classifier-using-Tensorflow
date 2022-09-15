import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
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

def order_test_set(path_to_images,path_to_csv_file,path_to_new_images):#here we are preparing the test set
    #here we want to create a dictionary with the name of image as key and the class it belongs to as value
    try:
        with open(path_to_csv_file,'r') as csvfile:
            reader=csv.reader(csvfile,delimiter=',')
            for i,row in enumerate(reader):
                if i==0:#to skip the first row
                    continue
                
                img_name=row[-1].replace('Test/','')
                label=row[-2]
                path_to_folder=os.path.join(path_to_new_images,label)
                path_to_folder_exist=os.path.join(path_to_folder,img_name)
                if os.path.exists(path_to_folder_exist):
                    continue
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)
                img_full_path=os.path.join(path_to_images,img_name)
                shutil.move(img_full_path,path_to_folder)
    except:
        print('Error reading csv file')

def createdatagenerators(batch_size,train_data_path,val_data_path,test_data_path):#creating data generators
    # for batch sizes its better to have it as a multiple of 2
    # batch size is the number of images to be loaded in one cycle
    train_preprocessor=ImageDataGenerator(rescale=1/255, rotation_range=10,width_shift_range=0.1)
    test_preprocessor=ImageDataGenerator(rescale=1/255)
    val_preprocessor=ImageDataGenerator(rescale=1/255)
    # it is necessary to use different preprocessors for testing, validation and training images
    traingenerator=train_preprocessor.flow_from_directory(train_data_path,class_mode='categorical',target_size=(60,60),color_mode='rgb',shuffle=True,batch_size=batch_size)
    testgenerator=test_preprocessor.flow_from_directory(test_data_path,class_mode='categorical',target_size=(60,60),color_mode='rgb',shuffle=False,batch_size=batch_size)
    validategenerator=val_preprocessor.flow_from_directory(val_data_path,class_mode='categorical',target_size=(60,60),color_mode='rgb',shuffle=False,batch_size=batch_size)
    #since we've used categorical there will be one hot encoding and hence during compilig we should use the categorical loss function 
    # since the images we are using are rgb hence the color mode should be rgb
    # shuffle= True means that the order of images is not sequential there is the random selection
    return traingenerator,testgenerator,validategenerator

