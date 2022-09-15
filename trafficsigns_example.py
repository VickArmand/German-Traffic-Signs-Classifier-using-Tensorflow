# Dataset used is the GTSRB - German Traffic Sign Recognition Benchmark find it on kaggle
import os
import tensorflow as tf
from pickletools import optimize
from posixpath import split
from myutils import split_data,order_test_set,createdatagenerators
from neuralnetwork import streetsignsmodel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
if __name__=='__main__':#allows or prevent parts of code from being run when the modules are imported. When the Python interpreter reads a file, the __name__ variable is set as __main__ if the module being run, or as the module's name if it is imported.
    # path_to_data = "C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Train"
    path_to_train="C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Model Training Data\\Train"
    path_to_val="C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\Datasets\\Model Training Data\\Validation"
    # split_data(path_to_data,path_to_train=path_to_train,path_to_val=path_to_val)
    path_to_images='C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Test'
    path_to_csv_file='C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Test.csv'
    path_to_new_test_images='C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Model Training Data\\Test'
    # batch size is the number of images to be loaded in one cycle
     #saving the best model at a specific epoch depending on the accuracy at an epoch we use callbacks
    path_to_save_model='./models'
    checkpointsaver=ModelCheckpoint(path_to_save_model,monitor="val_accuracy",mode='max',save_best_only=True,save_freq='epoch',verbose=1)#reason for these parameters is to enable saving the model on highest validation accuracy on an epoch
    # EarlyStopping is another callback used whereby according to the number of epochs if the model doesnt improve the training stops. It is useful when one has multiple number of epochs
    early_stop=EarlyStopping(monitor="val_loss",patience=10)#this means if after 10 epochs the model is not improving(ie:validation loss is not reducing) then the training stops 
    order_test_set(path_to_images,path_to_csv_file,path_to_new_test_images)
    batch_size=64
    traingenerator,testgenerator,validategenerator=createdatagenerators(batch_size,path_to_train,path_to_val,path_to_new_test_images)
    nbr_classes=traingenerator.num_classes
    model=streetsignsmodel(nbr_classes)
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True)
    # it is necessary to use variables to store parameter values and later assign them to the method as shown below with the optimzer value in the compile()
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    # callbacks have to be passed into the fitting function
    model.fit(traingenerator,epochs=3,batch_size=batch_size,validation_data=validategenerator,callbacks=[checkpointsaver,early_stop])
    # model loading and evaluation
    model=tf.keras.models.load_model('./models')
    print("Evaluating validation data:")
    model.evaluate(validategenerator)
    print("Evaluating test data:")
    model.evaluate(testgenerator)
    # metrics improvement
    # One can improve metrics in the following ways:
        # changing the batch size and number of epochs
        # modify you number of layers
        # change the algorithm
        # change the number of filters
        # data augmentation techniques that minimize overfitting