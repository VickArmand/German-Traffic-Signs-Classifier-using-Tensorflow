
from turtle import shape
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from tensorflow.keras import Model
def streetsignsmodel(numofclasses):
    my_input=Input(shape=(60,60,3))#shape parameter takes three values width,height,number of channels. Since most images in the dataset are rgb hence number of channels are 3

    x=Conv2D(32,(3,3),activation='relu')(my_input) #here we are creating 32 filters of 3 by 3 each
    x=Conv2D(64,(3,3),activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)
    
# layer 2
    x=Conv2D(128,(3,3),activation='relu')(x)
    x=MaxPool2D()(x)
    x=BatchNormalization()(x)
    
# layer 3
    x=Flatten()(x)
    #x=GlobalAvgPool2D()(x)# finds average of output values from BatchNormalization()
    x=Dense(64,activation='relu')(x)
    x=Dense(numofclasses,activation='softmax')(x)#output layer
    model=Model(inputs=my_input,outputs=x)
    return model

if __name__=='__main__':
    model=streetsignsmodel(10)
    model.summary()