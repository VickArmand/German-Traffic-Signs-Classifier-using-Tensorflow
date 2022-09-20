import tensorflow as tf
import numpy as np

def predict_with_model(model,imgpath):
    image=tf.io.read_file(imgpath)#loading and reading image
    image=tf.image.decode_png(image,channels=3)#decode image like we said before channels are 3 because of use of rgb images
    image=tf.image.convert_image_dtype(image,dtype=tf.float32)#rescale image or converting pixel values of image to flat and scaling them from 0 to 1 instead of from 0 to 255
    # produces an image tensor that has been normalized to values between 0 and 1
    image=tf.image.resize(image,(60,60))#we are using 60, 60 since we used it during model development(60,60,3)
    image= tf.expand_dims(image, axis=0)#expanding the dimensions to (1,60,60,3)
    #is used to insert an addition dimension in input Tensor.Parameters:input: It is the input Tensor, axis: It defines the index at which dimension should be inserted
    predictions=model.predict(image)#[0.005,0.00003,0.99,.....]
    predictions=np.argmax(predictions)
    return predictions
if __name__=="__main__":
    img_path="C:\\Users\\VICKFURY\\Documents\\projects\\Python Scripts\\ml\\ml codes\\supervised\\INTRODUCTION TO TENSORFLOW FOR COMPUTER VISION\\traffic signs\\Datasets\\Model Training Data\\Test\\18\\00006.png"
    model=tf.keras.models.load_model('./models')
    prediction=predict_with_model(model, img_path)
    print(f"Resulting prediction for the image is that it belongs to class : {prediction} " )