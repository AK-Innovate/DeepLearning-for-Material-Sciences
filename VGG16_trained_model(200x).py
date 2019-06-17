import keras
from keras.models import load_model

import cv2
#path = '/home/dj/Desktop/Datasets/at_200x/val/turning@200x/turning3.jpg'
#path = '/home/dj/Desktop/Datasets/at_200x/val/grinding@200x/grinding3.jpg'
path = '/home/dj/Desktop/Datasets/at_200x/val/milling@200x/milling3.jpg'
#path = '/home/dj/Desktop/Datasets/at_200x/val/buffing@200x/buffing3.jpg'
new = load_model('/home/dj/Desktop/Weight Files/VGG16_bgmt_1_0.96(200x)_224_224.h5')
print('This model classifies the given image at an accuracy of 96%\nGive Input Image size as 224 X 224\n')
import numpy as np
from keras.preprocessing import image
# predicting images
img = image.load_img(path, target_size=(224,224,3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])/255
classes = new.predict(images, batch_size=10)
print(classes)
categ = ['buffing','grinding','milling','turning']
print(categ[np.argmax(classes)])

img = cv2.imread(path,1)
cv2.imshow(categ[np.argmax(classes)],img)
cv2.waitKey(0)
