import keras
from keras.models import load_model

import cv2
path = '/home/dj/Desktop/Datasets/at_200x/val/milling@200x/milling5.jpg'


new = load_model('/home/dj/Desktop/Weight Files/inception_v3_0.97_0.86(200x).h5')
print('This model classifies the given image at an accuracy of 86%\nGive Input Image size as 256 X 256\n')
import numpy as np
from keras.preprocessing import image
# predicting images
img = image.load_img(path, target_size=(256,256,3))
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
