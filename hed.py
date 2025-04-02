import math
import sys
import numpy as np
import cv2

#input
img = cv2.imread("sample.png")
(H, W) = img.shape[:2]
print('Height:',H)
print('Width:',W)

# The pre-trained model that OpenCV uses has been trained in Caffe framework
protoPath = "deploy.prototxt"
modelPath = "hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# construct a blob out of the input image
#blob is basically preprocessed image.
#OpenCV’s new deep neural network (dnn ) module contains two functions that
#can be used for preprocessing images and preparing them for
#classification via pre-trained deep learning models.
# It includes scaling and mean subtraction
#How to calculate the mean?
mean_pixel_values= np.average(img, axis = (0,1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             #mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             mean=(105, 117, 123),
                             swapRB= False, crop=False)

#View image after preprocessing (blob)
blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)


# set the blob as the input to the network and perform a forward pass
# to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0,0,:,:]  #Drop the other axes
#hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")  #rescale to 0-255

cv2.imwrite("DLEdgedetection_output.png",hed)