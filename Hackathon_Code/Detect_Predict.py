
import os
import copy
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


### classification model


from keras.models import load_model
 
# load model
sentiment = load_model('face_resnet_model.h5')
stack = load_model('multi_stack.h5')





# mappings for predictions 
age_mapping = 116
ethnicity_mapping = np.array(["White", "Black", "Asian", "Indian", "Hispanic"])
gender_mapping = np.array(["Male", "Female"])



############################################################################################################
class FaceDetector():

    def __init__(self,faceCascadePath):
        self.faceCascade=cv2.CascadeClassifier(faceCascadePath)


    def detect(self, image, scaleFactor=1.25,
               minNeighbors=20,
               minSize=(50,50)):
        
        #function return rectangle coordinates of faces for given image
        rects=self.faceCascade.detectMultiScale(image,
                                                scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors,
                                                minSize=minSize)
        return rects
    

# detect faces

def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces=fd.detect(image_gray,
                   scaleFactor=scaleFactor,
                   minNeighbors=minNeighbors,
                   minSize=minSize)

    for x, y, w, h in faces:
        
        ## detect sentiment with model 1
        fc1 = image[y:y+h, x:x+w]
        
        ## face sentiment calssification
        roi1 = cv2.resize(fc1, (56,56)) # image size is 56x56
        roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2RGB)
        roi1 = roi1.reshape(-1, 56, 56, 3)
        pred1 = sentiment.predict(roi1)#[np.newaxis, np.newaxis,3])
        text_idx1=np.argmax(pred1)
        text_list1 = ['Angry', 'Happy', 'Neutral']
        if text_idx1 == 0:
            text1= text_list1[0]
        if text_idx1 == 1:
            text1= text_list1[1]
        elif text_idx1 == 2:
            text1= text_list1[2]
      
        cv2.putText(image, text1, (x, y-5),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
           
          
          
        ## three stack predictions
        fc2 = image_gray[y:y+h,x:x+w]
        
        roi2 = cv2.resize(fc2, (48,48))
        
        #pred2 = stack.predict(roi2[np.newaxis, :, :, np.newaxis])
        #predictions = stack.predict(np.expand_dims(test_X[0],axis = 0))
        predictions = stack.predict(roi2[np.newaxis, :, :, np.newaxis])
        #print((roi2[np.newaxis,:,:,np.newaxis]).shape)
        age_pred = predictions[0]*116#age_mapping
        age_predictions = [int(np.round(ages)) for ages in age_pred]

        for pred in predictions[1]:
            eth_index = np.where(pred == np.amax(predictions[1]))
        eth_predictions = ethnicity_mapping[eth_index]
        
        for pred in predictions[2]:
            gen_index = np.where(pred == np.amax(predictions[2]))
        gen_predictions = gender_mapping[gen_index]


        # cv2.putText(image, str(age_predictions[0]), (x, y+100),
           # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
        cv2.putText(image, str(eth_predictions[0]), (x, y+25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
        cv2.putText(image,str(gen_predictions[0]), (x, y+50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
           
        #detected faces shown in color image
        cv2.rectangle(image,(x,y),(x+w, y+h),(127, 255,0),3)

       




#Frontal face of haar cascade loaded
frontal_cascade_path= cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml'

#Detector object created
fd=FaceDetector(frontal_cascade_path)





# ## capturing image

# cap = cv2.VideoCapture(1)


# while True:
    
    # ret, frame = cap.read()
    # img = copy.deepcopy(frame)
    
    # detect_face(img, 
                # scaleFactor = 1.1,
                # minNeighbors = 3,
                # minSize = (50,50))
    
    # cv2.imshow("frame", img)
    # key = cv2.waitKey(1) & 0xFF
    # if key== ord('q'):
        # break
    
# cap.release()
# cv2.destroyAllWindows()


# import sys

# image_path = sys.argv[1]

# img = cv2.imread(image_path)

# img = copy.deepcopy(img)
    
# detect_face(img, 
                # scaleFactor = 1.1,
                # minNeighbors = 3,
                # minSize = (50,50))
    
# cv2.imshow("frame", img)


import sys
import copy
image_path = sys.argv[1]
save_name = sys.argv[2]
# image = cv2.imread('man.png')
#image = cv2.imread('image1.jpeg')
img = cv2.imread(image_path)
img = copy.deepcopy(img)
    
detect_face(img, 
                scaleFactor = 1.1,
                minNeighbors = 3,
                minSize = (50,50))
    
cv2.imshow("frame", img)
cv2.imwrite(save_name,img)
key = cv2.waitKey(20000) & 0xFF
if key== ord('q'):
    cv2.destroyAllWindows()
cv2.destroyAllWindows()


