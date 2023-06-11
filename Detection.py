import cv2
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('best_model.h5')

def prediction_result(model,pic):
    labels = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
              8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',
              17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
    label = np.argmax(model.predict(pic,verbose =0)[0])
    return labels[label]

cam = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame

    ret, frame = cam.read()
    #frame size
    img = frame[100:500,300:800]   
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28),interpolation = cv2.INTER_AREA)
    
    copy = frame.copy()
    #detection area
    cv2.rectangle(copy,(300,100),(550,350),(255,0,0),5)
    
    img = img.reshape(1,28,28,1)
    result = prediction_result(model,img)
    cv2.putText(copy, result, (290,90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                  (0,0,255), 2, cv2.LINE_AA) 
    cv2.imshow('frame',copy)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
    