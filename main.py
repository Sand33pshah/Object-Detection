import cv2


thres = 0.6   #threshold to detect the object

cap = cv2.VideoCapture(0)  #feed from my camera

classNames = []  #list of class names
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)   #this is the inbuilt function which will do all the processing for us after we pass the parameter.
net.setInputSize(320,320)         
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5)) 
net.setInputSwapRB(True)

while True:
    success, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)     
    print(classIds,bbox)
    if len(classIds)!=0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classId<=80:
                cv2.rectangle(frame, box, color =(255,255,), thickness=1)
                cv2.putText(frame, classNames[classId-1].capitalize(), (box[0]+10, box[1]), cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1)
    cv2.imshow('Object Detection Model',frame)
    if cv2.waitKey(1) == ord('x'):
        break
cap.release()
cv2.destroyAllWindows()