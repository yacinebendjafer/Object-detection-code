import cv2
from djitellopy import tello
import cvzone


thres = 0.6 #Threshold parameter
nmsThres = 0.2


# img = cv2.imread('TR.jpg')

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb" 

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


me = tello.Tello()
me.connect()
me.streamoff()
me.streamon()
while True:
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold= thres, nmsThreshold=nmsThres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(confidence * 100, 2)}',
                        (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0,255,0), 2)
            # cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            # cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            # cv2.putText(img,str(round(confidence*100,2)),(box[0]+80,box[1]+30),
            #             cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            

    me.send_rc_control(0,0,0,0)
    
    cv2.imshow("Output",img)
    cv2.waitKey(1)