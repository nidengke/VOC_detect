#coding:utf8
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)


# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# fps = FPS().start()
#定义人,沙发标志位,
p=0
s=0
#设置人坐在沙发持续时间,单位(秒)
strict_time = 20
sc = []
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # label = "{}: {:.2f}%".format(CLASSES[idx],
            #                              confidence * 100)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            #               [0, 0, 255], 2)
            # cv2.putText(frame, label, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
            # if CLASSES[idx] == 'cat':
            #     p = 1
            #     cv2.rectangle(frame, (startX, startY), (endX, endY),
            #                   COLORS[idx], 2)
            #     pcx = (startX+endX)/2
            #     pcy = (startY+endY)/2
            #     y = startY - 15 if startY - 15 > 15 else startY + 15
            #     cv2.putText(frame, label, (startX, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # if CLASSES[idx] == 'sofa':
            #     s = 1
            #     cv2.rectangle(frame, (startX, startY), (endX, endY),
            #                   COLORS[idx], 2)
            #     y = startY - 15 if startY - 15 > 15 else startY + 15
            #     ssx = startX
            #     sex = endX
            #     ssy = startY
            #     sey = endY
            #     cv2.putText(frame, label, (startX, y),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            # if p == 1 and s == 1:
            #     if pcx>ssx and pcx<sex and pcy>ssy and pcy<sey:
            #         print("person in sofa")
            #         t1 = time.time()
            #         sc.append(t1)
            #         if sc[-1]-sc[0]>strict_time:
            #             print("please not always in the sofa,go up!")
            #             cv2.rectangle(frame, (startX, startY), (endX, endY),
            #                           [0,0,255], 2)
            #             cv2.putText(frame, label, (startX, y),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
            #
            #
            # else:
            #     print("People are not sedentary.")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



#python detct_cat.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --video cat1.mp4
