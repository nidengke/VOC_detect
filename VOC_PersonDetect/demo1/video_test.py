# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from matplotlib import pyplot as plt
import argparse
import imutils
import time
import cv2
from PIL import Image
from pylab import imshow
from pylab import array
from pylab import plot
from pylab import title

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

# loop over the frames from the video stream
resultArray = np.array([[0],[0]])
im = array(Image.open('dogTemplate.jpg'))
i=0
try:
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

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                if CLASSES[idx] == 'dog':
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    pcx = (startX+endX)/2
                    pcy = (startY+endY)/2
                    i = i+1
                    if(i>100):
                        raise RuntimeError('testError')
                        print(i)
                        fh = open("testfile", "w")
                        fh.write("weqewq!!")
                    temp = np.array([[pcx],[pcy]])
                    resultArray = np.hstack((resultArray, temp))
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                # cv2.rectangle(frame, (startX, startY), (endX, endY),
                #               COLORS[idx], 2)
                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(frame, label, (startX, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                print(resultArray.shape)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
except :
    x = resultArray[0]
    y = resultArray[1]
    N = len(x)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N)) ** 2
    plt.scatter(x, y, s=area, c=colors, alpha=0.5, marker=(9, 3, 30))
    plt.show()

# stop the timer and display FPS information
x = resultArray[0]
y = resultArray[1]
N=len(x)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N)) ** 2
plt.scatter(x, y, s=area, c=colors, alpha=0.5, marker=(9, 3, 30))
plt.show()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# update the FPS counter
fps.update()

#python video_test.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --video dog7.flv
