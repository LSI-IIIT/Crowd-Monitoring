# USAGE
# python multi_object_tracking.py --video videos/soccer_01.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
from PIL import Image, ImageDraw
import face_recognition
import os
import numpy




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

def recognize_faces(frame):
	face_encodings = face_recognition.face_encodings(frame)
	face_locations = face_recognition.face_locations(frame)

	# print(len(face_encodings))
	if len(face_encodings) > 0:
		for file in os.listdir("known_identities/"):
			kn_en = []
			un_im = face_recognition.load_image_file("known_identities/"+file)
			face_encodings_tmp = face_recognition.face_encodings(un_im)
			kn_en.extend(face_encodings_tmp)

			print("Number of total known instances of the face: ",len(kn_en))

			match_all = []

			for face_location, face_encoding in zip(face_locations, face_encodings):
				top, right, bottom, left = face_location
				match = face_recognition.face_distance(kn_en, face_encoding)
				match_all.append(max(match))

			array = numpy.array(match_all)
			temp = array.argsort()
			ranks = numpy.empty_like(temp)
			ranks[temp] = numpy.arange(len(array))

			for face_location, rank in zip(face_locations, ranks):
				top, right, bottom, left = face_location
				if rank == 0 :
					box = (left, top, right - left, bottom - top)
					print(box)
					tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
					trackers.add(tracker, frame, box)
	else:
		print("No faces found!")

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# print(frame)

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)

	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(20) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		print(box)
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, box)

	elif key == ord("r"):
	# Recognize the faces
		recognize_faces(frame)

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
