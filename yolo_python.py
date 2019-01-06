
# import required packages
import cv2
import argparse
import numpy as np
from threading import Thread

# handle command line arguments
ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required=True,
 #               help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


class VideoStream:
	
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		
	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):	
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()
	
	def read(self):
        # Return the latest frame
		return self.frame
	
	def stop(self):
		self.stopped = True
		
		

#funtion to get the output layer names
def get_output_layer(net):

	layers_names  = net.getLayerNames()
	output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return output_layers
	
#bounding box

def draw_box(img,class_id,confidence,x,y,w,h):
	label = str(classes[class_id])
	
	color = COLORS[class_id]
	
	cv2.rectangle(img,(x,y),(w,h),color,2)
	
	cv2.putText(img,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


vs = VideoStream(src=0).start()
frame_count = 0
while True:
	image = vs.read()
	frame_count+=1

	if frame_count%5 == 0:
		Width = image.shape[1]
		Height = image.shape[0]
		scale = 0.00392	
		# read class names from txt file
		classes = None

		with open(args.classes,'r') as f:
			classes = [line.strip() for line in f.readlines()]
			
		#giving diff colors to classes
		COLORS = np.random.uniform(0,255,size = (len(classes),3))

		#read pre-trained model and weights
		net = cv2.dnn.readNet(args.weights,args.config)

		#create input blob
		blob = cv2.dnn.blobFromImage(image,scale,(416,416),(0,0,0),True,crop=False)

		#set input blob for network

		net.setInput(blob)

		#output layer and bounding box


		#running inference
		#get predictions

		outputs = net.forward(get_output_layer(net))

		#initialize
		class_ids = []
		confidences = []
		boxes = []
		cnf_thres = 0.5
		nms_thres = 0.4

		for out in outputs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					c_x = int(detection[0] * Width)
					c_y = int(detection[1] * Height)
					w = int(detection[2] * Width)
					h = int(detection[3] * Height)
					x= c_x - w / 2
					y = c_y - h/2
					class_ids.append(class_id)
					confidences.append(float(confidence))
					boxes.append([x,y,w,h])
					
		#Non-ma Suppression
		#to detect and eliminate duplicate detection

		indices = cv2.dnn.NMSBoxes(boxes,confidences,cnf_thres,nms_thres)

		# go through the detections remaining
		# after nms and draw bounding box
		for i in indices:
			i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]
			
			draw_box(image,class_ids[i],confidences[i],round(x),round(y),round(x+w),round(y+h))

		#display output image

		cv2.imshow("object detection",image)
		
		#wait until key is pressed	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

#save output 
#cv2.imwrite("detected.jpg",image)
vs.stop()
#preparing input
#release resourses

cv2.destroyAllWindows()













