"""trmet_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import sys 
from utils_deepsort.parser import get_config
from utils.yolo_with_plugins import Tracker_tiny
from utils_deepsort.draw import draw_boxes
from collections import deque
import os
import time
import argparse
import numpy as np 
import cv2
import torch
from ultralytics import YOLO
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.project_lanedetection import *


WINDOW_NAME = 'ProjectDemo'


class OpenCVYolo:
	"""Simple OpenCV DNN wrapper for YOLO Darknet models.
	Provides a .detect(img, conf_th) method that returns boxes, scores, classes
	with the same shape/semantics as the TensorRT TrtYOLO.detect used by this repo.
	"""
	def __init__(self, cfg_path, weights_path, input_shape=(416,416)):
		self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
		# prefer CPU
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		self.input_shape = input_shape

	def detect(self, img, conf_th=0.3, letter_box=False):
		h, w = img.shape[:2]
		inp_w, inp_h = self.input_shape[1], self.input_shape[0]
		blob = cv2.dnn.blobFromImage(img, 1/255.0, (inp_w, inp_h), swapRB=True, crop=False)
		self.net.setInput(blob)
		layer_names = self.net.getLayerNames()
		out_names = [layer_names[i[0]-1] if isinstance(i, (list, tuple, np.ndarray)) else layer_names[i-1]
					 for i in self.net.getUnconnectedOutLayers()]
		outs = self.net.forward(out_names)

		class_ids = []
		confidences = []
		boxes = []

		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = int(np.argmax(scores))
				confidence = float(scores[class_id] * detection[4])
				if confidence > conf_th:
					center_x = int(detection[0] * w)
					center_y = int(detection[1] * h)
					bw = int(detection[2] * w)
					bh = int(detection[3] * h)
					x1 = int(center_x - bw / 2)
					y1 = int(center_y - bh / 2)
					x2 = x1 + bw
					y2 = y1 + bh
					boxes.append([x1, y1, x2, y2])
					confidences.append(confidence)
					class_ids.append(class_id)

		# NMS
		if len(boxes) > 0:
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_th, 0.5)
			filtered_boxes = []
			filtered_scores = []
			filtered_classes = []
			if isinstance(idxs, (list, tuple)):
				idxs = idxs
			else:
				try:
					idxs = idxs.flatten()
				except Exception:
					idxs = [int(i) for i in idxs]
			for i in idxs:
				i = int(i)
				filtered_boxes.append(boxes[i])
				filtered_scores.append(confidences[i])
				filtered_classes.append(class_ids[i])
			boxes = np.array(filtered_boxes, dtype=np.int32)
			scores = np.array(filtered_scores, dtype=np.float32)
			classes = np.array(filtered_classes, dtype=np.int32)
		else:
			boxes = np.zeros((0,4), dtype=np.int32)
			scores = np.zeros((0,), dtype=np.float32)
			classes = np.zeros((0,), dtype=np.int32)

		return boxes, scores, classes


class PyTorchYolo:
	"""PyTorch YOLO wrapper for .pt models (YOLOv5/v8 via ultralytics).
	Provides a .detect(img, conf_th) method that returns boxes, scores, classes
	with the same shape/semantics as the other detectors.
	"""
	def __init__(self, model_path, device='auto'):
		if device == 'auto':
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = YOLO(model_path)
		self.device = device
		self.model.to(self.device)

	def detect(self, img, conf_th=0.3, letter_box=False):
		# Inference with ultralytics
		results = self.model(img, conf=conf_th, verbose=False)

		# Parse YOLOv8 results
		detections = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

		boxes = []
		scores = []
		classes = []

		for det in detections:
			if len(det) >= 6:
				x1, y1, x2, y2, conf, cls = det
				boxes.append([int(x1), int(y1), int(x2), int(y2)])
				scores.append(float(conf))
				classes.append(int(cls))

		if len(boxes) > 0:
			boxes = np.array(boxes, dtype=np.int32)
			scores = np.array(scores, dtype=np.float32)
			classes = np.array(classes, dtype=np.int32)
		else:
			boxes = np.zeros((0,4), dtype=np.int32)
			scores = np.zeros((0,), dtype=np.float32)
			classes = np.zeros((0,), dtype=np.int32)

		return boxes, scores, classes


def parse_args():
	"""Parse input arguments."""
	desc = ('Capture and display live camera video, while doing '
					'real-time object detection with TensorRT optimized '
					'YOLO model on Jetson')
	parser = argparse.ArgumentParser(description=desc)
	parser = add_camera_args(parser)
	parser.add_argument(
			'-c', '--category_num', type=int, default=80,
			help='number of object categories [80]')
	parser.add_argument(
			'-m', '--model', type=str, required=False,
			help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
						'[{dimension}], where dimension could be a single '
						'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
	parser.add_argument(
			'-l', '--letter_box', action='store_true',
			help='inference with letterboxed image [False]')
	#############add deepsort yaml
	parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
	parser.add_argument('--use_opencv', action='store_true', help='Use OpenCV DNN fallback instead of TensorRT')
	parser.add_argument('--use_pytorch', action='store_true', help='Use PyTorch YOLO model (.pt file)')
	parser.add_argument('--yolo_cfg', type=str, default='./yolo/yolov4-tiny.cfg', help='Path to YOLO cfg for OpenCV fallback')
	parser.add_argument('--yolo_weights', type=str, default='./yolo/yolov4-tiny.weights', help='Path to YOLO weights for OpenCV fallback')
	parser.add_argument('--pytorch_model', type=str, default='./yolo/best.pt', help='Path to PyTorch YOLO model (.pt)')
	parser.add_argument('--unsafe_dist', type=float, default=8.0, help='Distance (m) threshold to mark Unsafe')
	parser.add_argument('--danger_dist', type=float, default=4.0, help='Distance (m) threshold to mark Dangerous')
	parser.add_argument('--no_deepsort', action='store_true', help='Disable DeepSORT tracking')
	parser.add_argument('--skip_frames', type=int, default=1, help='Process every Nth frame (e.g., 2 for every 2nd frame)')
	parser.add_argument('--input_width', type=int, default=1280, help='Input image width')
	parser.add_argument('--input_height', type=int, default=960, help='Input image height')
	parser.add_argument('--enable_trajectory_extrapolation', action='store_true', help='Enable trajectory extrapolation for proactive alerts')
	parser.add_argument('--enable_ttc_forecasting', action='store_true', help='Enable enhanced TTC forecasting')
	parser.add_argument('--enable_optical_flow', action='store_true', help='Enable optical flow motion analysis')
	parser.add_argument('--enable_proactive_thresholds', action='store_true', help='Enable proactive alert thresholds based on context')
	parser.add_argument('--disable_lane_detection', action='store_true', help='Disable lane detection for speedup')
	parser.add_argument('--confidence_threshold', type=float, default=0.55, help='Confidence threshold for detection')
	#######################    
	args = parser.parse_args()
	return args

def append_speed(ids,deque_list):
	speed_list = []
	for j in range(0 , len(deque_list[ids]) ):
		speed_list.append((deque_list[ids][j]))
	if len(deque_list[ids])>10:
		spd_avg = np.average(speed_list,axis=0)
		return spd_avg
	else:
		return "still appending"

#fix bbox issues
def compute_xc_yc(out):
	w = out[:,[2]] - out[:,[0]]
	h = out[:,[3]] - out[:,[1]]
	xmin = out[:,[0]]
	ymin = out[:,[1]]
	xc = w/2 + xmin
	yc = h/2 + ymin 
	return xc,yc,w,h
		
def draw (pos,img):
	for poss in pos :
		cv2.circle(img, poss, 4, (0, 255,255), -1)
		cv2.polylines(img,[np.int32(pos)], False, (0,255,255), 1)
		
				
def ez_show(img):
	img0 = np.zeros_like(img)
	cv2.line(img0,(1000,960),(586,570),(255,255,0),3)  
	cv2.line(img0,(586,570),(500,570),(255,255,0),4)
	pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
	cv2.fillPoly(img0,pol, (0,255,0))
	return img0
		
def Distance_finder(real_width, face_width_in_frame):	
	Focal_Length = 958
	distance = (real_width * Focal_Length)/face_width_in_frame
	return distance    

def motion_cord(starting_points,line_parameters):
		slope, intercept = line_parameters
		x1 , y1 = starting_points
		y2 = y1 + 100
		#y2 = y1 + 30 #extended line
		x2 = int((y2-intercept)/(slope))
		return x1, y1, x2, y2
		
def trajectory_extrapolation(outputs, img_shape, safety_zone):
	"""Extrapolate trajectories and check for future hazards."""
	alert = False
	for x1, y1, x2, y2, ids in outputs:
		# Simple extrapolation: assume constant velocity from recent positions
		# For demo, predict 2 seconds ahead (assuming 20 FPS, 40 frames)
		pred_x = x1 + (x2 - x1) * 2  # rough velocity
		pred_y = y1 + (y2 - y1) * 2
		# Check if predicted position is in safety zone (simplified)
		if pred_y > img_shape[0] * 0.8:  # bottom 20% as safety
			alert = True
			break
	return alert

def enhanced_ttc_forecasting(dis, speed, threshold=2.0):
	"""Forecast TTC and alert if predicted < threshold seconds."""
	if speed > 0:
		ttc = dis / (speed * 1000 / 3600)  # convert to seconds
		if ttc < threshold:
			return True
	return False

def optical_flow_analysis(prev_gray, gray, bbox):
	"""Compute optical flow for motion vectors."""
	flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	# Average flow in bbox
	x1, y1, x2, y2 = bbox
	mean_flow = np.mean(flow[y1:y2, x1:x2], axis=(0,1))
	magnitude = np.linalg.norm(mean_flow)
	direction = np.arctan2(mean_flow[1], mean_flow[0])  # towards camera if positive y
	return magnitude > 1.0 and direction > 0  # simple threshold

def proactive_alert_thresholds(dis, speed, in_lane, motion_predict):
	"""Adjust thresholds based on context."""
	if in_lane and motion_predict:
		danger_dist = 6.0  # lower threshold
		unsafe_dist = 12.0
	else:
		danger_dist = 4.0
		unsafe_dist = 8.0
	return dis <= danger_dist or (dis <= unsafe_dist and speed > 10)
		
def loop_and_detect(cam, detector, tracker, conf_th, vis, args=None):
	"""Continuously capture images from camera and do object detection.

	# Arguments
		cam: the camera instance (video source).
		trt_yolo: the TRT YOLO object detector instance.
		conf_th: confidence/score threshold for object detection.
		vis: for visualization.
	"""
	# global img_final
	full_scrn = False
	fps = 0.0
	tic = time.time()
	start_time = time.time()  # For total processing time
	f = [] 
	m = []
	n = 0
	cls = ""
	framenumber = -1
	speed = ""
	k = 0
	#tic = 0
	time_start = time_end = 0
	dis_start = dis_end = 0
	#create deque container
	pts = [deque(maxlen=30) for _ in range(100)]
	pt = [deque(maxlen=50) for _ in range(100)]
	#h_ls = [deque(maxlen=30) for _ in range(100)]
	w_list = [deque(maxlen=30) for _ in range(100)]
	car_spd = [deque(maxlen=30) for _ in range(50)]
	moto_spd = [deque(maxlen=30) for _ in range(50)]
	history = {}  # for drawing trajectory paths
	unsafe_v = False
	danger_v = False
	used = False
	# previous status to avoid spamming logs every frame
	prev_status = None
	lanedetection = not args.disable_lane_detection  # Disable for speedup if flagged
	puttext_deer = False
	puttext_moto = False
	bad = False
	motion_predict = False
	drw = False
	unsafe_v = False
	danger_v = False
	avg_spd_moto = "still appending"
	avg_spd_deer = "still appending"
	deer_speed = 0
	x_dir = []
	y_dir = []
	prev_gray = None  # For optical flow
	
	# Create incremental output folder
	import os
	if not os.path.exists('output'):
		os.makedirs('output')
	existing_runs = [d for d in os.listdir('output') if d.startswith('run_') and os.path.isdir(os.path.join('output', d))]
	if existing_runs:
		run_nums = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
		next_num = max(run_nums) + 1 if run_nums else 1
	else:
		next_num = 1
	output_dir = f'output/run_{next_num:03d}'
	os.makedirs(output_dir)
	
	#save output video
	#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5) 
	#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
	#size = (width, height)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
	#out = cv2.VideoWriter('output_testingvid03.avi', fourcc, 10.0, (640,  480))
	#out2 = cv2.VideoWriter('combo.avi', fourcc, 20.0, (1280,960))
	out = cv2.VideoWriter(os.path.join(output_dir, 'line_vis.mp4'), fourcc, 20.0, (args.input_width, args.input_height))
	out1 = cv2.VideoWriter(os.path.join(output_dir, 'final_res.mp4'), fourcc, 20.0, (args.input_width, args.input_height))
	out2 = cv2.VideoWriter(os.path.join(output_dir, 'combo.mp4'), fourcc, 20.0, (args.input_width, args.input_height))
	#out1 = cv2.VideoWriter('deepsort_out4.avi', fourcc, 20.0, (1280,  960))
	##
	while True:
		framenumber+=1
		if framenumber % args.skip_frames != 0:
			continue  # Skip this frame for speedup
		#mylcd = I2C_LCD_driver.lcd()
		#if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
			#  break
		img = cam.read()
		if img is None:
			break
		img = cv2.resize(img, (args.input_width, args.input_height))
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # For optical flow
		tim = framenumber/20 
		#cv2.putText(img_better_look, f"time {tim}s",  (1100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)  #bgr 
		pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
		img = img.astype('uint8')
		original_image = img
		img_better_look = img
		cv2.putText(img_better_look, f"time {tim}s",  (1100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)  #bgr 
		#input_cropped = frame[550:(550+IMAGE_H), 0:IMAGE_W]
		add_trans = np.zeros_like(img)
		#add_trans = add_trans[:,:,0] #force one channel 
		#img_trans = perspective_transformation(img)
		#img_trans = select_yellow_white(img_trans)
		#img_trans = canny(img_trans)
		
		'''
		lanedetection
		==============
		filtering out not interested region 
		'''
		#yellow_white = select_yellow_white(img)
		#cannyresult = canny(yellow_white)
		#frame_for_dis = draw_dis_lines(frame)
		cannyresult = canny(img)
		#get the right vertice automatically
		vertice = get_vetices()
		cropped_image , mask = region_of_interest2(cannyresult,vertice)
		lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)  #minLineLength=40, maxLineGap=5
		#print("lines\n",lines)
		if lines is not None :
			lines = np.reshape(lines, [len(lines),4]) #lines will be None sometimes
			#avg_lines = average_slope_intercept(frame,lines)
			avg_lane, left , right = average_slope_intercept(img,lines)
			#print(len(avg_lane)) #if len(avg_lane)==1 ->only left or right if len(avg_lane)==2 ->both left and right
			#fix road disappear issue ->works well
			if len(avg_lane)==2:
				left_avg_lines = avg_lane[[0]]
				right_avg_lines = avg_lane[[1]]
				for x1 , y1 , x2 , y2 in left_avg_lines :
					xl1 , yl1 ,xl2 ,yl2 = x1 , y1 , x2 , y2  

				for x1 , y1 , x2 , y2 in right_avg_lines :
					xr1 , yr1 ,xr2 ,yr2 = x1 , y1 , x2 , y2

			elif left == True:
				for x1 , y1 , x2 , y2 in avg_lane:
					xl1 , yl1 ,xl2 ,yl2 = x1 , y1 , x2 , y2

			elif right == True:
				for x1 , y1 , x2 , y2 in avg_lane:
					xr1 , yr1 ,xr2 ,yr2 = x1 , y1 , x2 , y2

			try:   
				#vertices_polly = np.array([[(xl1, yl1), (xl2, yl2), (xr2, yr2), (xr1, yr1)]], dtype=np.int32)
				if xr1 - xl1 < 900:
					xr1 = xl1 + 1100
				if (xr2+5)-(xl2-5) < 130:
					xl2 = xr2 + 10 +150
				if (xr2+5) < (xl2-5) :
					xl2 , xr2 = xr2 + 10 , xl2 - 10 
				vertices_polly = np.array([[(xl1, yl1), (xl2-5, yl2-80), (xr2+5, yr2-80), (xr1, yr1)]], dtype=np.int32) #extend trapezoid
				vertices_polly_unextd = np.array([[(xl1, yl1), (xl2, yl2), (xr2, yr2), (xr1, yr1)]], dtype=np.int32) #unextend trapezoid
			except (NameError,OverflowError):
				print("xl1 is not defined ->only one side of line works")
				
		else:
			print("default avg_lines(not detecting lanes)")
			avg_lane = np.array([[0 ,572 ,479 ,205],   #0 572 ; 479  205 ; 641 193 ; 1268 481
                                 [1268 ,481 ,641 ,193]])
			vertices_polly = None
		
		img_zero = np.zeros_like(img)
		img0 =np.zeros_like(img) 
		color_polly =  (0,255,0) #BGR
		line_image_not_avg = draw_lines1(img0, lines)
		line_image = draw_lines2(img, avg_lane)
		line_visualize = cv2.addWeighted(img_better_look,1,line_image,1,1)
		#line_visualize = cv2.addWeighted(line_image_not_avg,1,line_visualize,1,1)
		god = filterout2(img,vertice,mask)
		#print(vertices_polly)
		try:
			#cv2.fillPoly(line_image, vertices_polly, color_polly)
			cv2.fillPoly(img_zero, vertices_polly_unextd, (0,255,0))
			filtered = filterout(img,vertices_polly)
			#print("vertices polly :\n",vertices_polly)
			if lanedetection == True:
					
				img=filtered #img = filtered
				#img = god
				#img = img   #not filtering
			#god = filterout2(frame,mask)
		except NameError:
			#filtered = original_image 
			filtered = god
			print("vertices polly is not defined")
			
		normal_result = cv2.addWeighted(img,1,img_zero,1,1)
		#print("vertices polly :",vertices_polly)
		combo_image = cv2.addWeighted(img, 1, line_image, 1, 1)
		#combo_image = cv2.addWeighted(combo_image, 1, line_image_not_avg, 1, 1) #addWeighted function cant add two srcs
		img_notavg = cv2.addWeighted(img, 1, line_image_not_avg, 1, 1)
															
		#allowing safety zone to draw on ->or the color of safety zone will be too dark        
		im0 = np.zeros_like(img)      
		if unsafe_v == False and danger_v == False:
			pass  # Banner will show SAFE
			
			#img0 = np.zeros_like(img_better_look)
			cv2.fillPoly(im0,pol, (0,255,0))
			#img_better_look = cv2.addWeighted(img0,0.7,img_better_look,1,1)
					
			#speed estimate zone
			#cv2.line(img_better_look,(50,530),(1260,530),(0,127,255),3)
			#cv2.line(img_better_look,(50,560),(1260,560),(0,127,255),3)
		
		'''
		yolov4 + Tensorrt
		
		'''

		# boxes, confs, clss = detector.detect(img, conf_th) ## for region

		boxes, confs, clss = detector.detect(original_image, conf_th)
		# Apply additional NMS to avoid duplicate bounding boxes
		if len(boxes) > 0:
			idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), 0.55, 0.4)
			filtered_boxes = []
			filtered_confs = []
			filtered_clss = []
			for i in idxs.flatten():
				filtered_boxes.append(boxes[i])
				filtered_confs.append(confs[i])
				filtered_clss.append(clss[i])
			boxes = np.array(filtered_boxes)
			confs = np.array(filtered_confs)
			clss = np.array(filtered_clss)

		if args.no_deepsort:
			# Draw raw detections without tracking
			img_better_look = vis.draw_bboxes(img_better_look, boxes, confs, clss)
			outputs = np.zeros((0, 5), dtype=np.float32)  # Empty outputs to skip tracked drawing
		else:
			# Use DeepSORT tracking
			# compute width and height of bboxs
			output = boxes
			w = output[:,[2]] - output[:,[0]]
			h = output[:,[3]] - output[:,[1]]
			xc , yc , w , h = compute_xc_yc(output)
			#print(xc,yc,"center")
			boxes = np.concatenate((xc,yc,w,h),axis=1)
			outputs = tracker.run(img, boxes, confs)

		# Predictive checks after outputs is defined
		if args.enable_trajectory_extrapolation and len(outputs) > 0:
			if trajectory_extrapolation(outputs, img.shape, pol):
				cv2.putText(img_better_look, "Trajectory Alert!", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
		
		if prev_gray is not None and args.enable_optical_flow and len(outputs) > 0:
			for x1,y1,x2,y2,ids in outputs:
				if optical_flow_analysis(prev_gray, gray, (int(x1),int(y1),int(x2),int(y2))):
					cv2.putText(img_better_look, "Optical Flow Alert!", (50, 250), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)
					break
		
		prev_gray = gray

		if args.no_deepsort:
			# Draw paths for raw detections
			for i, (x1,y1,x2,y2) in enumerate(boxes):
				xc = int((x1 + x2)/2)
				yc = int((y1 + y2)/2)
				ids = i  # dummy id per detection
				if ids not in history:
					history[ids] = deque(maxlen=50)
				history[ids].append((xc, yc))
				if len(history[ids]) > 1:
					points = list(history[ids])
					for j in range(1, len(points)):
						cv2.line(img_better_look, points[j-1], points[j], (0,255,0), 2)
				# Draw ID for raw detection
				cv2.putText(img_better_look, f"ID: {i}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

		#print('boxes_changed\n',boxes,'confs\n',confs,'clss',clss,"\n############")
		#print("         deepsort bboxs:            \n ",outputs)
		#for tensorrt_yolo
		#img_better_look = vis.draw_bboxes(img_better_look, output, confs, clss) 
		if len(clss)==1:
			clss = int(clss)
			cls = vis.cls_dict.get(clss)
		else:
			cls = ""
			#print("class      :",cls)
		#print("the type of class :\n",type(clss),"class :",clss)
		#f = []
		
		
		'''
		safety zone geometry setting
		
		'''
		
		
		if len(outputs) > 0 :
			#print(xc,"xc")
			#outputs.astype(int)
			#print("before x1 y1......",outputs)
			for x1,y1,x2,y2,ids in outputs:
				# Draw bounding box
				cv2.rectangle(img_better_look, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
				# Draw ID
				cv2.putText(img_better_look, f"ID: {int(ids)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
				xmin = x1
				ymin = y2
				w = x2 - x1 #w
				h = y2 - y1 #h
				xc = w/2 + x1 #xmin = x1
				yc = h/2 + y1 #ymin = y2
				xc = int(xc)
				yc = int(yc)
				w = int(w)
				h = int(h)
				low_mid = (xc,y2)
				low_left = (x1,y2)
				center = (xc,yc,h)
				#cent = (xc,yc)
				w_tim = (w,tim,cls)
				#xc = np.array(xc[0,0],dtype = np.int32) 
				#yc = np.array(yc[0,0],dtype = np.int32)
				x_res_yc = 1.062*yc - 20 
				x_res_y = 1.062*y2 - 20
				pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
				#pts[ids].append(center)
				print("id",ids,"w_tim",w_tim)
				pt[ids].append(low_left)
				#h_ls[ids].append(h)
				w_list[ids].append(w_tim)
				#print("pt:\n",pt,"\n")
				print("w_list:\n",w_list,"\n")
				
				# Add to history for path drawing
				if ids not in history:
					history[ids] = deque(maxlen=50)
				history[ids].append((xc, yc))
				# Draw the trajectory path
				if len(history[ids]) > 1:
					points = list(history[ids])
					for i in range(1, len(points)):
						cv2.line(img_better_look, points[i-1], points[i], (0,255,0), 2)
				# Draw predicted path arrow
				if not args.no_deepsort and hasattr(tracker, 'tracks'):
					for track in tracker.tracks:
						if track.track_id == ids and track.is_confirmed():
							kf = track.kalman_filter
							vel_x = kf.x[4]
							vel_y = kf.x[5]
							vel_mag = (vel_x**2 + vel_y**2)**0.5
							if vel_mag > 0:
								dir_x = vel_x / vel_mag
								dir_y = vel_y / vel_mag
								arrow_length = 50  # fixed length for visibility
								pred_center = (xc + dir_x * arrow_length, yc + dir_y * arrow_length)
								cv2.arrowedLine(img_better_look, (xc, yc), tuple(map(int, pred_center)), (0,255,255), 3, tipLength=0.3)
							break
				
				#print("the ids now :",ids,"\n")
				for j in range(0, len(pt[ids])): #start with 1
					#cent = (pts[ids][j][0] , pts[ids][j][1])      
					#cent = (pts[ids][j-1] , pts[ids][j])
					
					
					#cv2.line(img_better_look,(pt[ids][j-1]) , (pt[ids][j]),(0,255,255),3)
					#greatest > curr
					#print("len(pt[ids])",len(pt[ids]))
					if abs(pt[ids][j][1] - pt[ids][j-1][1]) < 10 :
						#print("in abs!!!!!!",(pt[ids][j-1]) , (pt[ids][j]))
						cv2.line(img_better_look,(pt[ids][j-1]) , (pt[ids][j]),(0,255,255),3)
					if len(pt[ids]) > 5:
						if j%5 == 0:
							#motion = True
							#x_dir_avg = np.average(x_dir)
							#y_dir_avg = np.average(y_dir)
							#parameters_avg = np.polyfit(x_dir_avg, y_dir_avg, 1)
							#cv2.line(img_better_look,(pts[ids][j-1][0],pts[ids][j-1][1]),(pts[ids][j][0],pts[ids][j][1]),(255,255,255),3)
							x_dirr = (pt[ids][0][0] , pt[ids][j][0])
							y_dirr = (pt[ids][0][1] , pt[ids][j][1])
							#x direction has same value
							#if pt[ids][0][0] == pt[ids][j][0] :
								#x_dirr = (pt[ids][0][0] , pt[ids][0][0]+2)
								#x direction has same value
							if pt[ids][0][1] == pt[ids][j][1] :
								y_dirr = (pt[ids][0][1] , pt[ids][0][1]+5)
								
							try:
								parameters = np.polyfit(x_dirr, y_dirr, 1)
								drw = True
								print("x dirr y dirr",x_dirr,y_dirr)
							except np.linalg.LinAlgError:
								drw = False
				if drw == True:
					x1 ,y1 ,x2 ,y2 = motion_cord((x1,y2) , parameters)
					#print("x1 y1 x2 y2",parameters,x1,y1,x2,y2)
					x_res_y2 = 1.062*y2 - 20
					cv2.line(img_better_look,(x1,y1),(x2,y2),(255,0,255),3)
					drw = False
					if x_res_y2 > x2 and y2 > 570 :
						motion_predict = True
						#cv2.putText(img_better_look, f"motion True", (700, 80), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,255), 2)  #bgr
						#parameters = np.polyfit(x_dir, y_dir, 1)
						#parameters = np.polyfit((pts[ids][j][0]), (pts[ids][j][1]),1)
						#print("(pts[ids][j][0] :",(pts[ids][j][0]))
						
						#x1 ,y1 ,x2 ,y2 = motion_cord((pts[ids][j][0], pts[ids][j][1] + (pts[ids][j][2]//2)) , parameters)
						#cv2.line(add_trans,(x1,y1),(x2,y2),(255,0,255),1)
						#cv2.line(img_better_look,(x1,y1),(x2,y2),(255,0,255),3)
						
						
					#trans = perspective_transformation(add_trans)
					#x1,y1 = pts[ids][j-1]
					#x2,y2 = pts[ids][j]
					#cv2.line(trans,(x1,(y1*300//960)), (x2,(y2*300//960)),(0,255,255),3)
					#img_trans = cv2.addWeighted(img_trans,1,trans,1,1) 
					
					
					
					
					
				'''
				speed estimation 
				==============
				estimate speed using deque method
				'''
				for i in range(0, len(w_list[ids])):
					#print("len(w_list[ids]) :",len(w_list[ids]),"; i :",i)
					#print("k in for loop",k)
					#i+=2
					width_curr = w_list[ids][i][0]
					if i%5 ==0 and k+1 <= i and len(w_list[ids]) > 5:  #sample every 3 points
						#print("k in if statement",k)
						width_1 = (w_list[ids][k-1][0])  #near wider
						width_2 = (w_list[ids][k-5][0])  #far
						#width_curr = (w_list[ids][i][0])
						time_passed = abs((w_list[ids][k-1][1]) - (w_list[ids][k-5][1]))
						name = (w_list[ids][k-1][2])
						if name == "" :
							name = (w_list[ids][k-5][2])
						print("time passed :",time_passed," time1 ",(w_list[ids][k-1][1])," time2 ",(w_list[ids][k-5][1]),)
						print("width difference :",abs(width_1-width_2))
						if time_passed > 0:
							if name == "deer":
								dis_deer2 = Distance_finder(100,width_2)/100
								dis_deer1 = Distance_finder(100,width_1)/100
								dis_deer =  Distance_finder(100,width_curr)/100
								dis_diff_deer = abs(dis_deer2 - dis_deer1)
								deer_speed = (dis_diff_deer/time_passed)*3600/1000
								car_spd[ids].append(deer_speed)
								avg_spd_deer = append_speed(ids,car_spd)
								if avg_spd_deer != "still appending":
									#print(avg_spd_deer)
									avg_spd_deer = int(avg_spd_deer)
									puttext_deer = True
									#cv2.putText(img_better_look, f"average deer speed {avg_spd}   km/h",  (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 2)  #bgr
								#print("deer speed",car_spd)
								#print("deer speed cord  :",car_spd[ids],"average speed :",avg_spd_deer)
								#cv2.putText(img_better_look, f"deer speed {int(deer_speed)}   km/h",  (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,255), 2)  #bgr
								#no class found ->usually motorbike
							if name == "motorbike" or name =="":
								dis_moto1 = Distance_finder(85,width_1)/100
								dis_moto2 = Distance_finder(85,width_2)/100
								dis_moto = Distance_finder(85,width_curr)/100
								dis_diff_moto = abs(dis_moto2 - dis_moto1)
								moto_speed = (dis_diff_moto/time_passed)*3600/1000
								moto_spd[ids].append(moto_speed)
								avg_spd_moto = append_speed(ids,moto_spd)
								#cv2.putText(img_better_look, f"motorbike speed {int(moto_speed)}  km/h",  (50, 90), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,255,0), 2)  #bgr    
								if avg_spd_moto != "still appending":
									# print(avg_spd_moto)
									avg_spd = int(avg_spd_moto)
									puttext_moto = True 
							
					#when w_list is short   
					elif len(w_list[ids]) == 5:
						width_1 = (w_list[ids][4][0])  #near wider
						width_2 = (w_list[ids][0][0])  #far
						time_passed = abs((w_list[ids][4][1]) - (w_list[ids][0][1]))
						name = (w_list[ids][3][2])
						if name == "" :
							name = (w_list[ids][1][2])
						if name == "deer":
							dis_deer2 = Distance_finder(100,width_2)/100
							dis_deer1 = Distance_finder(100,width_1)/100
							dis_diff_deer = abs(dis_deer2 - dis_deer1)
							deer_speed = (dis_diff_deer/time_passed)*3600/1000
							car_spd[ids].append(deer_speed)
							avg_spd_deer = append_speed(ids,car_spd)
							if avg_spd_deer != "still appending":
								print(avg_spd_deer)
								avg_spd_deer = int(avg_spd_deer)
								puttext_deer = True
							if avg_spd_moto != "still appending":
								avg_spd_moto = int(avg_spd_moto)
								puttext_moto = True      
					#k = 3*i
					k = i

				'''
				plot speed information
				
				'''
				if puttext_deer == True and  avg_spd_deer != "still appending" and deer_speed != 0 and avg_spd_deer != 0:
					deer_imptim_avg = round(dis_deer/(avg_spd_deer*1000/3600),2)               
					deer_imptim = round(dis_deer/(deer_speed*1000/3600),2)
					cv2.putText(img_better_look, f"average deer speed {avg_spd_deer} km/h Collision time {deer_imptim_avg} s",  (50, 90), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 2)  #bgr
					cv2.putText(img_better_look, f"deer speed {int(deer_speed)} km/h  Collision time {deer_imptim}s  ",  (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,255), 2)  #bgr
					if deer_imptim_avg < 1.25 and motion_predict == True:
						unsafe_v = True
						danger_v = False
					if deer_imptim_avg < 0.75 and motion_predict == True:
						danger_v = True
						unsafe_v = False
					else:
						danger_v = False
				
					#speed estimation ends here       
					
					if args.enable_ttc_forecasting and deer_speed > 0:
						if enhanced_ttc_forecasting(dis_deer, deer_speed):
							cv2.putText(img_better_look, "TTC Forecast Alert!", (50, 300), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,0), 2)       
					
					
				
				if cls == "deer":
					dis = Distance_finder(100,w)//100
					dis = int(dis)
					cv2.putText(img_better_look, f"Distance {dis} m", (xc-6, yc-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2)  #bgr
					# immediate threshold check per-object to ensure flags are set
					try:
						if args.enable_proactive_thresholds:
							if proactive_alert_thresholds(dis, deer_speed, True, motion_predict):
								danger_v = True
								unsafe_v = False
								print(f"[DBG] frame={framenumber} id={ids} class=deer dis={dis} proactive alert -> DANGER")
						else:
							if dis <= args.danger_dist:
								danger_v = True
								unsafe_v = False
								print(f"[DBG] frame={framenumber} id={ids} class=deer dis={dis} <= danger({args.danger_dist}) -> DANGER")
							elif dis <= args.unsafe_dist:
								unsafe_v = True
								danger_v = False
								print(f"[DBG] frame={framenumber} id={ids} class=deer dis={dis} <= unsafe({args.unsafe_dist}) -> MEDIUM")
					except Exception:
						pass
					#560
					if yc > 530 and yc <540:
						time_start = tim
						dis_start = dis
					#
					if yc > 560 and yc < 570:
						used = True
						time_end = tim
						dis_end = dis
						if used == True:
							if time_end-time_start == 0:
								print("time = 0") 
							else:
								speed = ((dis_start-dis_end)/(time_end-time_start))*3600/1000
							
								#cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr
								if speed < 0:
									speed = "calculating speed"
								else:
									bad = True
									speed = int(speed)
					if bad ==True :
						print("")
						cv2.putText(img_better_look, f"deer speed {speed} km/hr ",  (50,400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)
					if bad == True :
						# placeholder for additional deer logic
						pass
					#f.insert(0,(xc,yc))
					#print("deer center list :  ",f)

				# -- distance-based alerts (configurable thresholds)
				try:
					if args.enable_proactive_thresholds:
						if proactive_alert_thresholds(dis, deer_speed, True, motion_predict):
							danger_v = True
							unsafe_v = False
					else:
						if cls == "deer":
							if 'dis' in locals():
								if dis <= args.danger_dist:
									danger_v = True
									unsafe_v = False
								elif dis <= args.unsafe_dist:
									unsafe_v = True
									danger_v = False
				except Exception:
					# ignore any thresholding errors
					pass

					if yc > 560 and yc < 570:
						used = True
						time_end = tim
						dis_end = dis
						if used == True:
							if time_end-time_start == 0:
								print("time = 0")
							else: 
								bad = True
								speed = ((dis_start-dis_end)/(time_end-time_start))*3600/1000
								#cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr
					if bad == True :
						print("debuging speed",speed)
						if speed < 0 :
							print("speed is negative")
						else :
							speed = int(speed)
							cv2.putText(img_better_look, f"motorcycle speed {speed} km/hr ",  (50,300 ), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)   
											
										#cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr                           
										#cv2.putText(img_better_look, f"time start {time_start}  time end {time_end}",  (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)  #bgr
							#cv2.putText(img_better_look, f"motorcycle speed {speed} km/hr ",  (50,300 ), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)   
							#m.insert(0,(xc,yc))
							#print("motorbike center list:  ",m)
								
				'''
				safety zone
				=============
				safe unsafe danger 
				'''
				
				#unsafe
				
				#if x_res_y > xmin and ymin > 570 and danger==False or unsafe_v == True and danger==False:
				if unsafe_v == True: #and danger==False:
					#buzz(unsafe_v)
					#mylcd.lcd_display_string("unsafe",  2,3)
					#unsafe = True
					print("satis1")
					cv2.putText(img_better_look, f"Unsafe", (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)  #bgr
					#img1 = np.zeros_like(img_better_look)
					cv2.fillPoly(im0,pol, (255,0,255))
					#img = cv2.addWeighted(img1,0.7,img,1,1)
				'''             
				if ymin > 570 and xmin < 586 and danger == False: #straight behind
					unsafe = True
					print("satis2")                                   
					cv2.putText(img_better_look, f"Unsafe", (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)  #bgr
					#img2 = np.zeros_like(img_better_look)
					cv2.fillPoly(im0,pol, (255,0,255))
					#img = cv2.addWeighted(img2,0.7,img,1,1)
				'''
				#dangerous         
				#if x_res_yc > xc and yc > 570 or danger_v == True and unsafe == False:
				if danger_v == True: #and unsafe == False:
					#mylcd.lcd_display_string("dangerous!",  2,3)
					#buzz(unsafe_v)
					print("satis3")
					#danger = True #dangerous
					#unsafe = False
					cv2.putText(img_better_look, f"dangerous!!", (800, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)  #bgr
					#img3 = np.zeros_like(img_better_look)
					cv2.fillPoly(im0,pol, (0,0,255))
					#img = cv2.addWeighted(img3,0.7,img,1,1)
				'''
				elif yc > 570 and xc < 586:
					danger = True
					unsafe = False
					print("satis4")
					cv2.putText(img_better_look, f" danger!!", (800, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)  #bgr                          
					#img4 = np.zeros_like(img_better_look)
					cv2.fillPoly(im0,pol, (0,0,255))
					#img = cv2.addWeighted(img4,0.7,img,1,1)
					'''
		
		if len(outputs) > 0:
			#print("outputs after output right box",outputs)
			bbox_xyxy = outputs[:, :4]
			identities = outputs[:, -1]
			img_final = draw_boxes(img_better_look, bbox_xyxy, identities)
			#img_better_look = show_fps(img_better_look, fps)
		###################################
		img_better_look = show_fps(img_better_look, fps)
		imgx = img0
		real_result = cv2.addWeighted(img_better_look,0.7,img,1,1) #view lanedetection filtering
		img_better_look = cv2.addWeighted(im0,1,img_better_look,1,1) # 
		# Persistent top-right status banner: SAFE / MEDIUM / DANGER
		# This is drawn every frame so the state is always visible in the window and saved video.
		try:
			overlay = img_better_look.copy()
			alpha = 0.55
			# banner size and position (top-right)
			banner_w = 380
			banner_h = 70
			pad = 20
			x1 = img_better_look.shape[1] - banner_w - pad
			y1 = pad
			x2 = img_better_look.shape[1] - pad
			y2 = pad + banner_h
			# choose color and text based on state
			status_text = "SAFE"
			color = (0, 255, 0)  # green
			if danger_v:
				status_text = "DANGER"
				color = (0, 0, 255)  # red
			elif unsafe_v:
				status_text = "MEDIUM"
				color = (0, 127, 255)  # orange
			# draw filled rounded rectangle (approx) on overlay
			# cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
			# blend overlay
			# cv2.addWeighted(overlay, alpha, img_better_look, 1 - alpha, 0, img_better_look)
			# draw text right-aligned within banner
			font = cv2.FONT_HERSHEY_DUPLEX
			font_scale = 1.2
			thickness = 3
			(text_w, text_h), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
			text_x = x2 - 20 - text_w
			text_y = y1 + (banner_h + text_h) // 2
			# cv2.putText(img_better_look, status_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
		except Exception:
			# If overlay drawing fails for any reason, keep running without crashing
			pass

		# Log status changes (only when state changes) so we can debug why alerts may not appear
		try:
			if danger_v:
				status = 'DANGER'
			elif unsafe_v:
				status = 'MEDIUM'
			else:
				status = 'SAFE'
			if status != prev_status:
				print(f"[ALERT] frame={framenumber} status={status} unsafe_v={unsafe_v} danger_v={danger_v}")
				prev_status = status
		except Exception:
			pass
		
		# Add center flashing alert for danger
		if danger_v:
			# Flash by alternating color
			flash_color = (0, 0, 255) if (framenumber // 10) % 2 == 0 else (255, 255, 255)
			cv2.putText(img_better_look, "DANGER!", (320, 240), cv2.FONT_HERSHEY_DUPLEX, 2.0, flash_color, 5, cv2.LINE_AA)
			print(f"[ALERT] Center danger alert displayed frame={framenumber}")

		# Draw status banner
		banner_x = img.shape[1] - 200
		banner_y = 20
		banner_w = 180
		banner_h = 40
		if danger_v:
			banner_color = (0, 0, 255)  # Red
			status_text = "DANGER"
			text_color = (255, 255, 255)
		elif unsafe_v:
			banner_color = (0, 165, 255)  # Orange
			status_text = "MEDIUM"
			text_color = (255, 255, 255)
		else:
			banner_color = (0, 255, 0)  # Green
			status_text = "SAFE"
			text_color = (0, 0, 0)
		# cv2.rectangle(img_better_look, (banner_x, banner_y + 150), (banner_x + banner_w, banner_y + 200 + banner_h), banner_color, -1)
		# cv2.putText(img_better_look, status_text, (banner_x + 10, banner_y + 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, text_color, 2)

		out.write(line_visualize)
		out1.write(img_better_look)
		out2.write(combo_image)
		
		#show result
		#cv2.imshow(WINDOW_NAME, img)
		####
		#cv2.imshow("normal lanedetection without extended",normal_result)
		#cv2.imshow("combo img",combo_image)
		#cv2.imshow(" img",img_notavg) 
		#cv2.imshow("avg line",line_visualize)
		####
		#cv2.imshow("only safety zone",img_better_look)
		# cv2.imshow("cropped image ",cropped_image)
		#cv2.imshow("real result ",real_result)
		#cv2.imshow("example ",real_result)
		#cv2.imshow("image better look",img_better_look)
		#cv2.imshow("predict motion  ",img_trans)
		# cv2.imshow("predict motion  ",god)
		# try:
		#  cv2.imshow("filtered",filtered)
		# except NameError:
		#   print("")
		#cv2.imshow("imgx",imgx)
		toc = time.time()
		curr_fps = 1.0 / (toc - tic)
		# calculate an exponentially decaying average of fps number
		fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
		tic = toc
		key = cv2.waitKey(1)
		if key == 27:  # ESC key: quit program
			break
		elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
			full_scrn = not full_scrn
			set_display(WINDOW_NAME, full_scrn)

	# Processing complete
	total_time = time.time() - start_time
	if hasattr(detector, 'device'):
		print(f"Processing complete. Device used: {detector.device}. Total time: {total_time:.2f} seconds")
	else:
		print(f"Processing complete. Total time: {total_time:.2f} seconds")


def main():
	args = parse_args()
	########
	cfg = get_config()
	cfg.merge_from_file(args.config_deepsort)    
	########
	if args.category_num <= 0:
		raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
	if not args.use_opencv and not args.use_pytorch:
		if not os.path.isfile('yolo/%s.trt' % args.model):
			raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

	cam = Camera(args)
	if not cam.isOpened():
		raise SystemExit('ERROR: failed to open camera!')
	########    
	tracker = Tracker_tiny(cfg) 
	########
	if args.use_pytorch:
		# For custom PyTorch model, assume class 0 is deer
		cls_dict = {0: 'deer'}
	else:
		cls_dict = get_cls_dict(args.category_num)
	if args.use_pytorch:
		# Use PyTorch YOLO
		detector = PyTorchYolo(args.pytorch_model)
	elif args.use_opencv:
		# Default input shape for OpenCV YOLO
		h = w = 416
		# Use OpenCV DNN fallback on CPU
		detector = OpenCVYolo(args.yolo_cfg, args.yolo_weights, input_shape=(h, w))
	else:
		yolo_dim = args.model.split('-')[-1]
		if 'x' in yolo_dim:
			dim_split = yolo_dim.split('x')
			if len(dim_split) != 2:
					raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
			w, h = int(dim_split[0]), int(dim_split[1])
		else:
			h = w = int(yolo_dim)
		if h % 32 != 0 or w % 32 != 0:
			raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
		trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)
		detector = trt_yolo

	#open_window(WINDOW_NAME, 'Camera TensorRT YOLO Demo',cam.img_width, cam.img_height)
			
	vis = BBoxVisualization(cls_dict)
	
	loop_and_detect(cam, detector, tracker, conf_th=args.confidence_threshold, vis=vis, args=args)

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
