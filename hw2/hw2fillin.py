#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt

# import modules used here -- sys is a very standard one
import sys

refPt = []
xcoord = []
ycoord = []
newx = [0, 700, 700, 0]
newy = [0, 0, 400, 400]
testx = [0, 813, 813, 0]			# coordinates of test image
testy = [0, 0, 353, 353]
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1

def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image,lx,ly
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates 
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		#print  (x,y)
		lx = x
		ly = y
		xcoord.append(lx)				# problem 1: collects coordinates to move to frontal view
		ycoord.append(ly)				# problem 2: collects coordinates to project to

# Calculates new points using direct linear transform
def dlt(image, homography):
	y, x, z = image.shape
	twoD = []
	mapx = np.zeros(image.shape[:2], np.float32)
	mapy = np.zeros(image.shape[:2], np.float32)
	
	for j in range(y):
		for i in range(x):
			h = np.linalg.inv(homography).dot([i, j, 1]).T
			a = h/h[2]
			mapx[j,i] = a[0]
			mapy[j,i] = a[1]

	return cv2.remap(image, mapx, mapy, 1)

# gets a frontal view of the selected frame
def frontView(image):
	# Creates matrix A for the first problem
	def matrixA():
		A = []
		for n in range(4):
			print(n)
			A.append([-1*xcoord[n], -1*ycoord[n], -1, 0, 0, 0, newx[n]*xcoord[n], newx[n]*ycoord[n], newx[n]])
			A.append([0, 0, 0, -1*xcoord[n], -1*ycoord[n], -1, newy[n]*xcoord[n], newy[n]*ycoord[n], newy[n]])
			print(A)
		
		return np.matrix(A)

	# problem 2
	A = matrixA()
	#stri = 'This is the shape of A ' + repr(A.shape)
	#print(str)
	U, s, V = np.linalg.svd(A)
	V = V.T
	H = V[:,-1]						# retrieves last column
	H.shape = (3,3)
	#stri = 'This is H ' + repr(H)
	#print(stri)
	cv2.imshow("Frontal View", dlt(image, H))
	k = cv2.waitKey(0)
	cv2.destroyAllWindows()

def subBillboard():
	# Creates matrix A for the second problem; new coordinates are actually clicked coordinates stored in xcoord[n] and ycoord[n]
	def testMatrixA():
		testA = []

		for n in range(4):
			print(n)
			testA.append([-1*testx[n], -1*testy[n], -1, 0, 0, 0, xcoord[n]*testx[n], xcoord[n]*testy[n], xcoord[n]])
			testA.append([0, 0, 0, -1*testx[n], -1*testy[n], -1, ycoord[n]*testx[n], ycoord[n]*testy[n], ycoord[n]])
			print(testA)
		
		return np.matrix(testA)
	
	image = cv2.imread('ts.jpg',1);
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
 
	# keep looping until the 'q' key is pressed
	while True:
	# display the image and wait for a keypress
		image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
		cv2.imshow(windowName, image)
		key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break

	# print("problem 3")
	testimage = cv2.imread('testimage.jpg',1);
	image = cv2.imread('ts.jpg',1);
	rect = np.zeros(image.shape, np.uint8)
	rect[0:testimage.shape[0], 0:testimage.shape[1]] = testimage
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)

# keep looping until the 'q' key is pressed
# chooses billboard to replace! selects (x,y) to match new image to
	#print(testimage.shape)
	#print(image.shape)
	testA =testMatrixA()
	#stri = 'This is the shape of A ' + repr(testA.shape)
	#print(stri)
	U, s, V = np.linalg.svd(testA)
	V = V.T
	H = V[:,-1]						# retrieves last column
	H.shape = (3,3)					# homography of test image to be projected
	#stri = 'This is H ' + repr(H)
	#print(stri)
	testdlt = dlt(rect, H)

	rows, cols, ch = testimage.shape

	grey = cv2.cvtColor(testdlt, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)

	imgBG = cv2.bitwise_and(image, image, mask = mask_inv)
	imgFG = cv2.bitwise_and(testdlt, testdlt, mask = mask)
	dst = cv2.add(imgBG, imgFG)

	cv2.imshow('Projection', dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def house():
	f = 100
	sx = sy = 1
	ox = oy = 200
	dim = (600, 600)
	
	def to2D(housePts):
		pts = []
		for p in housePts:
			shifted3D = np.array([p[0]-200, p[1]+100, p[2], 1])		# displaces viewpoint
			pt = M.dot(shifted3D).T
			pt = pt/pt[2]
			pts.append((pt[0].item(), pt[1].item()))
		return pts

	def line(coords):
		coords.append(coords[0])
		coor = np.array(coords, np.int32)
		coor.shape = (len(coords), 1, 2)
		return coor

	Mext = np.matrix([[-0.707, -0.707, 0, 3],		# 3x4 external camera matrix
		[0.707, -0.707, 0, 0.5],
		[0, 0, 1, 3]])
	Mint = K = np.matrix([[1/sx, 0, ox],			# 3x3 internal camera matrix
		[0, 1/sy, oy],
		[0, 0, 1]])
	M = Mint.dot(Mext)								# 3x4 camera matrix
	
# house matrix
	s = 300											# sides of house
	houseCoords = [[(0, 0, 0), (0, s, 0), (s, s, 0), (s, 0, 0)], 		# front
		[(0, 0, s), (0, s, s), (s, s, s), (s, 0, s)], 					# back
		[(0, 0, 0), (0, s, 0), (0, s, s), (0, 0, s)], 					# left
		[(s, 0, 0), (s, s, 0), (s, s, s), (s, 0, s)], 					# right
		[(0, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, s), (0, s, s)],	 	# roof panel 1
		[(s, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, s), (s, s, -s)]		# roof panel 2
	]

	twoDpts = [to2D(p) for p in houseCoords]
	model = np.zeros(dim, np.uint8)

# creates 3D model
	for d in twoDpts:
		model = cv2.polylines(model, [line(d)], False, (255, 255, 255))

	cv2.imshow('House', model)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Gather our code in a main() function
def main():
	# Read Image
	global image, testimage
	image = cv2.imread('ts.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
 
# keep looping until the 'q' key is pressed

	while True:
	# display the image and wait for a keypress
		image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
		cv2.imshow(windowName, image)
		key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break
	
	# warp image
	frontView(image)
	subBillboard()
	house()

	# Close the window will exit the program
	cv2.destroyAllWindows()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

# Select End Points of foreshortened window or billboard

# Set the corresponding point in the frontal view as 

# Estimate the homography 

# Warp the image
# cv.Remap(src, dst, mapx, mapy, flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, fillval=(0, 0, 0, 0)) 

#Crop the image

# replaces a selected frame with testimage.jpg