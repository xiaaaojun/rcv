#!/usr/bin/env python
import numpy as np
import cv2
import math
import glob
from os import listdir
from os.path import isfile, join

# import modules used here -- sys is a very standard one
import sys
import cv2
import numpy as np

def conv(polyPts, M):
	pts = []
	for p in polyPts:
		vec = np.array([p[0]+20, p[1]-70, p[2]-20, 1])
		newpt = M.dot(vec).T
		newpt = newpt/newpt[2]
		pts.append((newpt[0].item(), newpt[1].item()))
	return pts

def to2D(poly, M):
	Dpts = []
	for p in poly:
	#	print(p[0])
		ve = np.array([p[0]+20, p[1]-70, p[2]-20, 1])
	#	print(shifted3D)
		pt = M.dot(ve).T
		pt = pt/pt[2]
		Dpts.append((pt[0].item(), pt[1].item()))
		
	Dpts = np.array(Dpts, np.int32)
	Dpts = Dpts.reshape((len(Dpts),1,2))
	return Dpts

def drawMyObject(Dpts, model):	
	for dpt in Dpts:
	#	print(type(dpt))
	#	print(dpt.shape)
		cv2.polylines(model, [dpt], 1, (255,255,255))
	return

def close(key):
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord(key):
            break

def problem1():
	def calcFundamental(pL, pR):
		mat = np.array([])
		for i in range(len(pL)):
			l = pL[i]
			r = pR[i]
			mat = np.append(mat, [l[0]*r[0], l[0]*r[1], l[0], l[1]*r[0], l[1]*r[1], l[1], r[0], r[1], 1], axis=0)

		mat.shape = (9, 9)
		U, S, V = np.linalg.svd(mat)
		V = V.T[:,-1]
		V.shape = (3,3)
		return V

	ML = Mint.dot(MextL)								# 3x4 Left camera matrix
	MR = Mint.dot(MextR)								# 3x4 Right camera matrix

	modelL = np.zeros(dim, np.uint8)
	modelR = np.zeros(dim, np.uint8)

	ptsL = conv(poly, ML)
	ptsL = np.asarray(ptsL)
	DptsL = [to2D(p, ML) for p in polyarr]
	drawMyObject(DptsL, modelL)
	cv2.imshow('Left', modelL)
	close("c")

	ptsR = conv(poly, MR)
	ptsR = np.asarray(ptsR)
	DptsR = [to2D(p, MR) for p in polyarr]
	drawMyObject(DptsR, modelR)
	cv2.imshow('Right', modelR)
	close("c")

	fund1 = calcFundamental(ptsL, ptsR)
	print("fundamental matrix")
	print(fund1)
	print("fundamental matrix using the function")
	fund2 = cv2.findFundamentalMat(np.array(ptsL), np.array(ptsR), cv2.FM_8POINT)[0]
	print(fund2)
	
	cv2.destroyAllWindows()
	return fund1, ptsL, ptsR

def problem2(fund, K):
	E = K.T.dot(fund).dot(K)
	print("This is the Essential matrix calculated in Problem 2: ")
	print(E)
	return E

def problem3(fund, ess, K, ptsL, ptsR):
	fin = False
	w = np.array([[0, -1, 0],
        		  [1,  0, 0],
        		  [0,  0, 1]])
	z = np.array([[ 0, 1, 0],
				  [-1, 0, 0],
				  [ 0, 0, 0]])
	u, s, v = np.linalg.svd(ess, full_matrices=1)
	Mr = np.array([[1, 0, 0, 0],
				   [0, 1, 0, 0],
				   [0, 0, 1, 0]])
	Mr = K.dot(Mr)
	s = [-u.dot(z).dot(u.T), u.dot(z).dot(u.T)]
	r = [u.dot(w.T).dot(v.T), u.dot(w).dot(v.T)]

	for i in s:
		for j in r:
			Ml = np.zeros((3,4))
			Ml[:, :3] = j
			Ml[:, 3] = np.array([i[2, 1], i[0,2], -i[0,1]])
			Ml = K.dot(Ml)
			tDrecon = cv2.triangulatePoints(Ml, Mr, np.array(ptsL).T, np.array(ptsR).T)
			if np.amin(s) > 0:
				fin = True
			if fin:
				break
		if fin:
			break
	return

	# My display function doesn't work so I removed this section.

def problem4(ptsL, ptsR):
	Ml = Mint.dot(MextL)
	Mr = Mint.dot(MextR)
	recon = cv2.triangulatePoints(Ml, Mr, np.array(ptsL).T, np.array(ptsR).T)

	# My display function doesn't work so I removed this section.

def problem5():
	r = 7
	c = 7
	rlpts = np.zeros((r*c, 3), np.float32)
	rlpts[:,:2] = np.mgrid[0:r, 0:c].T.reshape(-1,2)
	
	wldpts = []       # 3D points taken from pictures
	imgpts = []       # 2D points for projected image
	images = glob.glob('/home/xxkanade/Documents/rcv/hw5/*.JPG')            # compiles all images into a list
	
	for file in images:
		img = cv2.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		find, corners = cv2.findChessboardCorners(gray, (r, c), None)
		
		if find:
			wldpts.append(rlpts)                                            # adds rl points to 3D picture pts
			imgpts.append(corners)                                          # adds corners to 2D pts
			cv2.drawChessboardCorners(img, (r,c), corners, find)
 			cv2.namedWindow('Checkerboard', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('Checkerboard', 1000, 800)
			cv2.imshow('Checkerboard', img)
			close("n")
			cv2.destroyWindow('Checkerboard')

	cv2.destroyAllWindows()
	found, M, dist, rvecs, tvecs = cv2.calibrateCamera(wldpts, imgpts, gray.shape[::-1], None, None)
	K = M               # intrinsic camera matrix
#	print(rvecs)
	rvecs = np.asarray(rvecs)
#	print(type(rvecs))
	rMat = cv2.Rodrigues(rvecs)[0]
	print("K Values:")
	print(K)
	print("cTw Values:")
	print(rMat)
	return

def main():
	global Mint, MextL, ML, MextR, MR, dim, s, poly, polylist, polyarr, polyModel, polyPts
	
	f = 100
	sx = sy = 2
	ox = oy = 200
	dim = (1000,1000)
	MextL = np.matrix([[-0.507, 0.407, 0, 2],		# 3x4 external L camera matrix
						[0.507, 0.407, 0, 0.3],
						[0, 0, 1, 3]])
	MextR = np.matrix([[-0.607, 0.420, 0, 25],		# 3x4 external R camera matrix
						[0.607, 0.420, 0, 0.1],
						[0, 0, 1, 7]])					
	Mint = K = np.matrix([[f/sx, 0, ox],			# 3x3 internal  camera matrix
						[0, f/sy, oy],
						[0, 0, 1]])
	s = 10										           
	polyPts = [
			(0, 0, 0), (0, s, 0), (s, s, 0), (s, 0, 0), (0, s+20, s), (s+10, s, 0), (0, 0, s), (s+10, 0, s), (10+s, s, 0) 					# front      
		   	]
	polylist = polyPts
	polyarr = [polyPts]
	poly = np.asarray(polyPts)
	
	fund, ptsl, ptsr = problem1()
	E = problem2(fund, K)
	problem3(fund, E, K, ptsl, ptsr)
	problem4(ptsl, ptsr)
	problem5()

if __name__ == '__main__':
	main()