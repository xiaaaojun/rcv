#!/usr/bin/env python
import numpy as np
import cv2
import math
import glob
from PIL import Image
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

# import modules used here -- sys is a very standard one
import sys

def problem1():
    M = Mint.dot(Mext)								# 3x4 camera matrix

    twoDpts = [to2D(p, M) for p in houseCoords]
    model = np.zeros(dim, np.uint8)
    drawMyObject(twoDpts, model)

    cv2.imshow('House', model)
    close("c")
    cv2.destroyWindow('House')

def to2D(housePts, M):
    pts = []
    for p in housePts:
        shifted3D = np.array([p[0]-30, p[1]-100, p[2]-70, 1])		# displaces origin
        pt = M.dot(shifted3D).T
        pt = pt/pt[2]
        #print(pt)
        pts.append((pt[0].item(), pt[1].item()))
 
    pts = np.array(pts, np.int32)
    pts = pts.reshape((len(pts),1,2))
    return pts

def drawMyObject(twoDpts, model):       
    for coords in twoDpts:
        cv2.polylines(model, [coords], True, (255,255,255))
    return

def close(key):
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord(key):
            break

def problem2():
    a, b, c, tx, ty, tz = -5, -5, 0, -30, -200, -70

    for n in range(15):
        #print(n)
        rotX = np.matrix([[1, 0, 0],
                        [0, math.cos(math.radians(a)), -math.sin(math.radians(a))],
                        [0, math.sin(math.radians(a)), math.cos(math.radians(a))]])
        rotY = np.matrix([[math.cos(math.radians(b)), 0, math.sin(math.radians(b))],
                        [0, 1, 0],
                        [-math.sin(math.radians(b)), 0, math.cos(math.radians(b))]])
        rotZ = np.matrix([[math.cos(math.radians(c)), -math.sin(math.radians(c)), 0],
                        [math.sin(math.radians(c)), math.cos(math.radians(c)), 0],
                        [0, 0, 1]])

        tr = np.matrix([tx, ty, tz]).T
        R = rotX.dot(rotZ.dot(rotY))

        Rt = np.zeros((3,4),np.float64)
        Rt[:3,:3] = R     
        Rt[:,3] = np.array([tx, ty, tz])

        M2 = Mint.dot(Rt)

        twoDpts = [to2D(p, M2) for p in houseCoords]
        model = np.zeros(dim, np.uint8)
        drawMyObject(twoDpts, model)

        cv2.imshow('Fly by', model)
        close("n")
        cv2.destroyWindow('Fly by')

        a += 2          # alpha angle increment
        b -= 1          # beta angle increment
        c -= 0.5        # theta angle increment
        tx -= 10        # translation increment in x
        ty += 3         # translation increment in y
        tz -= 5         # translation increment in z
    return

def problem4():
    r = 7
    c = 7
    rlpts = np.zeros((r*c, 3), np.float32)
    rlpts[:,:2] = np.mgrid[0:r, 0:c].T.reshape(-1,2)
    
    wldpts = []       # 3D points taken from pictures
    imgpts = []       # 2D points for projected image

    images = glob.glob('/home/xxkanade/Documents/rcv/hw3/*.JPG')            # compiles all images into a list

    for file in images:
        img = cv2.imread(file);
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        find, corners = cv2.findChessboardCorners(gray, (r, c), None)
        #cornerSubPix()

        if find:
            wldpts.append(rlpts)                                            # adds rl points to 3D picture pts
            imgpts.append(corners)                                          # adds corners to 2D pts
            cv2.drawChessboardCorners(img, (r,c), corners, find)
            #cv2.namedWindow('Checkerboard', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('Checkerboard', 1000, 800)
            #cv2.imshow('Checkerboard', img)
            #close("p")
            #cv2.destroyWindow('Checkerboard')
    
    cv2.destroyAllWindows()
    found, M, dist, rvecs, tvecs = cv2.calibrateCamera(wldpts, imgpts, gray.shape[::-1], None, None)
    K = M               # intrinsic camera matrix
    print(K)
    return

def main():
    global Mint, Mext, M, dim, s, houseCoords

    f = 100
    sx = sy = 1
    ox = oy = 200
    dim = (500,500)

    Mext = np.matrix([[-0.507, 0.407, 0, 3],		# 3x4 external camera matrix
                      [0.507, 0.407, 0, 0.5],
                      [0, 0, 1, 3]])
    Mint = K = np.matrix([[f/2, 0, ox],			    # 3x3 internal camera matrix
                        [0, f/2, oy],
                        [0, 0, 1]])
    # house matrix
    s = 150											                    # sides of house
    houseCoords = [[(0, 0, 0), (0, s, 0), (s, s, 0), (s, 0, 0)], 		# front
        [(0, 0, -s), (s, 0, -s), (s, s, -s), (0, s, -s)], 				# back
    	[(0, s, -s), (0, s, 0), (0, 0, 0), (0, 0, -s)], 				# left
		[(s, 0, 0), (s, s, 0), (s, s, -s), (s, 0, -s)], 				# right
		[(0, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, -s), (0, s, -s)],	 	# roof panel 1
		[(s, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, -s), (s, s, -s)]]		# roof panel 2

    problem1()
    problem2()
    problem4()

if __name__ == '__main__':
    main()