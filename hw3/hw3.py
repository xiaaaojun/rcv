#!/usr/bin/env python
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# import modules used here -- sys is a very standard one
import sys

def problem1():
    M = Mint.dot(Mext)								# 3x4 camera matrix

    twoDpts = [to2D(p, M) for p in houseCoords]
    model = np.zeros(dim, np.uint8)
    drawMyObject(twoDpts, model)

    cv2.imshow('House', model)
    cv2.waitKey(0)
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

def problem2():
    Mext = np.matrix([[-0.507, 0.707, 0, 3],		# 3x4 external camera matrix
                      [0.507, 0.707, 0, 0.5],
                      [0, 0, 1, 3]])
    a, b, c, tx, ty, tz = 0, 0, 0, -30, -100, -70


    for n in range(2):
        print(n)
        rotX = np.matrix([[1, 0, 0],
                        [0, math.cos(math.radians(a)), -math.sin(math.radians(a))],
                        [0, math.sin(math.radians(a)), math.cos(math.radians(a))]])
        rotY = np.matrix([[math.cos(math.radians(b)), 0, math.sin(math.radians(b))],
                        [0, 1, 0],
                        [-math.sin(math.radians(b)), 0, math.cos(math.radians(b))]])
        rotZ = np.matrix([[math.cos(math.radians(c)), -math.sin(math.radians(c)), 0],
                        [math.sin(math.radians(c)), math.cos(math.radians(c)), 0],
                        [0, 0, 1]])

        #rotX = np.matrix([[1, 0, 0],
        #                [0, math.cos(a), -math.sin(a)],
        #                [0, math.sin(a), math.cos(a)]])
        #rotY = np.matrix([[math.cos(b), 0, math.sin(b)],
        #                [0, 1, 0],
        #                [-math.sin(b), 0, math.cos(b)]])
        #rotZ = np.matrix([[math.cos(c), -math.sin(c), 0],
        #                [math.sin(c), math.cos(c), 0],
        #                [0, 0, 1]])
        tr = np.matrix([tx, ty, tz]).T
        R = rotX.dot(rotZ.dot(rotY))

        Rt = np.zeros((3,4),np.int32)
        Rt[:3,:3] = R
        Rt[:,3] = np.array([tx, ty, tz])

        print(Rt)

        M2 = Mint.dot(Rt)
        print(M2)
        #i, j = Rt. shape
        #if j == 4:
        #    Rt = np.delete(Rt, -1, axis=1)

        twoDpts = [to2D(p, M2) for p in houseCoords]
        model = np.zeros(dim, np.uint8)
        drawMyObject(twoDpts, model)

        cv2.imshow('Fly by', model)
        cv2.waitKey(0)
        cv2.destroyWindow('Fly by')

        #a -= 1          # alpha angle
        #b -= 2         # beta angle
        #c -= 2         # theta angle
        tx += 5         # translation in x
        ty -= 5
        tz += 5
    return
    cv2.destroyAllWindows


def main():
    global Mint, Mext, M, dim, s, houseCoords

    f = 100
    sx = sy = 1
    ox = oy = 200
    dim = (800,800)

    Mext = np.matrix([[-0.507, 0.407, 0, 3],		# 3x4 external camera matrix
                      [0.507, 0.407, 0, 0.5],
                      [0, 0, 1, 3]])
    Mint = K = np.matrix([[f/2, 0, ox],			    # 3x3 internal camera matrix
                        [0, f/2, oy],
                        [0, 0, 1]])
    # house matrix
    s = 100											                    # sides of house
    houseCoords = [[(0, 0, 0), (0, s, 0), (s, s, 0), (s, 0, 0)], 		# front
        [(0, 0, -s), (s, 0, -s), (s, s, -s), (0, s, -s)], 				# back
    	[(0, s, -s), (0, s, 0), (0, 0, 0), (0, 0, -s)], 				# left
		[(s, 0, 0), (s, s, 0), (s, s, -s), (s, 0, -s)], 				# right
		[(0, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, -s), (0, s, -s)],	 	# roof panel 1
		[(s, s, 0), (s/2, 1.5*s, 0), (s/2, 1.5*s, -s), (s, s, -s)]]		# roof panel 2

    problem1()
    problem2()

if __name__ == '__main__':
    main()