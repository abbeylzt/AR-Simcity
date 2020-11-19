#########################################################################
# https://drive.google.com/open?id=1Wxqz4FFKxQcei_j0XZbztI10QS2AwkSL
# Folder name: Zhaotong_Liu_112_Project
# There are pictures needed to run this file in the above link
#########################################################################

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from math import cos, sin, radians
from tkinter import *
from PIL import Image
from PIL import ImageTk
from time import time 
from tkinter.ttk import Button 
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class App(object):
    def __init__(self, cam, viewer, city):
        self.cam = cam
        cam.app = self
        self.viewer = viewer
        viewer.app = self
        self.city = city
        city.app = self
        self.pause = False
        # self.interval = 1.0
        # self.lastTime = time()
        self.start = time()
        self.lastTime = time()
        self.interval = 2

    def keyPress(self, event):
        if event.char == 'u':
            city.upgrade()
        if event.char == 'd':
            city.demolish()
        if event.char == 'i':  # up
            city.move(0, 1)
        if event.char == 'k':  # down
            city.move(0, -1)
        if event.char == 'l':  # right
            city.move(-1, 0)
        if event.char == 'j':  # left
            city.move(1, 0)
        if event.char == '1':
            city.build('house')
        if event.char == '2':
            city.build('apartment')
        if event.char == '3':
            city.build('official')

    def callBack(self):
        self.viewer.img, self.viewer.camData = self.cam.capture()
        self.city.checkSelection()
        if time() - self.lastTime >= self.interval:
            self.city.update()
            self.lastTime = time()
        self.viewer.drawAll(self.city.objects())
        # self.drawAllTest()
        
        toExit = False
        for b in viewer.actionButtons:
            if b.text == 'Exit' and b.hovered:
                toExit = True
            if b.text == 'Upgrade' and b.clicked and b.vis:
                city.upgrade()
                b.clicked = False
            if b.text == 'Demolish' and b.clicked and b.vis:
                city.demolish()
                b.clicked = False
            if b.text == 'Cancel' and b.clicked and b.vis:
                city.cancel()
                b.clicked = False
            if b.text == 'Back' and b.clicked and b.vis:
                b.clicked = False
                for c in viewer.actionButtons:
                    c.vis = c.devis
            if b.text == 'Move' and b.clicked and b.vis:
                b.clicked = False
                subMove = ['Upgrade', 'Demolish', 'Back', 'Build', 'Cancel',
                 'Up', 'Down', 'Left', 'Right']
                for c in viewer.actionButtons:
                    if c.text in subMove:
                        c.vis = not c.vis

            if b.text == "Up" and b.clicked and b.vis:
                city.move(0, 1)
                b.clicked = False
            if b.text == "Down" and b.clicked and b.vis:
                city.move(0, -1)
                b.clicked = False
            if b.text == "Right" and b.clicked and b.vis:
                city.move(1, 0)
                b.clicked = False
            if b.text == "Left" and b.clicked and b.vis:
                city.move(-1, 0) 
                b.clicked = False

            if b.text == 'Build' and b.clicked and b.vis:
                b.clicked = False
                subBuild = ['Upgrade', 'Demolish', 'Apartment', 'Official', 'House',
                'Back', 'Cancel', 'Move']
                for c in viewer.actionButtons:
                    if c.text in subBuild:
                        c.vis = not c.vis
            if b.text == 'Apartment' and b.clicked and b.vis:
                city.build('apartment')
                b.clicked = False
            if b.text == 'Official' and b.clicked and b.vis:
                city.build('official')
                b.clicked = False
            if b.text == 'House' and b.clicked and b.vis:
                city.build('house')
                b.clicked = False

        if not toExit:
            viewer.panelA.after(20, self.callBack)
        else:
            viewer.root.destroy()


    def run(self):
        self.viewer.init()
        viewer.panelA.after(100, self.callBack)
        self.viewer.root.bind('<KeyPress>', self.keyPress)
        viewer.root.mainloop()

class Camera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        try:
            # try part: https://rdmilligan.wordpress.com/2015/07/31/3d-augmented
            # -reality-using-opencv-and-python/
            # Load previously saved data
            with np.load('webcam.npz') as X:
                self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
        except:
            self.calibrate()
            with np.load('webcam.npz') as X:
                self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((7*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

        self.rvec0 = np.array([[0,0,0]], dtype=np.float32)
        self.tvec0 = np.array([[0,0,0]], dtype=np.float32)
        self.rvecHand = None
        self.rvecGround = None
        self.pHand = None
        self.pHand0 = None

    def close(self):
        self.cap.release()

    def calibrate(self):
        # Algorithm: https://rdmilligan.wordpress.com/2015/07/02/augmented-reality-
        # using-opencv-and-python/
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((7*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('train*.png')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            cv2.imshow('img',gray)
            cv2.waitKey(500)
            ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
        gray.shape[::-1],None,None)
        np.savez("webcam", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    def convertHandToGround(self):
        foundHand = False
        # check whether found hand or ground
        if len(self.ids) > 0:
            if self.ids[0] == 1:
                self.rvecHand = self.rvecs[0]
                self.tvecHand = self.tvecs[0]
                foundHand = True
            if self.ids[0] == 2:
                self.rvecGround = self.rvecs[0]
                self.tvecGround = self.tvecs[0]

        if len(self.ids) > 1:
            if self.ids[1] == 1:
                self.rvecHand = self.rvecs[1]
                self.tvecHand = self.tvecs[1]
                foundHand = True
            if self.ids[1] == 2:
                self.rvecGround = self.rvecs[1]
                self.tvecGround = self.tvecs[1]

        if (not self.rvecHand is None) and (not self.rvecGround is None) and foundHand:
            # convert the position of the pointer
            p = np.float32([0,15,0]).reshape(3,1)
            tvecGroundFromCam = - self.tvecGround.reshape(3,1)
            RGroundFromCam = cv2.Rodrigues(self.rvecGround)[0].transpose()
            RHand = cv2.Rodrigues(self.rvecHand)[0]
            # convert to camera coordinate
            p = np.dot(RHand, p)
            p = p + self.tvecHand.reshape(3,1)
            # conver to ground coordinate
            p = p + tvecGroundFromCam.reshape(3,1)
            p = np.dot(RGroundFromCam, p)
            self.pHand = p
            # convert the positon of the hand
            p = np.float32([0,0,0]).reshape(3,1)
            p = np.dot(RHand, p)
            p = p + self.tvecHand.reshape(3,1)
            p = p + tvecGroundFromCam.reshape(3,1)
            p = np.dot(RGroundFromCam, p)
            self.pHand0 = p

        else:
            self.pHand = None
            self.pHand0 = None


    def capture(self):
        # Algorithm: https://rdmilligan.wordpress.com/2015/07/02/augmented-reality-
        # using-opencv-and-python/
        ret, img = self.cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, 
        parameters=parameters)
        font = cv2.FONT_HERSHEY_SIMPLEX
        data = []
        if np.all(ids != None):
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 3, self.mtx, self.dist)
            data = [rvec, tvec, self.mtx, self.dist, ids]
            self.rvecs = rvec
            self.tvecs = tvec
            self.ids = ids
            self.convertHandToGround()
        else:
            self.pHand = None
            self.pHand0 = None

        return img, data

class Viewer(object):
    def __init__(self, cam):
        self.img = None
        self.camData = None
        self.width = 480
        self.height = 300
        self.location = np.array([-3,-12, -3])
        self.lookAt = np.array([0,0,0])
        self.cam = cam
        self.gap =150
        self.actionWidth = 100
        self.actionHeight = 50
        self.actionButtons = [
            ButtonCV(50, 25, self.actionWidth, self.actionHeight, True, True, "Build", (100,100,250)),
            ButtonCV(50+self.actionWidth+self.gap, 25, self.actionWidth, self.actionHeight, True, True, "Upgrade", (100,100,250)),
            ButtonCV(50+self.actionWidth*2+self.gap*2, 25, self.actionWidth, self.actionHeight, True, True, "Demolish", (100,100,250)),
            ButtonCV(50+self.actionWidth*3+self.gap*3, 25, self.actionWidth, self.actionHeight, True, True, "Move", (100,100,250)),
            ButtonCV(50+self.actionWidth*4+self.gap*4, 25, self.actionWidth, self.actionHeight, True, True, "Exit", (100,100,250)),
            ButtonCV(50+self.actionWidth+self.gap, 100, self.actionWidth, self.actionHeight, False, False, "House", (100,100,250)),
            ButtonCV(50+self.actionWidth*2+self.gap*2, 100, self.actionWidth, self.actionHeight, False, False, "Apartment", (100,100,250)),
            ButtonCV(50+self.actionWidth*3+self.gap*3, 100, self.actionWidth, self.actionHeight, False, False, "Official", (100,100,250)),
            ButtonCV(50, 100, self.actionWidth, self.actionHeight, False, False, "Left", (100,100,250)),
            ButtonCV(50+self.actionWidth+self.gap, 100, self.actionWidth, self.actionHeight, False, False, "Up", (100,100,250)),
            ButtonCV(50+self.actionWidth*2+self.gap*2, 100, self.actionWidth, self.actionHeight, False, False, "Right", (100,100,250)),
            ButtonCV(50+self.actionWidth*3+self.gap*3, 100, self.actionWidth, self.actionHeight, False, False, "Down", (100,100,250)),
            ButtonCV(100, 600, 100, 50, False, False, "Cancel", (100,100,250)),
            ButtonCV(50+self.actionWidth*4+self.gap*4, 600, 100, 50, False, False, "Back", (100,100,250))
        ]
        
    def init(self):
        cv2.startWindowThread()
        self.root = Tk()
        self.panelA = Label(width=self.width,height=self.height)
        self.panelA.pack(side="top", padx=10, pady=10)
        self.panelB = Canvas(width=self.width,height=self.height)
        self.panelB.pack(side="bottom", padx=10, pady=10)

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # pass

    def _drawCube(self, cube, rvec=[], tvec=[]):
        img = self.img
        data = self.camData

        x, y, l, w, h, color, z = (cube.x, cube.y, cube.l, cube.w, cube.h, cube.c, cube.z)

        points = np.float32([
            [x, y, z],
            [x, y+l, z],
            [x+w, y+l, z],
            [x+w, y, z],
            [x, y, h+z],
            [x, y+l, h+z],
            [x+w, y+l, h+z],
            [x+w, y, h+z]
            ])

        isGUI = True
        if len(rvec) == 0: 
            rvec = data[0][self.iGround]
            isGUI = False
        if len(tvec) == 0: 
            tvec = data[1][self.iGround]

        imgpts, _ = cv2.projectPoints(points, rvec, tvec, cam.mtx, cam.dist)

        try:
            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, 3)

            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), color, 3)
            
            img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), color, 3)
            self.img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), color, 3)
        except:
            pass

    def _drawBlock(self, block, rvec=[], tvec=[]):
        img = self.img
        data = self.camData
        x, y, l, w, h, color, z = (block.x, block.y, block.l, block.w, block.h, block.c, block.z)

        if block.info['name'] == 'land' and not block.hovered and not (app.city.x == x and app.city.y == y):
            return
        
        thickness = 3
        #if block.selected:
        if app.city.x == x and app.city.y == y:
            color = (255,255,255)
            thickness = 5
        
        if block.hovered:
            color = (255, 0 ,0)

        points = np.float32([
            [x, y, z],
            [x, y + l, z],
            [x + w, y + l, z],
            [x + w, y, z],
            [x, y, h + z],
            [x, y + l, h + z],
            [x + w, y + l, h + z],
            [x + w, y, h + z]
        ])

        isGUI = True
        if len(rvec) == 0:
            rvec = data[0][self.iGround]
            isGUI = False
        if len(tvec) == 0:
            tvec = data[1][self.iGround]

        imgpts, _ = cv2.projectPoints(points, rvec, tvec, cam.mtx, cam.dist)

        try:
            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, thickness)

            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), color, thickness)

            img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), color, thickness)
            img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), color, thickness)
            self.img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), color, thickness)
        except:
            pass

    def _drawCubeTK(self, cube):

        x, y, l, w, h, color, z = (cube.x, cube.y, cube.l, cube.w, cube.h, cube.c, cube.z)
        p0,p1,p2,p3,p4,p5,p6,p7 = np.float32([
            [x, y, z],
            [x, y+l, z],
            [x+w, y+l, z],
            [x+w, y, z],
            [x, y, h+z],
            [x, y+l, h+z],
            [x+w, y+l, h+z],
            [x+w, y, h+z]
            ])

        p0 = self.xyz2xy(p0)
        p1 = self.xyz2xy(p1)
        p2 = self.xyz2xy(p2)
        p3 = self.xyz2xy(p3)
        p4 = self.xyz2xy(p4)
        p5 = self.xyz2xy(p5)
        p6 = self.xyz2xy(p6)
        p7 = self.xyz2xy(p7)

        self.panelB.create_line(p0[0], p0[1], p1[0],p1[1])
        self.panelB.create_line(p1[0], p1[1], p2[0],p2[1])
        self.panelB.create_line(p2[0], p2[1], p3[0],p3[1])
        self.panelB.create_line(p3[0], p3[1], p0[0],p0[1])

        self.panelB.create_line(p0[0], p0[1], p4[0],p4[1])
        self.panelB.create_line(p1[0], p1[1], p5[0],p5[1])
        self.panelB.create_line(p2[0], p2[1], p6[0],p6[1])
        self.panelB.create_line(p3[0], p3[1], p7[0],p7[1])

        self.panelB.create_line(p4[0], p4[1], p5[0],p5[1])
        self.panelB.create_line(p5[0], p5[1], p6[0],p6[1])
        self.panelB.create_line(p6[0], p6[1], p7[0],p7[1])
        self.panelB.create_line(p7[0], p7[1], p4[0],p4[1])

    def _drawSquare(self, sqr):
        img = self.img
        data = self.camData

        x, y, l, w, color = (sqr.x, sqr.y,1, 1, sqr.c)

        points = np.float32([
            [x, y, 0],
            [x, y+l, 0],
            [x+w, y+l, 0],
            [x+w, y, 0]
            ])

        imgpts, _ = cv2.projectPoints(points, data[0][self.iGround], data[1]
        [self.iGround], data[2], data[3])

        try:
            img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), color, 3)
            img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), color, 3)

            self.img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), color, 3)
        except:
            pass

    def _drawSquareTK(self, x,y):
        l = w = 1
        h = 0
        color = (255,0,0)
        z = 0
        p0,p1,p2,p3,p4,p5,p6,p7 = np.float32([
            [x, y, z],
            [x, y+l, z],
            [x+w, y+l, z],
            [x+w, y, z],
            [x, y, h+z],
            [x, y+l, h+z],
            [x+w, y+l, h+z],
            [x+w, y, h+z]
            ])

        p0 = self.xyz2xy(p0)
        p1 = self.xyz2xy(p1)
        p2 = self.xyz2xy(p2)
        p3 = self.xyz2xy(p3)
        p4 = self.xyz2xy(p4)
        p5 = self.xyz2xy(p5)
        p6 = self.xyz2xy(p6)
        p7 = self.xyz2xy(p7)

        self.panelB.create_line(p0[0], p0[1], p1[0],p1[1])
        self.panelB.create_line(p1[0], p1[1], p2[0],p2[1])
        self.panelB.create_line(p2[0], p2[1], p3[0],p3[1])
        self.panelB.create_line(p3[0], p3[1], p0[0],p0[1])

        self.panelB.create_line(p0[0], p0[1], p4[0],p4[1])
        self.panelB.create_line(p1[0], p1[1], p5[0],p5[1])
        self.panelB.create_line(p2[0], p2[1], p6[0],p6[1])
        self.panelB.create_line(p3[0], p3[1], p7[0],p7[1])

        self.panelB.create_line(p4[0], p4[1], p5[0],p5[1])
        self.panelB.create_line(p5[0], p5[1], p6[0],p6[1])
        self.panelB.create_line(p6[0], p6[1], p7[0],p7[1])
        self.panelB.create_line(p7[0], p7[1], p4[0],p4[1])

    def _drawGround(self):
        data = self.camData

        points = np.float32([
            [-1, -1, 0],
            [8, -1, 0],
            [8, 8, 0],
            [-1, 8, 0]
            ])

        imgpts, _ = cv2.projectPoints(points, data[0][self.iGround], data[1]
        [self.iGround], data[2], data[3])
        # Algorithm: https://docs.opencv.org/2.4/modules/core/doc/drawing_
        # functions.html
        pts = np.array([[p.ravel() for p in imgpts]], dtype=np.int32)
        self.img = cv2.fillPoly(self.img, pts, (50, 50, 50) )

    def _drawGroundTK(self):
        v0,v1,v2,v3 = np.float32([
            [-1, -1, 0],
            [8, -1, 0],
            [8, 8, 0],
            [-1, 8, 0]
            ])

        p0 = self.xyz2xy(v0)
        p1 = self.xyz2xy(v1)
        p2 = self.xyz2xy(v2)
        p3 = self.xyz2xy(v3)
        self.panelB.create_line(p0[0], p0[1], p1[0],p1[1])
        self.panelB.create_line(p1[0], p1[1], p2[0],p2[1])
        self.panelB.create_line(p2[0], p2[1], p3[0],p3[1])
        self.panelB.create_line(p3[0], p3[1], p0[0],p0[1])

    def _drawParameterPanelTk(self):
       
        cityRevenueText = city.totalRevenue
        cityRevenueText = format(cityRevenueText, '.2f')
        cityPopulationText = city.population
        cityPopulationText = int(cityPopulationText)
        cityPopulationCapacityText = city.populationCapcity
        cityPopulationCapacityText = int(cityPopulationCapacityText)
        cityScoreText = city.totalScore
        cityScoreText = format(cityScoreText, '.2f')
        cityLevel = city.cityLevel
        cityPrintText = city.printing
        
        block = city.getCurrentBlock()
        blockNameText = block.name
        blockRevenueText = block.getRevenue()
        blockRevenueText = format(blockRevenueText, '.2f')
        blockPopulationText = block.getPopulationCapacity()
        blockPopulationText = int(blockPopulationText)
        blockCostText = block.getCost()
        blockCostText = format(blockCostText, '.2f')
        blockScoreText = block.getScore()
        blockScoreText = format(blockScoreText, '.2f')
        blockPrintText = block.printing

        self.panelB.create_text(self.width-60, 10,fill="darkblue",font="Times 10 italic bold",
                        text='CityInfo:')
        self.panelB.create_text(self.width-60, 20,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Revenue', cityRevenueText))
        self.panelB.create_text(self.width-60, 30,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Population', cityPopulationText))
        self.panelB.create_text(self.width-60, 40,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('PopulationCapcity', cityPopulationCapacityText))
        self.panelB.create_text(self.width-60, 50,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Score', cityScoreText))
        self.panelB.create_text(self.width-60, 60,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Level', cityLevel))
        self.panelB.create_text(self.width//2, self.height-20,fill="darkblue",font="Times 15 italic bold",
                        text=cityPrintText)
        
        self.panelB.create_text(50, 10,fill="darkblue",font="Times 10 italic bold",
                        text='CurrentBlockInfo:')
        self.panelB.create_text(50, 20,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('BlockType', blockNameText))
        self.panelB.create_text(50, 30,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Revenue', blockRevenueText))
        self.panelB.create_text(50, 40,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Population', blockPopulationText))
        self.panelB.create_text(50, 50,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Cost', blockCostText))
        self.panelB.create_text(50, 60,fill="darkblue",font="Times 10 italic bold",
                        text="%s: %s" % ('Score', blockScoreText))
        self.panelB.create_text(self.width//2, self.height-20,fill="darkblue",font="Times 15 italic bold",
                        text=blockPrintText)

    def _drawHand(self):
        data = self.camData
        points = np.float32([
            [0, 0, 0],
            [0, 15, 0]
            ])

        imgpts, _ = cv2.projectPoints(points, data[0][self.iHand], 
        data[1][self.iHand], data[2], data[3])
        self.img = cv2.line(self.img, tuple(imgpts[0].ravel()), 
        tuple(imgpts[1].ravel()),  (150, 150, 250), 3)

        self.pointer = imgpts[1].ravel()
        points = np.float32([
            [0, 15, 0],
            [1, 14, 0]
            ])

        imgpts, _ = cv2.projectPoints(points, data[0][self.iHand], 
        data[1][self.iHand], data[2], data[3])
        self.img = cv2.line(self.img, tuple(imgpts[0].ravel()), 
        tuple(imgpts[1].ravel()),  (150, 150, 250), 3)

        points = np.float32([
            [0, 15, 0],
            [-1, 14, 0]
            ])

        imgpts, _ = cv2.projectPoints(points, data[0][self.iHand], 
        data[1][self.iHand], data[2], data[3])
        self.img = cv2.line(self.img, tuple(imgpts[0].ravel()), 
        tuple(imgpts[1].ravel()),  (150, 150, 250), 3)

    def _drawGUI(self):
        pass
        radius = 15

        for b in self.actionButtons:
            if b.vis:
                self.img = cv2.rectangle(self.img, (b.x, b.y), (b.x+b.w, b.y+b.h), color=b.c, thickness=3)
                self.img = cv2.putText(self.img, b.text, (b.x+b.w//6, b.y+(b.h*2)//3), cv2.FONT_HERSHEY_PLAIN, 1, b.c, thickness=2)
                if b.hovered:
                    self.img = cv2.fillPoly(self.img, np.array([[[b.x, b.y],[b.x+b.w,b.y], [b.x+b.w, b.y+b.h], [b.x, b.y+b.h]]]), b.c )
                    self.img = cv2.putText(self.img, b.text, (b.x+b.w//6, b.y+(b.h*2)//3), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), thickness=2)

    def updateButtons(self):
        for b in self.actionButtons:
            b.isHovered(self.pointer)

    def angles(self):
        x, y, z = self.lookAt - self.location
        az = np.arctan2(y, x)
        ay = np.arctan2(z, x)
        ax = np.arctan2(z, y)
        return ax, ay, az

    #Algorithm: (rx, ry, rz)https://www.youtube.com/watch?v=29Vw2zzeMRM
    def rx(self, v, angle):
        return (v[0],(cos(radians(angle))*v[1]) + ((-sin(radians(angle)))*v[2]),
         (sin(radians(angle))*v[1]) + ((cos(radians(angle)))*v[2]))

    def ry(self, v, angle):
        return (((cos(radians(angle)))*v[0]) + ((-sin(radians(angle)))*v[1]), 
        v[2], ((sin(radians(angle)))*v[0]) + ((cos(radians(angle)))*v[1]))

    def rz(self, v, angle):
        return (((cos(radians(angle)))*v[0]) + ((-sin(radians(angle)))*v[1]), 
        ((sin(radians(angle)))*v[0]) + ((cos(radians(angle)))*v[1]), v[2])

    def xcor(self, x, y):
        u = int(self.width/16)
        fl=0.15
        try:
            if (x<0): return (self.width/2)-(x/(y*fl))*(-1*u)
            else: return (self.width/2)+(x/(y*fl))*u
        except(ZeroDivisionError):return 0
    def ycor(self, z, y):
        u = int(self.width/16)
        fl=0.15
        try:
            if (z<0): return (self.height/2)-(z/(y*fl))*u
            else: return (self.height/2)+(z/(y*fl))*(-1*u)
        except(ZeroDivisionError):return 0

    def xyz2xy(self, v):
        u = int(self.width/16)
        fl=0.15
        ax, ay, az = self.angles()
        ax, ay, az = 40,0,0
        px = self.rx(v, ax)
        py = self.ry(px, ay)
        pz = self.rz(py, az)
        pp = [0,0,0]
        pp[0] =pz[0]+ self.location[0]
        pp[1] =pz[1]+ self.location[1]
        pp[2] =pz[2]+ self.location[2]
        p = (self.xcor(pp[0], pp[1]), self.ycor(pp[2], pp[1]))
        return p

    def drawAll(self, objects):
        self.panelB.delete('all')
        self._drawGUI()
        self._drawGroundTK()
        self._drawParameterPanelTk()

        for obj in objects:
            if isinstance(obj, Cube):
                self._drawCubeTK(obj)
            if isinstance(obj, Square):
                self._drawSquareTK(obj.x, obj.y)

        if self.camData:
            [rvec, tvec, mtx, dist, ids] = self.camData
            self.iHand = -1
            self.iGround = -1
            try:
                if ids[0][0] == 1:
                    self.iHand = 0
                elif ids[1][0] == 1:
                    self.iHand = 1
            except:
                pass

            try:
                if ids[0][0] == 2:
                    self.iGround = 0
                elif ids[1][0] == 2:
                    self.iGround = 1
            except:
                pass

            if self.iHand != -1:
                self._drawHand()
                self.updateButtons()
                self._drawGUI()

            if self.iGround != -1:
                self._drawGround()

                for obj in objects:
                    if isinstance(obj, Cube):
                        self._drawCube(obj)
                    if isinstance(obj, Square):
                        self._drawSquare(obj)
                    if isinstance(obj, Block):
                        self._drawBlock(obj)

            points = np.array([[2,0,100]], dtype=np.float32)

        img = cv2.resize(self.img, (self.width, self.height))
        img = np.stack([img[:,:,2],img[:,:,1],img[:,:,0]],2)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.panelA.config(image=img)
        self.panelA.image = img

class City(object):
    def __init__(self):
        self.nX = 7
        self.nY = 7
        self.map = []   # y first
        self.initCity()
        self.x = 0
        self.y = 0
        self.x0 = 0
        self.y0 = 0
        self.houseCountPor = 0
        self.apartmentCountPor = 0 
        self.officialCountPor = 0 
        self.totalBuildingNum = 0
        self.totalPor = 0
        self.population = 100
        self.populationCapcity = 500
        self.totalRevenue = 1000
        self.totalScore = 0
        self.cityLevel = 0
        self.gameOver = False
        self.printing = None

    def getBlocks(self):
        blocks = []
        for x in range(self.nX):
            for y in range(self.nY):
                blocks.append(self.map[x][y])
        return blocks
    
    def getCurrentBlock(self):
        currentBlock = self.map[self.x][self.y]
        return currentBlock

    def initCity(self):
        for x in range(self.nX):
            col = []
            for y in range(self.nY):
                block = Land(x, y, self)
                col.append(block)
            self.map.append(col)

    def neighborBlocks(self, qx, qy, distance=1):
        neighborBlocks = []
        xs = [qx-1, qx, qx+1]
        ys = [qy-1, qy, qy+1]
        for x in xs:
            if x < 0 or x >= self.nX:
                continue
            for y in ys:
                if y < 0 or y >= self.nY:
                    continue
                if x == qx or y == qy:
                    continue
                neighborBlocks.append(self.map[x][y])
        return neighborBlocks

    def numNeighborBlocks(self, qx, qy, blockType, distance=1):
        neighborBlocks = self.neighborBlocks(qx, qy, distance)
        num = 0
        for block in neighborBlocks:
            if isinstance(block, blockType):
                num += 1
        return num

    def blockOnCorner(self, qx, qy):
        if qx != 0 and qx != self.nX:
            return False
        if qy != 0 and qy != self.nY:
            return False
        return True

    def blockOnBorder(self, qx, qy):
        if self.blockOnCorner(qx, qy):
            return False
        if qx == 0 or qx == self.nX:
            return True
        if qy == 0 and qy == self.nY:
            return True
        return False

    def update(self):
        populationCapcity = 0
        totalRevenue = 0
        cost = 0
        score = 0
        for block in self.getBlocks():
            populationCapcity += block.getPopulationCapacity()
            totalRevenue += block.getRevenue()
            cost += block.getCost()
            score += block.getScore()
        self.populationCapcity = populationCapcity
        self.cost = cost
        self.totalRevenue = totalRevenue - cost
        self.totalScore = score
        # self.cityLevel = self.populationCapcity // 800

        #if self.cityLevel >= 3:
            # self.populationCapcity += self.cityLevel * 200 
            #if self.totalPor >= 2/3:
             #   self.population += 50
        print(self.population)
        self.population *= 1.02
        
        self.checkGameOver()
    
    def getDifBuildingPor(self):
        blocks = self.getBlocks()
        for block in neighborBlocks:
            if isinstance(block, House):
                houseCount += 1
            if isinstance(block, Apartment):
                apartmentCount += 1
            if isinstance(block, Official):
                officialCount += 1
        totalBuildingNum = houseCount + apartmentCount + officialCount
        self.houseCountPor = houseCount / totalBuildingNum
        self.apartmentCountPor = apartmentCount / totalBuildingNum
        self.officialCountPor = officialCount / totalBuildingNum
        self.totalPor = self.houseCountPor + self.apartmentCountPor + self.officialCountPor
        self.totalBuildingNum = totalBuildingNum
        return self.houseCountPor, self.apartmentCountPor, self.officialCountPor, self.totalBuildingNum
    
    def checkGameOver(self):
        if self.population >= self.populationCapcity:
            self.gameOver = True
            self.printing = 'GameOver'
        if self.population <= 10 or self.populationCapcity <= 50 or self.totalRevenue <= 100:
            self.gameOver = True
            print(self.population, self.populationCapcity, self.totalRevenue)
            self.printing = 'GameOver'
        pass
    
    def getBlock(self, x, y):
        if x < 0 or x >= self.nX or y < 0 or y >= self.nY:
            return False
        return self.map[x][y]

    def _clear(self):
        self.map = np.zeros([self.nY, self.nX])

    def checkSelection(self):
        #self.x = self.x0
        #self.y = self.y0
        if not app.cam.pHand is None:
            p = app.cam.pHand
            p0 = app.cam.pHand0
            for y in range(self.nY):
                for x in range(self.nX):
                    block = self.map[x][y]
                    block.isHovered(p0, p)
                    block.isSelected()
                    if block.selected:
                        break

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        if self.x < 0 or self.x >= self.nX:
            self.x -= dx
        if self.y < 0 or self.y >= self.nY:
            self.y -= dy
        self.x0 += dx
        self.y0 += dy
        if self.x0 < 0 or self.x0 >= self.nX:
            self.x0 -= dx
        if self.y0 < 0 or self.y0 >= self.nY:
            self.y0 -= dy

    def upgrade(self):
        if self.cityLevel <= 1:
            pass
        if self.totalScore <= 20:
            pass    
        if self.totalRevenue <= 0:
        	pass
        self.map[self.x][self.y].upgrade()

    def demolish(self):
        self.map[self.x][self.y] = Land(self.x, self.y, self)
        block = self.map[self.x][self.y]
        if isinstance(block, Block):
            print('hehehe')
        print(type(block))
        print(block.name)
        if block.name == 'land':
            print('heyhey')
            block.printing = 'Whoops, can not demolish anymore.'
        # self.map[self.x][self.y] = Land(self.x, self.y, self)
    
    def cancel(self):
        self.map[self.x][self.y].cancel()
    
    def build(self, t):
        thres = 1/3
        block = self.map[self.x][self.y]
        if self.houseCountPor > thres:
            print(self.houseCountPor)
            pass
        if self.apartmentCountPor > thres:
            pass
        if self.officialCountPor > thres:
            pass
        if self.totalRevenue <= 0:
        	pass
        if not isinstance(block, Land):
            block.printing = 'Heyy, I am already here.'
        if t == 'house' and isinstance(block, Land):
	        self.map[self.x][self.y] = House(self.x, self.y, self)
        if t == 'apartment' and isinstance(block, Land):
	        self.map[self.x][self.y] = Apartment(self.x, self.y, self)
        if t == 'official' and isinstance(block, Land):
	        self.map[self.x][self.y] = Official(self.x, self.y, self)

    def objects(self):
        objects = []

        for x in range(self.nX):
            for y in range(self.nY):
                block = self.map[x][y]
                objects.append(block)
                block.selected = False
                if x == self.x and y == self.y:
                    block.selected = True

                    square = Square(x, y)
                    objects.append(square)

                if not isinstance(block, Land):
                    cube = Cube(x, y, 1, 1, block.level, block.info['color'])
                    objects.append(cube)

        return objects

class Block(object):
    basicInfo = {
        'land' : {
            'name' : 'land',
            'color' : (255,255,255),
            'population' : 20,
            'revenue': 100,
            'cost': 0,
            'score': 0,
        },
        'house' : {
            'name' : 'house',
            'color' : (178, 179, 157),
            'population' : 10,
            'revenue': 200,
            'cost': 60,
            'score': 2
        },
        'apartment' : {
            'name' : 'apartment',
            'color' : (132, 100, 126),
            'population' : 40,
            'revenue': 700,
            'cost': 300,
            'score': 5
        },
        'official' : {
            'name' : 'official',
            'color' : (147,121,209),
            'population' : 50,
            'revenue' : 1000,
            'cost': 400,
            'score': 5
        },

    }
    maxLevel = 5

    def __init__(self, x, y, city, t='land', isHolding=False, level=0):
        self.level = level
        self.city = city
        self.info = Block.basicInfo[t]
        self.isHolding = False
        self.hovered = False
        self.selected = False
        

        self.x = x
        self.y = y
        self.w = 1
        self.l = 1
        self.h = 0
        self.c = self.info['color']
        self.z = 0
        self.printing = None
        self.count = 0
        self.leaveCount = 0

    def demolish(self):
        # isinstance(block, Land)
        if self.info['name'] == 'land':
            self.printing = 'Woops, can not demolish anymore.'
            return False
        self.level = 0
        self.h = 0
        self.info = Block.basicInfo['land']
        self.c = self.info['color']
  
    def build(self, t, x=0, y=0):
        # block = city.getCurrentBlock()
        # if isinstance(block, Land):
        if self.info['name'] != 'land':
            self.printing = 'Heyy, I am already here.'
            return False
        self.info = Block.basicInfo[t]
        self.c = self.info['color']
        self.level = 1
        self.h = 1
        self.x = x
        self.y = y

    def isSelected(self):
        if self.count == 0 and self.hovered:
            self.count = 1
        if self.count > 0 and self.hovered:
            self.count += 1
            self.leaveCount =0
            if self.count > 10:
                if not self.selected:
                    self.city.x = self.x
                    self.city.y = self.y
                self.selected = True
        if self.count > 0 and not self.hovered:
            self.leaveCount += 1
            if self.leaveCount > 3:
                self.count = 0
                self.selected = False

    def upgrade(self):
        if self.level == Block.maxLevel:
            self.printing = 'Heyy, I am already max.'
            return False
        self.level += 1
        self.h += 1
 
    def cancel(self):
        self.level -= 1
        self.h -= 1

    def isHovered(self, p0, p1):
        self.hovered = False
        c0 = np.array([self.x, self.y, self.z])
        c1 = np.array([self.x + self.w, self.y, self.z])
        c2 = np.array([self.x + self.w, self.y+self.l, self.z])
        c3 = np.array([self.x, self.y+self.l, self.z])
        if self.h == 0:
            p0 = p0.reshape(-1)
            p1 = p1.reshape(-1)
            vec = p1 - p0
            vecPC = c0 - p0
            vec0 = c1 - c0
            vec1 = c3 - c0
            normal = np.cross(vec0, vec1)
            distance = np.dot(normal, vecPC) / np.linalg.norm(normal)
            t = distance / np.linalg.norm(vec)
            # print(self.x, self.y, p1)
            p0 *= 0.2
            if self.x < p1[0] < self.x +1 and self.y < p1[1] < self.y + 1:
                # print('selected')
                self.hovered = True
                return True
            if t < 0 or t > 1:
                self.hovered = False
                return False
            pInter = p0 + t * vec
            # print(pInter)
            if self.x<pInter[0]<self.x+self.w and self.y<pInter[1]<self.y+self.h:
                print('intersected')
                return True
        else:
            if self.x < p1[0] <self.x+1 and self.y <p1[1] < self.y+1:
                if self.z < p1[2] < self.z + 1:
                    self.hovered = True
                    return True 

    # def hold(self):
    #     self.isHolding = True

    # def place(self):
    #     self.isHolding = False

class Land(Block):
    def __init__(self, x, y, city, t='land', isHolding=False, level=0):
        super().__init__(x, y, city, t, isHolding, level)
        info = Block.basicInfo['land']
        self.name = info['name']
        self.population = info['population']
        self.basicRevenue = info['revenue']
        self.basicCost = info['cost']
        self.score = info['score']

    def getRevenue(self):
        return self.basicRevenue

    def getPopulationCapacity(self):
        return self.population

    def getCost(self):
        return self.basicCost
    
    def getScore(self):
        return self.score

class House(Block):
    def __init__(self, x, y, city, t='land', isHolding=False, level=1):
        super().__init__(x, y, city, t, isHolding, level)
        info = Block.basicInfo['house']
        self.name = info['name']
        self.basicPopulation = info['population']
        self.basicRevenue = info['revenue']
        self.basicCost = info['cost']
        self.basicScore = info['score']

    def getRevenue(self):
        ws = []
        ws.append( self.level * 1.1 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.1)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) * 1.4)
        ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.1)
        revenue = self.basicRevenue
        for w in ws: 
            if w !=0:
                revenue *= w
        return revenue

    def getPopulationCapacity(self):
        ws = []
        ws.append( self.level * 1.3 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.1)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) * 1.4 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.5)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 2)
        if (self.city.numNeighborBlocks(self.x, self.y, House) >= 1 
        and self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.2)
        ws.append( 0.5 if self.city.blockOnCorner else 1.0 )
        ws.append( 0.8 if self.city.blockOnBorder else 1.0 )
        population = self.basicPopulation
        for w in ws: 
            if w !=0:
                population *= w
        return population

    def getCost(self):
        ws = []
        ws.append( self.level * 1.5 )
        cost = self.basicCost
        for w in ws: 
            if w !=0:
                cost *= w
        return cost
    
    def getScore(self):
        ws = []
        ws.append( self.level )
        if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 and 
        self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( 1.1 )
        # elif (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 or 
        # self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        # or self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
        #     ws.append( 1.1 )
        score = self.basicScore
        for w in ws: 
            if w !=0:
                score *= w
        return score

class Apartment(Block):
    def __init__(self, x, y, city, t='land', isHolding=False, level=1):
        super().__init__(x, y, city, t, isHolding, level)
        info = Block.basicInfo['apartment']
        self.name = info['name']
        self.population = info['population']
        self.basicRevenue = info['revenue']
        self.basicCost = info['cost']
        self.score = info['score']
    	
    def getRevenue(self):
        ws = []
        ws.append( self.level * 1.5 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.1)
        ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) * 1.3)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.3)
        ws.append( 0.5 if self.city.blockOnCorner else 1.0 )
        ws.append( 0.8 if self.city.blockOnBorder else 1.0 )
        revenue = self.basicRevenue
        for w in ws: 
            if w !=0:
                revenue *= w
        return revenue

    def getPopulationCapacity(self):
        ws = []
        ws.append( self.level * 1.5 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.3)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) ** 1.4 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) ** 1.2)
        if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 and self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.2)
        ws.append( 0.4 if self.city.blockOnCorner else 1.0 )
        ws.append( 0.6 if self.city.blockOnBorder else 1.0 )
        population = self.population
        for w in ws: 
            if w !=0:
                population *= w
        return population

    def getCost(self):
        ws = []
        ws.append( self.level * 1.1 )
        cost = self.basicCost
        for w in ws: 
            if w !=0:
                cost *= w
        return cost
    
    def getScore(self):
        ws = []
        ws.append( self.level * 1.1)
        if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 and 
        self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( 1.21 )
        # if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 or 
        # self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        # or self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
        #     ws.append( 1.05 )
        score = self.score
        for w in ws: 
            if w !=0:
                score *= w
        return score

class Official(Block):
    def __init__(self, x, y, city, t='land', isHolding=False, level=1):
        super().__init__(x, y, city, t, isHolding, level)
        info = Block.basicInfo['official']
        self.name = info['name']
        self.population = info['population']
        self.basicRevenue = info['revenue']
        self.basicCost = info['cost']
        self.score = info['score']
    	
    def getRevenue(self):
        ws = []
        ws.append( self.level * 1.4 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.1)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) * 2)
        ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 0.9)
        ws.append( 0.5 if self.city.blockOnCorner else 1.0 )
        ws.append( 0.8 if self.city.blockOnBorder else 1.0 )
        revenue = self.basicRevenue
        for w in ws: 
            if w !=0:
                revenue *= w
        return revenue

    def getPopulationCapacity(self):
        ws = []
        ws.append( self.level * 1.5 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, House) * 1.25)
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) ** 1.3 )
        # ws.append( self.city.numNeighborBlocks(self.x, self.y, Official) * 1.4)
        if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 and self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( self.city.numNeighborBlocks(self.x, self.y, Apartment) * 1.3)
        ws.append( 0.8 if self.city.blockOnCorner else 1.0 )
        ws.append( 0.9 if self.city.blockOnBorder else 1.0 )
        population = self.population
        for w in ws: 
            if w !=0:
                population *= w
        return population

    def getCost(self):
        ws = []
        ws.append( self.level * 1.4 )
        cost = self.basicCost
        for w in ws: 
            if w !=0:
                cost *= w
        return cost
    
    def getScore(self):
        ws = []
        ws.append( self.level * 1.1)
        if (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 and 
        self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        and self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( self.level * 1.2 )
        elif (self.city.numNeighborBlocks(self.x, self.y, House)>= 1 or 
        self.city.numNeighborBlocks(self.x, self.y, Apartment)>=1 
        or self.city.numNeighborBlocks(self.x, self.y, Official) >=1):
            ws.append( self.level * 1.18 )
        score = self.score
        for w in ws: 
            if w !=0:
                score *= w
        return score

class Cube(object):
    def __init__(self, x, y, w, l, h, c=(73, 75, 185), z=0):
        self.x = x
        self.y = y
        self.w = w
        self.l = l
        self.h = h
        self.c = c
        self.z = z

class Square(object):
    def __init__(self, x, y, c=(255, 255, 255)):
        self.x = x
        self.y = y
        self.c = c

class ButtonCV(object):
    def __init__(self, x,y,w,h, vis, devis, text="", c=(255,50,50)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.c = c
        self.text=text
        self.hovered = False
        self.vis = vis
        self.devis = devis
        self.clicked = False

    def isHovered(self, p):
        if p[0] > self.x and p[0] < self.x + self.w and p[1] > self.y and p[1] < self.y + self.h:
            hoveredNow = True
        else:
            hoveredNow = False

        if not self.hovered and hoveredNow:
            self.clicked = True
        self.hovered = hoveredNow

    def isClicked(self):
        if not self.hovered and not self.inTimeDelay():
            self.clicked = False
        else:
            self.clicked = True

    def inTimeDelay(self):
        period = -1
        self.count = 0
        if self.hovered:
            self.start = time()
            self.count = 1
        if self.count == 1 and not self.hovered:
            self.end = time()
            self.count = 0
            period = self.end - self.start
        return period <= 2000

class Contour(object):
    def __init__(self, points, color):
        self.points = points
        self.color = color

if __name__ == '__main__':
    cam = Camera()
    viewer = Viewer(cam)
    city = City()
    app = App(cam, viewer, city)


    app.run()
