#!/usr/bin/env python

import cv2
import numpy as np

class Demo:
    def __init__(self, img1, img2):
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)
        self.img1_ori = self.img1.copy()
        self.img2_ori = self.img2.copy()
        
        # init x y, dest x y
        self.line_1 = [[], []]
        self.line_2 = [[], []]

        self.draw_num_1 = 1
        self.init_y_1 = -1
        self.init_x_1 = -1

        self.draw_num_2 = 1
        self.init_y_2 = -1
        self.init_x_2 = -1

    def DrawLine_1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.draw_num_1 == 1:
                self.init_y_1 = y
                self.init_x_1 = x
                cv2.line(self.img1, (x, y), (x, y), (0, 0, 255), thickness=3)
                self.draw_num_1 += 1
            else:
                self.draw_num_1 = 1
                cv2.line(self.img1, (self.init_x_1, self.init_y_1), (x, y), (0, 0, 255), thickness=3)
                self.line_1[0].append([self.init_y_1, self.init_x_1])
                self.line_1[1].append([y, x])

    def DrawLine_2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.draw_num_2 == 1:
                self.init_x_2 = x
                self.init_y_2 = y
                cv2.line(self.img2, (x, y), (x, y), (0, 0, 255), thickness=3)
                self.draw_num_2 += 1
            else:
                self.draw_num_2 = 1
                cv2.line(self.img2, (self.init_x_2, self.init_y_2), (x, y), (0, 0, 255), thickness=3)
                self.line_2[0].append([self.init_y_2, self.init_x_2])
                self.line_2[1].append([y, x])

    def PlotFinish(self):
        if len(self.line_1) != len(self.line_2):
            print 'Line plot error, please plot it in same order.'
        self.line_1 = np.array(self.line_1)
        self.line_2 = np.array(self.line_2)

    def Run(self, window1, window2):
        ploting = True
        cv2.namedWindow(window1) 
        cv2.namedWindow(window2) 
        cv2.setMouseCallback(window1, self.DrawLine_1)
        cv2.setMouseCallback(window2, self.DrawLine_2)
        while ploting:
            cv2.imshow(window1, self.img1)
            cv2.imshow(window2, self.img2)
            key = cv2.waitKey(33)            
            if key == ord('q'):
                ploting = False
                self.PlotFinish()
                #print self.line_1
                #print self.line_2
                cv2.destroyAllWindows()

test = Demo('image.jpg', 'depth.png')
test.Run('GG', 'TT')
print test.line_1[0]
print test.line_1[1]

