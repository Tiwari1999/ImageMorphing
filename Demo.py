#!/usr/bin/env python

import os
import cv2
import numpy as np
from numpy import matlib
from PIL import Image


def GetTransform(theta):
    lst = []
    cos = np.cos
    sin = np.sin
    for angle in theta:
        lst.append(np.array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]], dtype=np.float32))
    #return np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]], dtype=np.float32)
    return np.array(lst)

def Perpendicular(vec):
    tmp = np.zeros(vec.shape, dtype=np.float32)
    tmp[:, 0] = vec[:, 1]
    tmp[:, 1] = -vec[:, 0]
    s = np.sqrt(np.sum(tmp**2, axis=1).reshape([-1, 1]))
    s = matlib.repmat(s, 1, 2)
    #print s
    #tmp /= s
    return tmp

class Demo:

    def __init__(self, img1, img2, a, b, p):  # source, destination
        self.img1 = cv2.resize(cv2.imread(img1), (300, 450))
        self.img2 = cv2.resize(cv2.imread(img2), (300, 450))
        #cv2.imwrite('alu_1.jpg', self.img1)
        #cv2.imwrite('alu_2.jpg', self.img2)
        #exit()
        self.img1_ori = self.img1.copy()
        self.img2_ori = self.img2.copy()
        self._img1_name = img1
        self._img2_name = img2
        self._frame_count = 0
        self._a = a
        self._b = b
        self._p = p

        # [[init x y]], [[dest x y]]
        self.line_1 = [[], []]
        self.line_2 = [[], []]
        self.warpline = None

        self.draw_num_1 = 1
        self.init_y_1 = -1
        self.init_x_1 = -1

        self.draw_num_2 = 1
        self.init_y_2 = -1
        self.init_x_2 = -1
    
    def _GetABP(self):
        return self._a, self._b, self._p

    def _DrawLine_1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.draw_num_1 == 1:
                self.init_y_1 = y
                self.init_x_1 = x
                cv2.line(self.img1, (x, y), (x, y), (0, 0, 255), thickness=3)
                self.draw_num_1 += 1
            else:
                self.draw_num_1 = 1
                cv2.line(self.img1, (self.init_x_1, self.init_y_1),
                         (x, y), (0, 0, 255), thickness=3)
                self.line_1[0].append([self.init_x_1, self.init_y_1])
                self.line_1[1].append([x, y])

    def _DrawLine_2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.draw_num_2 == 1:
                self.init_x_2 = x
                self.init_y_2 = y
                cv2.line(self.img2, (x, y), (x, y), (0, 0, 255), thickness=3)
                self.draw_num_2 += 1
            else:
                self.draw_num_2 = 1
                cv2.line(self.img2, (self.init_x_2, self.init_y_2),
                         (x, y), (0, 0, 255), thickness=3)
                self.line_2[0].append([self.init_x_2, self.init_y_2])
                self.line_2[1].append([x, y])

    def _PlotFinish(self):
        if len(self.line_1[0]) != len(self.line_2[0]):
            print 'Line plot error, please plot it in same order.'
            exit()
        self.line_1[0] = np.array(self.line_1[0])
        self.line_1[1] = np.array(self.line_1[1])
        self.line_2[0] = np.array(self.line_2[0])
        self.line_2[1] = np.array(self.line_2[1])

    def _Morphing(self):
        #os.system('mkdir')
        os.system('rm out*.jpg')
        height = self.img1_ori.shape[0]
        width = self.img1_ori.shape[1]

        line_1 = self.line_1
        line_2 = self.line_2

        start_1_map = line_1[0]
        start_2_map = line_2[0]
        end_1_map = line_1[1]
        end_2_map = line_2[1]
        line_count = start_1_map.shape[0]

        loc_map = np.zeros([height, width, 2])
        loc_map[:,:,1] = matlib.repmat(np.arange(height).reshape([-1,1]), 1, width)
        loc_map[:,:,0] = matlib.repmat(np.arange(width), height, 1)
        loc_map = loc_map.reshape([width*height, 2])
        #print loc_map[:5]
        #print loc_map[-5:]
        #exit()
        img_lst = [self._img1_name]
        repmat = matlib.repmat
        [a,b,p] = self._GetABP()
        for frame_index in range(self._frame_count-1):
            start_1 = start_1_map.copy()
            end_1 = end_1_map.copy()
            start_2 = ((start_2_map-start_1_map).astype(np.float32)/(self._frame_count-1))*(frame_index+1) + start_1_map
            end_2 = ((end_2_map-end_1_map).astype(np.float32)/(self._frame_count-1))*(frame_index+1) + end_1_map

            # ratio for ori and dest image
            ratio = float(frame_index) / (self._frame_count + 1)
            X_pron_lst_1 = []
            W_lst_1 = []
            X_pron_lst_2 = []
            W_lst_2 = []
            for line_index in range(line_count):
                X = loc_map.copy()
                Q = repmat(end_2[line_index, :], width*height, 1)
                P = repmat(start_2[line_index, :], width*height, 1)
                X_sub_P = X - P
                Q_sub_P = Q - P
                Q_P_len_square = np.sum(Q_sub_P**2, axis=1)

                U = np.sum(X_sub_P * Q_sub_P, axis=1) / Q_P_len_square
                V = np.sum(X_sub_P * Perpendicular(Q_sub_P), axis=1) / np.sqrt(Q_P_len_square)
                
                Q_pron = repmat(end_1[line_index, :], width*height, 1)
                P_pron = repmat(start_1[line_index, :], width*height, 1)
                
                Q_sub_P_pron = (Q_pron - P_pron)
                Q_sub_P_pron_len = np.sqrt(np.sum(Q_sub_P_pron**2, axis=1))
                X_pron = P_pron + repmat(U.reshape([-1,1]), 1, 2) * Q_sub_P_pron \
                        + repmat(V.reshape([-1,1]), 1, 2) * Perpendicular(Q_sub_P_pron) \
                        / repmat(Q_sub_P_pron_len.reshape([-1,1]), 1, 2)

                X_P_pron_len = np.sqrt(np.sum((X_pron - P_pron)**2, axis=1))
                X_Q_pron_len = np.sqrt(np.sum((X_pron - Q_pron)**2, axis=1))

                W = np.zeros([width*height, 1], np.float32)
                i1 = U <= 1
                i2 = U < 0
                i3 = U > 1
                W[i1, :] = np.abs(V[i1]).reshape([-1, 1])
                W[i2, :] = X_P_pron_len[i2].reshape([-1, 1])
                W[i3, :] = X_Q_pron_len[i3].reshape([-1, 1])
                W = ((Q_sub_P_pron_len.reshape([-1,1]))**p / (a + W))**b
                
                X_pron_lst_1.append(X_pron)
                W_lst_1.append(W)

            start_1 = start_2_map.copy()
            end_1 = end_2_map.copy()
            start_2 = (((start_2_map-start_1_map).astype(np.float32)/(self._frame_count-1))*(frame_index+1) + start_1_map)
            end_2 = (((end_2_map-end_1_map).astype(np.float32)/(self._frame_count-1))*(frame_index+1) + end_1_map)

            for line_index in range(line_count):
                X = loc_map.copy()
                Q = repmat(end_2[line_index, :], width*height, 1)
                P = repmat(start_2[line_index, :], width*height, 1)
                X_sub_P = X - P
                Q_sub_P = Q - P
                Q_P_len_square = np.sum(Q_sub_P**2, axis=1)

                U = np.sum(X_sub_P * Q_sub_P, axis=1) / Q_P_len_square
                V = np.sum(X_sub_P * Perpendicular(Q_sub_P), axis=1) / np.sqrt(Q_P_len_square)
                
                Q_pron = repmat(end_1[line_index, :], width*height, 1)
                P_pron = repmat(start_1[line_index, :], width*height, 1)
                
                Q_sub_P_pron = Q_pron - P_pron
                Q_sub_P_pron_len = np.sqrt(np.sum(Q_sub_P_pron**2, axis=1))
                X_pron = P_pron + repmat(U.reshape([-1,1]), 1, 2) * Q_sub_P_pron \
                        + repmat(V.reshape([-1,1]), 1, 2) * Perpendicular(Q_sub_P_pron) \
                        / repmat(Q_sub_P_pron_len.reshape([-1,1]), 1, 2)

                X_P_pron_len = np.sqrt(np.sum((X_pron - P_pron)**2, axis=1))
                X_Q_pron_len = np.sqrt(np.sum((X_pron - Q_pron)**2, axis=1))

                W = np.zeros([width*height, 1], np.float32)
                i1 = U <= 1
                i2 = U < 0
                i3 = U > 1
                W[i1, :] = np.abs(V[i1]).reshape([-1, 1])
                W[i2, :] = X_P_pron_len[i2].reshape([-1, 1])
                W[i3, :] = X_Q_pron_len[i3].reshape([-1, 1])
                W = ((Q_sub_P_pron_len.reshape([-1,1]))**p / (a + W))**b
                
                X_pron_lst_2.append(X_pron)
                W_lst_2.append(W)

            X_sum_1 = np.zeros([width*height, 2], dtype=np.float32)
            W_sum_1 = np.zeros([width*height, 1], dtype=np.float32)
            for i, w in enumerate(W_lst_1):
                X_sum_1 += X_pron_lst_1[i] * repmat(w, 1, 2)
                W_sum_1 += w
            X_sum_2 = np.zeros([width*height, 2], dtype=np.float32)
            W_sum_2 = np.zeros([width*height, 1], dtype=np.float32)
            for i, w in enumerate(W_lst_2):
                X_sum_2 += X_pron_lst_2[i] * repmat(w, 1, 2)
                W_sum_2 += w

            X_final_1 = X_sum_1 / repmat(W_sum_1, 1, 2)
            X_final_1 = np.round(X_final_1).astype(np.int32)
            X_final_1[X_final_1[:, 0] >= width, 0] = width - 1
            X_final_1[X_final_1[:, 0] < 0, 0] = 0
            X_final_1[X_final_1[:, 1] >= height, 1] = height - 1
            X_final_1[X_final_1[:, 1] < 0, 1] = 0

            X_final_2 = X_sum_2 / repmat(W_sum_2, 1, 2)
            X_final_2 = np.round(X_final_2).astype(np.int32)
            X_final_2[X_final_2[:, 0] >= width, 0] = width - 1
            X_final_2[X_final_2[:, 0] < 0, 0] = 0
            X_final_2[X_final_2[:, 1] >= height, 1] = height - 1
            X_final_2[X_final_2[:, 1] < 0, 1] = 0
               
            source_img = self.img1_ori.astype(np.float32)
            dest_img = self.img2_ori.astype(np.float32)
            #img = np.zeros([height*width, 3], dtype=np.float32)
            #print source_img[X_final].shape
            img = (1-ratio) * source_img[X_final_1[:,1], X_final_1[:, 0], :].reshape([height*width,-1]) + (ratio) \
                    * dest_img[X_final_2[:,1], X_final_2[:, 0], :].reshape([height*width,-1])
            img = img.reshape([height, width, -1]).astype(np.uint8)
            name = 'out_%d.jpg'%frame_index
            cv2.imwrite(name, img)
            img_lst.append(name)
            #print np.max(W_sum)
            #print np.min(W_sum)
        img_lst.append(self._img2_name)
        s = ''
        for one in img_lst:
            s += '%s '%one
        command = 'convert -delay 15 -loop 0 %s result.gif'%s
        os.system(command)

    def Run(self, window1, window2, frames):
        self._frame_count = frames
        ploting = True
        cv2.namedWindow(window1)
        cv2.namedWindow(window2)
        cv2.setMouseCallback(window1, self._DrawLine_1)
        cv2.setMouseCallback(window2, self._DrawLine_2)
        while ploting:
            cv2.imshow(window1, self.img1)
            cv2.imshow(window2, self.img2)
            key = cv2.waitKey(33)
            if key == ord('q'):
                ploting = False
                self._PlotFinish()
                # print self.line_1
                # print self.line_2
                cv2.destroyAllWindows()
        self._Morphing()

test = Demo('alu_1.jpg', 'alu_2.jpg', 1, 2, 0)
test.Run('GG', 'TT', 15)
print 'origin data'
# print test.line_1
# print test.line_2
print 'process'
# print test.warpline[0]
# print test.warpline[-1]
