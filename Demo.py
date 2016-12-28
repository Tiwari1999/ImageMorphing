#!/usr/bin/env python

import cv2
import numpy as np

def GetTransform(theta):
    cos = np.cos
    sin = np.sin
    return np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]], dtype=np.float32)
class Demo:
    def __init__(self, img1, img2): # source, destination
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)
        self.img1_ori = self.img1.copy()
        self.img2_ori = self.img2.copy()
        
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
        if len(self.line_1[0]) != len(self.line_2[0]):
            print 'Line plot error, please plot it in same order.'
            exit()
        self.line_1[0] = np.array(self.line_1[0])
        self.line_1[1] = np.array(self.line_1[1])
        self.line_2[0] = np.array(self.line_2[0])
        self.line_2[1] = np.array(self.line_2[1])

    def Run(self, window1, window2, frames):
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
        self.LineInterp(frames)
        self.Morphing(frames)
    def LineInterp(self, frames):
        line1 = self.line_1
        line2 = self.line_2
        total_line = line1[0].shape[0]

        line1_start = line1[0] # will be total_line * 2, which contain [y, x]
        line1_end = line1[1]
        line2_start = line2[0]
        line2_end = line2[1]

        #print line1_start
        #print line2_start
        #print line1_end
        #print line2_end
        interp_buf_start_x = np.zeros([total_line, frames])
        interp_buf_start_y = np.zeros([total_line, frames])
        interp_buf_end_x = np.zeros([total_line, frames])
        interp_buf_end_y = np.zeros([total_line, frames])

        for i in range(total_line):
            [start1_y, start1_x] = line1_start[i,:]
            [start2_y, start2_x] = line2_start[i,:]
            [end1_y, end1_x] = line1_end[i,:]
            [end2_y, end2_x] = line2_end[i,:]

            interp_buf_start_x[i, :] = np.round(np.linspace(start1_x, start2_x, frames)).astype(np.int32)
            interp_buf_start_y[i, :] = np.round(np.linspace(start1_y, start2_y, frames)).astype(np.int32)
            interp_buf_end_x[i, :] = np.round(np.linspace(end1_x, end2_x, frames)).astype(np.int32)
            interp_buf_end_y[i, :] = np.round(np.linspace(end1_y, end2_y, frames)).astype(np.int32)
        #print interp_buf_start_x
        #print interp_buf_start_y
        #print interp_buf_end_x
        #print interp_buf_end_y

        #print interp_buf_start_x[:, 0]

        #[np.zeros([total_line, 2]), np.zeros([total_line, 2])]
        self.warpline = [[np.zeros([total_line, 2]), np.zeros([total_line, 2])] for x in range(frames)]
        for frame in range(frames):
            self.warpline[frame][0][:, 1] = interp_buf_start_y[:, frame]
            self.warpline[frame][0][:, 0] = interp_buf_start_x[:, frame]
            self.warpline[frame][1][:, 1] = interp_buf_end_y[:, frame]
            self.warpline[frame][1][:, 0] = interp_buf_end_x[:, frame]
        #print self.warpline[0][0]
        #print self.warpline[-1][0]
        #print self.warpline

    def Morphing(self, frames):
        [height, width, _] = self.img1.shape
        total_line = self.line_1[0].shape[0]
        h = np.arange(height, dtype=np.float32)
        w = np.arange(width, dtype=np.float32)
        [x, y] = np.meshgrid(w, h)
        #print x
        #print y
        pixel_map = np.zeros([height, width, 2], dtype=np.float32)
        pixel_map[:,:, 0] = x
        pixel_map[:,:, 1] = y
        pixel_map = pixel_map.reshape([height*width, 2]).T
        #print pixel_map
        
        for frame in range(1, frames):
            warpline_dest = self.warpline[frame]
            warpline_source = self.warpline[frame-1] 
            
            P_map_dest = warpline_dest[0]
            Q_map_dest = warpline_dest[1]

            P_map_source = warpline_source[0]
            Q_map_source = warpline_source[1]

            vec_map_dest = Q_map_dest - P_map_dest
            vec_map_source = Q_map_source - P_map_source 

            len_map_dest = np.linalg.norm(vec_map_dest, axis=1)
            len_map_source = np.linalg.norm(vec_map_source, axis=1)

            img = np.zeros([height, width], dtype=np.float32)
            
            for line in range(total_line):
                cos_theta = np.dot(vec_map_source[line], vec_map_dest[line])/(len_map_dest[line]*len_map_source[line])
                theta = np.arccos(cos_theta)
                T = GetTransform(theta)
                pt_mapping = np.dot(T, pixel_map)
                print pt_mapping
            '''     
            for x in range(width):
                for y in range(height):
                    Weight = np.zeros([1, total_line], dtype=np.float32)
                    for line in range(total_line):
                        PX = pixel_map[y, x] - P_map_dest[line]
                        PQ = vec_map_dest[line]
                        u = np.abs(np.dot(PX, PQ) / (len_map_dest[line]))
                        v = np.sqrt(np.linalg.norm(PX)**2 - u**2)
                        u /= len_map_dest[line]
                        [new_x, new_y] = P_map_source[line] + u*(vec_map_source[line]) \
                        + v
                        #exit()
            '''
test = Demo('image.jpg', 'depth.png')
test.Run('GG', 'TT', 5)
print 'origin data'
#print test.line_1
#print test.line_2
print 'process'
#print test.warpline[0]
#print test.warpline[-1]

