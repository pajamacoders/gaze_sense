"""
In this work based on the output from openvino ead-pose-estimation-adas-0001.
link: https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/head-pose-estimation-adas-0001
Required yaw, pitch, roll angles follow below description.
The estimator outputs yaw pitch and roll angles measured in degrees. Suppose the following coordinate system:
OX points from face center to camera
OY points from face center to right
OZ points from face center to up
The predicted angles show how the face is rotated according to a rotation matrix:
Yaw - counterclockwise Pitch - counterclockwise Roll - clockwise
    [cosY -sinY 0]          [ cosP 0 sinP]       [1    0    0 ]   [cosY*cosP cosY*sinP*sinR-sinY*cosR cosY*sinP*cosR+sinY*sinR]
    [sinY  cosY 0]    *     [  0   1  0  ]   *   [0  cosR sinR] = [sinY*cosP cosY*cosR-sinY*sinP*sinR sinY*sinP*cosR+cosY*sinR]
    [  0    0   1]          [-sinP 0 cosP]       [0 -sinR cosR]   [  -sinP          -cosP*sinR                cosP*cosR       ]
"""
from typing import Tuple
import numpy as np
import cv2
from tqdm import tqdm
from coordinate_transformer import CoordinateTransformer
from gaze_point_tracker import GazePointTracker
from vector_visualizer import VectorVisualizer

class GazeEstimator:
    
    def __init__(self):
        """
        young_{male/female}: 10~19
        adult_{male/female}: 20~69
        size information from "sizekorea.kr" of koean agency for technology and standards.
        8'th 
        height: 눈살 - 턱끝 길이. frown to tip chin.
        width:
        self.avg_face_size={
        'target_width':[avg(cm), std(cm)],
        'target_height':[avg(cm), std(cm)].
        }
        
        """
        self.visualizer = VectorVisualizer()
        self.image = np.zeros((1080,1920,3), dtype=np.uint8)
        # focal length = sensor size/(2xtan(fov))
        # nutz udh2160 sensor:
        # horizontal fov: 98 degree
        # sony cmos: model(IMX267LQR)
        # number of recommended recording pixels: 4096(h)x2160(v)
        # unit cell size: 3.45um(h) x 3.45um(v)
        # image size: diagonal 16.1mm(type 1)==12.8mm(h)x9.3mm(v)
        # focal length = 12.8mm/(2*np.tan(rad(49)))=5.56mm
        # focal length(pixel) = 5.56mm*(1000um/mm)*(pixel/3.45um)
        #                     = 1612 pixel
        
        
        self.coord_trans = CoordinateTransformer()
        if 1:#face width
            self.avg_face_size={'young_M_face_width': [13.77,  0.81], #age:10~19, avg, std in mm
                                'young_M_face_height':[12.50,  1.26], # frown to tip chin, 눈살-턱끝
                                'adult_M_face_width': [14.55,  0.88],
                                'adult_M_face_height':[13.60,  0.60],
                                'young_F_face_width': [13.16,  0.81], #age:10~19, avg, std in mm
                                'young_F_face_height':[11.82,  0.94], # frown to tip chin, 눈살-턱끝
                                'adult_F_face_width': [13.34,  1.11],
                                'adult_F_face_height':[12.88,  0.57],
                                }
        else: # head width
            self.avg_face_size={'young_M_face_width': [15.76,  0.68], #age:10~19, avg, std in mm
                                'young_M_face_height':[12.50,  1.25], # frown to tip chin, 눈살-턱끝
                                'adult_M_face_width': [16.49,  0.75],
                                'adult_M_face_height':[13.60,  0.60],
                                'young_F_face_width': [15.16,  0.59], #age:10~19, avg, std in mm
                                'young_F_face_height':[11.82,  0.94], # frown to tip chin, 눈살-턱끝
                                'adult_F_face_width': [15.63,  0.67],
                                'adult_F_face_height':[12.88,  0.57],
                                }
            
    def estimation_gaze_point(self, head_position:np.ndarray, dir_vector:np.ndarray):
        """
        Estimated gaze point coordinate in camera xy plane.
        Plane equation in 3d space:
        ax+by+cz+d=0, where (a,b,c) is normal vector. In our case same with (0,0,1) z-axis of camera coordinate frame.
        Line equeation in 3d space(Given normalized direction vector (A,B,C) and a point on a line)
        * x = xh + At
        * y = yh + Bt
        * z = zh + Ct
        where (xh,yh,zh) is head position in camera coordinate frame and (A,B,C) is direction vector
        
        Because we know z=0.
        0 = zh + Ct
        -zh/C = t
        x = xh + A(-zh/C) 
        y = yh + B(-zh/C) 
        
        Args:
            head_position(np.ndarray): position of head in camera coordinate frame. [x_cam, y_cam, depth]
            dir_vector(np.ndarray): gaze direction. It is same with first column vector of camera to face rotation matrix.
        Returns:
            np.ndarray: 1x3 array, [x,y,z] which means the estimated gaze point.
        """
        if dir_vector[-1]==0: # it cause divide by zero
            return None
        xh,yh,zh = head_position
        A,B,C = dir_vector
        t = -zh/C
        x = xh + A*t
        y = yh + B*t
        return np.asarray([x,y,0], dtype=np.float32)
        
        
        
        
    
    def estimate_gaze_from_head_pose(self, recog_list:list, camera, display):
        self.image.fill(0)
        for i, face in enumerate(recog_list.get('recog_face', [])):#for each person
            tracker = GazePointTracker(4,2)
            for j, res in enumerate(face):
                all_keys_exist = self.check_if_required_key_exist(res)
                if not all_keys_exist: # if any of the keys are missing, skip.
                    continue
                yaw, pitch, roll = res['GazeInfo'][:-1] # this include yaw, pitch, roll value in degrees
                bbox = res['bbox']
                age = res['age']
                sex = res['sex']
                cx,cy,w,h = bbox
                lx,ty = cx-w//2, cy-h//2
                rx,by = cx+w//2, cy+h//2
                
                #compute head position in camera coordinate frame
                face_width_cm, face_height_cm = self.get_faze_size_using_sex_and_age(sex, age)
                est_depth_cm = self.coord_trans.estimate_depth(w, face_width_cm, camera['focal_length_x'])
                head_pos_in_cam = self.coord_trans.transform_image_point_2_cam_point([cx,cy], camera)
                head_pos_in_cam = np.array(head_pos_in_cam)*est_depth_cm

                # compute rotation matrix from camera coordinate frame to face coordinate frame.
                # the first column of rotation matrix from camear to face coordinate frame
                # will be the gaze direction
                
                # R is rotation matrix from idea face coordinate frame to real face coordinate frame.
                Rb2f = self.coord_trans.get_rot_mat_in_lhd_coord_system_from_yaw_pitch_roll(yaw, pitch, -roll)
                # Rc2b is camera to ideal face coordinate frame rotation matrix.
                Rc2b= self.coord_trans.get_rot_mat_in_lhd_coord_system_from_yaw_pitch_roll(-180, 90,90)
                Rc2f=Rb2f@Rc2b
                gaze_pt = self.estimation_gaze_point(head_pos_in_cam, Rc2f[:,0])
                
                if gaze_pt is None: 
                    continue
                
                if tracker.is_initialized():
                    tracker.predict()
                    tracker.update(gaze_pt)
                else:
                    tracker.initialize_state(gaze_pt)
                    tracker.set_initialized()
                    
                if 0:# this section is for debugging. It draw head orientation in image.
                    axis_length = int(0.4*w)
                    # OY:center to right
                    x1 = cx+int(Rb2f[1,1]*axis_length)
                    y1 = cy+int(-Rb2f[2,1]*axis_length)
                    cv2.line(self.image, (cx, cy), (x1,y1), (0,0,255),2)
                    # OZ: center to top
                    x1 = cx+int(Rb2f[1,2]*axis_length)
                    y1 = cy+int(-Rb2f[2,2]*axis_length)
                    cv2.line(self.image, (cx, cy), (x1,y1), (0,255,0),2)
                    # OX: center to camera
                    x1 = cx+int(Rb2f[1,0]*axis_length)
                    y1 = cy+int(-Rb2f[2,0]*axis_length)
                    cv2.line(self.image, (cx, cy), (x1,y1), (255,0,255),2)
                    cv2.imshow('vector', self.image)
                    cv2.waitKey(0)
                    
            x, P = tracker.get_state_and_covariance()
            if x is None or P is None:
                #handle exception
                print('something_wrong')
            else:
                self.visualizer.draw_gaze_point_and_uncertainty(x,P)
                self.visualizer.add_point_to_figure(np.array([*x.reshape(-1),0]))
                self.visualizer.add_vector_in_figure(head_pos_in_cam, Rc2f[:,0]*20, Rc2f[:,1]*20, Rc2f[:,2]*20, 'f')
                self.visualizer.show_figure()
                # print(P)
                
    def check_if_required_key_exist(self, res:dict)->bool:
        """
        check if all predifined keys exist in res.
        return True if all keys exist in res, return False otherwise.
        """
        if all(key in res for key in ('GazeInfo', 'bbox', 'sex', 'age')):
            return True
        else:
            return False
    
    def get_faze_size_using_sex_and_age(self, sex:str, age:int)->Tuple[list, list]:
        """
        Receive sex and age as input and return corresponding face width and height infor
        face_width: [average width(mm), width std(mm)]
        face_height: [average height(mm), height std(mm)]
        
        parameters:
        sex(string): M or F
        age(int): age of target
        
        return: 
        face_width, face_height. ex) face_width=[avg, std] in mm,
        """
        
        assert isinstance(age, int), "age must be int type"
        assert isinstance(sex, int), "sex must be int type"
        age_grp='adult'
        if age<20:
            age_grp='young'
        
        gender = 'M'
        if sex!=0:
            gender='F'
            
        key_width = f'{age_grp}_{gender}_face_width'
        key_height = f'{age_grp}_{gender}_face_height'
        
        return self.avg_face_size[key_width], self.avg_face_size[key_height]
            
            
    
    