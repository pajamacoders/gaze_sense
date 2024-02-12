from typing import Tuple
import numpy as np
import cv2
from coordinate_transformer import CoordinateTransformer
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
        self.visuzlier = VectorVisualizer()
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
            
    
    def estimate_gaze_point(self, recog_list:list, camera, display):
        self.image.fill(0)
        for face in recog_list.get('recog_face', []):#for each person
            for res in face:
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
                
                face_width_cm, face_height_cm = self.get_faze_size_using_sex_and_age(sex, age)
                est_depth_cm = self.coord_trans.estimate_depth(w, face_width_cm, camera['focal_length_x'])
                pts_xy_cam = self.coord_trans.transform_image_point_2_cam_point([cx,cy], camera)
                # R = self.coord_trans.get_rotation_matrix_from_rodrigues(roll, pitch, -yaw)
                R = self.coord_trans.get_rot_mat_in_lhd_coord_system_from_yaw_pitch_roll(yaw, pitch, roll)
                # R=np.linalg.inv(R)
                axes_points = np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                    ], dtype=np.float64)
                new_axes_pts = R@axes_points
                axis_length = int(0.4*w)
                
                # OY:center to right
                x1 = cx+int(R[1,1]*axis_length)
                y1 = cy+int(-R[2,1]*axis_length)
                cv2.line(self.image, (cx, cy), (x1,y1), (0,0,255),2)
                # OZ: center to top
                x1 = cx+int(R[1,2]*axis_length)
                y1 = cy+int(-R[2,2]*axis_length)
                cv2.line(self.image, (cx, cy), (x1,y1), (0,255,0),2)
                # OX: center to camera
                x1 = cx+int(R[1,0]*axis_length)
                y1 = cy+int(-R[2,0]*axis_length)
                cv2.line(self.image, (cx, cy), (x1,y1), (255,0,255),2)
                cv2.imshow('vector', self.image)
                cv2.waitKey(0)
                
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
            
            
    
    