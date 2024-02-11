from typing import Tuple
from coordinate_transformer import CoordinateTransformer
import numpy as np

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
        'target_width':[avg(mm), std(mm)],
        'target_height':[avg(mm), std(mm)].
        }
        
        
        
        """
        self.coord_trans = CoordinateTransformer()
        if 1:#face width
            self.avg_face_size={'young_M_face_width':[137.74,8.06], #age:10~19, avg, std in mm
                                'young_M_face_height':[124.95,12.56], # frown to tip chin, 눈살-턱끝
                                'adult_M_face_width':[145.5, 8.84],
                                'adult_M_face_height':[136.03, 5.97],
                                'young_F_face_width':[131.63,8.09], #age:10~19, avg, std in mm
                                'young_F_face_height':[118.16, 9.42], # frown to tip chin, 눈살-턱끝
                                'adult_F_face_width':[133.41, 11.13],
                                'adult_F_face_height':[128.79, 5.69],
                                }
        else: # head width
            self.avg_face_size={'young_M_face_width':[157.56, 6.84], #age:10~19, avg, std in mm
                                'young_M_face_height':[124.95,12.56], # frown to tip chin, 눈살-턱끝
                                'adult_M_face_width':[164.94, 7.47],
                                'adult_M_face_height':[136.03, 5.97],
                                'young_F_face_width':[151.55, 5.93], #age:10~19, avg, std in mm
                                'young_F_face_height':[118.16, 9.42], # frown to tip chin, 눈살-턱끝
                                'adult_F_face_width':[156.32, 6.7],
                                'adult_F_face_height':[128.79, 5.69],
                                }
            
    
    def estimate_gaze_point(self, face_list:list, camera, display):
        for face in face_list:
            for res in face:
                all_keys_exist = self.check_if_required_key_exist(res)
                if not all_keys_exist: # if any of the keys are missing, skip.
                    continue
                yaw, pitch, roll = res['GazeInfo'] # this include yaw, pitch, roll value in degrees
                bbox = res['bbox']
                age = res['age']
                gender = res['sex']
                
                # R = self.coord_trans.get_rotation_matrix_from_rodrigues(roll, pitch, -yaw)
                R = self.coord_trans.get_rot_mat_in_lhd_coord_system_from_yaw_pitch_roll(yaw, pitch, roll)
                axes_points = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                    ], dtype=np.float64)
                
                
                
                
                
      
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
        assert isinstance(sex, str), "sex must be str type"
        age_grp='adult'
        if age<20:
            age_grp='young'
            
        key_width = f'{age_grp}_{sex}_face_width'
        key_height = f'{age_grp}_{sex}_face_height'
        
        return self.avg_face_size[key_width], self.avg_face_size[key_height]
            
            
    