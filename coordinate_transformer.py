import numpy as np
import cv2
class CoordinateTransformer:
    def __init__(self):
        pass
    
    def get_rotated_axes_using_given_rotation_matrix(self, axes_pts:np.ndarray,\
        R:np.ndarray)->np.ndarray:
        """
        rotate axes_point using given rotation matrix R.
        return the result.
        parameters:
        axes_points(np.ndarray): 3xn(n>=1) matrix. dtype=np.float64. a column correspond to a point.
        R(np.ndarray): rotation matrix. shape 3x3.dtype=np.float64
        
        return:
        np.ndarray: result.
        """
        rotated_axes_pts=np.dot(R, axes_pts)
        return rotated_axes_pts
        
        
    def get_rotation_matrix_from_rodrigues(self, roll:float, \
        pitch:float, yaw:float)->np.ndarray:
        """
        Receive roll, pitch, yaw as input and return rotation matrix that makes using
        rodrigues.
        
        parameters:
        roll(float): roll in degrees
        pitch(float): pitch in degrees
        yaw(float): yaw in degrees
        
        return:
        R(np.ndarray): 3x3 ndarray
        """
        rad_roll = np.deg2rad(roll)
        rad_pitch = np.deg2rad(pitch)
        rad_yaw = np.deg2rad(yaw)
        R = cv2.Rodrigues(np.array(rad_pitch, rad_yaw, rad_roll))[0].astype(np.float64)
        return R
    
    def get_rot_mat_in_lhd_coord_system_from_yaw_pitch_roll(self,\
        yaw:float, pitch:float, roll:float)->np.ndarray:
        """
        this function is for left hand coordinate system
        Receive yaw, pitch, yaw as input and return rotation matrix.
        
        parameters:
        yaw(float): roll in degrees
        pitch(float): pitch in degrees
        roll(float): yaw in degrees
        
        return:
        R(np.ndarray): 3x3 ndarray
        """
        rad_roll = np.deg2rad(roll)
        rad_pitch = np.deg2rad(pitch)
        rad_yaw = np.deg2rad(yaw)
        # x: pitch, y:yaw, z: roll                              
        cos_p = np.cos(rad_roll)
        cos_y = np.cos(rad_yaw)
        cos_r = np.cos(rad_pitch)
        sin_p = np.sin(rad_roll)
        sin_y = np.sin(rad_yaw)
        sin_r = np.sin(rad_pitch)
        """
         Rotation matrix:
         Yaw - counterclockwise Pitch - counterclockwise Roll - clockwise
             [cosY -sinY 0]          [ cosP 0 sinP]       [1    0    0 ]
             [sinY  cosY 0]    *     [  0   1  0  ]   *   [0  cosR sinR] =
             [  0    0   1]          [-sinP 0 cosP]       [0 -sinR cosR]
#          [cosY*cosP cosY*sinP*sinR-sinY*cosR cosY*sinP*cosR+sinY*sinR]
         = [sinY*cosP cosY*cosR-sinY*sinP*sinR sinY*sinP*cosR+cosY*sinR]
           [  -sinP          -cosP*sinR                cosP*cosR       ]
        """
        Rx = np.asarray([[1,      0,     0],
                         [0,  cos_r, sin_r], 
                         [0, -sin_r, cos_r]], dtype=np.float32)
        Ry = np.asarray([[cos_p,     0, sin_p],
                         [    0,     1,     0],
                         [-sin_p,    0, cos_p]], dtype=np.float32)
        Rz = np.asarray([[cos_y, -sin_y, 0],
                         [sin_y,  cos_y, 0], 
                         [    0,      0, 1]], dtype=np.float32)
        R = np.dot(np.dot(Rz, Ry), Rx)
        return R
    
    def get_rot_mat_in_rhd_system_from_roll_pitch_yaw(self, roll:float, \
        pitch:float, yaw:float)->np.ndarray:
        """
        this function is for right hand coordinate system
        Receive roll, pitch, yaw as input and return rotation matrix.
        
        parameters:
        roll(float): roll in degrees
        pitch(float): pitch in degrees
        yaw(float): yaw in degrees
        
        return:
        R(np.ndarray): 3x3 ndarray
        """
        rad_roll = np.deg2rad(roll)
        rad_pitch = np.deg2rad(pitch)
        rad_yaw = np.deg2rad(yaw)
        # x: pitch, y:yaw, z: roll                              
        cos_z = np.cos(rad_roll)
        cos_y = np.cos(rad_yaw)
        cos_x = np.cos(rad_pitch)
        sin_z = np.sin(rad_roll)
        sin_y = np.sin(rad_yaw)
        sin_x = np.sin(rad_pitch)
        
        Rx = np.asarray([[1,     0,      0],
                         [0, cos_x, -sin_x], 
                         [0, sin_x,  cos_x]], dtype=np.float32)
        Ry = np.asarray([[cos_y,     0, sin_y],
                         [    0,     1,     0],
                         [-sin_y,    0, cos_y]], dtype=np.float32)
        Rz = np.asarray([[cos_z, -sin_z, 0],
                         [sin_z,  cos_z, 0], 
                         [    0,      0, 1]], dtype=np.float32)
        R = np.dot(np.dot(Rz, Ry), Rx)
        return R