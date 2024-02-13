from filterpy.kalman import KalmanFilter

import numpy as np 
class GazePointTracker:
    """
    state of gaze point tracker:
    state = [x,y,x',y']
    measurement = [x,y]
    F(StateTransition matrix)=[
        [1, 0, dt, 0],
        [0, 1,  0,dt],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
        ]
    
    H(measurement function)=[
        [1,0,0,0],
        [0,1,0,0]
    ]
    R(measurement noise) = [
        [2500, 0] # because model output mean error is almost 5.3 degree. tan(radian(5.3))*500(cm)
        [0, 2500]
    ]
    Q(process noise)=[
        [2500,    0,    0,    0],
        [   0, 2500,    0,    0],
        [   0,    0, 2500,    0],
        [   0,    0,    0, 2500]
    ]
    
    """
    def __init__(self, state_dim:int=4, measurement_dim:int=2):
        """
        initialize linear kalman tracker.
        Args:
            state_dim(int): Number of state variables for the Kalman filter. 
            measurement_dim(int): Number of of measurement inputs
        """
        self._is_initialized=False
        self._tracked_count = 0
        self.filter = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        
        self.filter.F = np.asarray([
            [1,   0, 0.1,   0],
            [0,   1,   0, 0.1],
            [0,   0,   1,   0],
            [0,   0,   0,   1]
        ])
        
        self.filter.H = np.asarray([
            [1,0,0,0],
            [0,1,0,0]
        ])
        self.filter.P*=2500.
        self.filter.Q = np.asarray([
            [100,   0,    0,    0],
            [   0, 100,    0,    0],
            [   0,   0, 2500,    0],
            [   0,   0,    0, 2500]
        ])
        
        self.filter.R = np.asarray([
            [2500,    0],
            [   0, 2500]
        ])
        
    def set_initialized(self):
        self._is_initialized=True
        
    def is_initialized(self):
        """Getter method for is_initialized"""
        return self._is_initialized
    
    def get_tracked_count(self):
        return self._tracked_count
    
    def increate_tracked_count(self):
        self._tracked_count+=1
        
    def initialize_state(self, gaze_point:np.ndarray):
        """
        initialize tracker state.
        [x,y,x',y']. initially x' and y' will be 0 and updated.
        Args:
            gaze_point(np.ndarray): estimated gaze point, [gaze_x, gaze_y, 0]
        """
        self.filter.x[:2]=gaze_point[:-1].reshape(2,1)
    
    def get_state_and_covariance(self):
        """
        Return current state and covariance.
        Returns:
            x: updated state, [x,y]
            P: state covariance 
            [ [cov_xx, cov_xy], 
              [cov_yx, cov_yy]]
        """ 
        if self._is_initialized:
            return self.filter.x[:2], self.filter.P[:2,:2]
        else:
            return None, None
    
    def predict(self):
        self.filter.predict()
    
    def update(self, gaze_point:np.ndarray):
        """
        perform measurement update using estimated gaze_point.
        Args:
            gaze_point(np.ndarray): gaze_point(np.ndarray): estimated gaze point, [gaze_x, gaze_y, 0]
        Returns:
            x: updated state, [x,y]
            P: state covariance 
            [ [cov_xx, cov_xy], 
              [cov_yx, cov_yy]]
        """
        self.filter.update(gaze_point[:-1])
        self.increate_tracked_count()
        