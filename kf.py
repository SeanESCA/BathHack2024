import math
import numpy as np


def convert_gyro(x, u):
    """
    Converts raw gyroscope measurements to the rate of change of roll, pitch, and yaw
    Args:
        x (np.array): Position of the IMU as a numpy array containing [roll, pitch, yaw]
        u (np.array): Raw measurements of the gyroscope as a numpy array [p, q, r]
    Returns:
        np.array: numpy array with [roll, pitch, yaw] rates of change
    """

    roll, pitch = x
    sin_roll = math.sin(roll)
    cos_roll = math.cos(roll)
    tan_pitch = math.tan(pitch)

    mat = np.array([
        [1, sin_roll * tan_pitch, cos_roll * tan_pitch],
        [0, cos_roll, -sin_roll]
    ])
    
    return mat@u
    
def h(x):
    '''
    Estimates the acceleration vector based on the current estimates of the current position of the IMU
    Args:
        x (np.array): Position of the IMU as a numpy array containing [roll, pitch, yaw]
    Returns:
        np.array: numpy array with [ax, ay, az]
    '''
    roll, pitch = x
    sin_roll = math.sin(roll)
    cos_roll = math.cos(roll)
    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)

    return 9.81 * np.array([
        sin_pitch,
        -cos_pitch*sin_roll,
        -cos_pitch*cos_roll
    ])


class ExtendedKalmanFilter:
    
    def __init__(self, x0, P0, Q, R):
        
        '''
        Q (np.array): Diagonal 3x3 matrix representing the covariance of the model.
        R (np.array): 
        '''
        
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        
    def predict(self, u, dt):
        
        '''
        u (np.array): Raw measurements of the gyroscope as a numpy array [p, q, r]
        dt (float): Seconds between gyroscope measurements.
        '''
        p, q, r = u

        sin_roll = math.sin(self.roll)
        cos_roll = math.cos(self.roll)
        sin_pitch = math.sin(self.pitch)
        cos_pitch = math.cos(self.pitch)
        tan_pitch = math.tan(self.pitch)

        A = np.array([
            [tan_pitch*(q*cos_roll - r*sin_roll), (tan_pitch**2 + 1)*(r*cos_roll + q*sin_roll), 0],
            [-q*sin_roll - r*cos_roll , 0, 0],
            [(q*cos_roll - r*sin_roll)/cos_pitch, (r*cos_roll + q*sin_roll)*sin_pitch*(tan_pitch**2 + 1), 0]
        ])
        
        rate_change_gyro = convert_gyro(self.x, u)
        predicted_x = self.x + dt*rate_change_gyro
        predicted_P = self.P + dt*(A@self.P + self.P@(A.T) + self.Q)
        
        self.x = predicted_x
        self.P = predicted_P
        
    def update(self, y):
        '''
        y (np.array): Raw measurements of the accelerometer as a numpy array [x, y, z]
        '''
    
        sin_roll = math.sin(self.roll)
        cos_roll = math.cos(self.roll)
        sin_pitch = math.sin(self.pitch)
        cos_pitch = math.cos(self.pitch)
        
        C = 9.81*np.array([
            [0, cos_pitch, 0],
            [-cos_roll*cos_pitch, sin_roll*sin_pitch, 0],
            [sin_roll*cos_pitch, cos_roll*sin_pitch, 0]
        ])
        
        K = self.P@(C.T)@np.linalg.inv(C@self.P@(C.T) + self.R)
        updated_x = self.x + K@(y - h(self.x))
        updated_P = (np.eye(3) - K@C)@self.P
        
        self.x = updated_x
        self.P = updated_P
        
    @property    
    def roll(self):
        
        return self.x[0]
    
    @property    
    def pitch(self):
        
        return self.x[1]
    
    @property    
    def yaw(self):
        
        return self.x[2]
    
    @property
    def cov(self):
        
        return self.P