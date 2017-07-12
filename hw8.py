import math
import numpy as np
from plot import plot, plot_trajectory, plot_covariance_2d

class UserCode:
    def __init__(self):
        # initialize with given beacons
        self.beacons = [Beacon(1.5, 0.5), Beacon(3.0, 0.5), Beacon(4.5, 0.5), Beacon(3.5, 2.0), Beacon(1.5, 3.5), 
                        Beacon(3.0, 3.5), Beacon(4.5, 3.5), Beacon(4.0, 5.5), Beacon(5.5, 5.5), Beacon(7.0, 5.5), 
                        Beacon(4.0, 7.0), Beacon(4.0, 8.5), Beacon(5.5, 8.5), Beacon(7.0, 8.5), Beacon(6.5, 11.0), 
                        Beacon(8.0, 11.0), Beacon(9.5, 11.0), Beacon(9.5, 9.5), Beacon(9.5, 12.5)]
        self.current_beacon = 0
        
        # state vector [x, y, yaw] in world coordinates
        self.state = State()
        self.state_desired = State()
        self.state_desired.position = np.array([1.5, 0])

        # 3x3 state covariance matrix
        self.sigma = 0.01 * np.identity(3) 
        
        # Gains for PD control
        Kp_xy = 3
        Kd_xy = 1
        self.Kp = np.array([[Kp_xy, Kp_xy]]).T
        self.Kd = np.array([[Kd_xy, Kd_xy]]).T
        self.Kp_psi = 0.5

        #process noise
        pos_noise_std = 0.005
        yaw_noise_std = 0.005
        self.Q = np.array([
            [pos_noise_std*pos_noise_std,0,0],
            [0,pos_noise_std*pos_noise_std,0],
            [0,0,yaw_noise_std*yaw_noise_std]
        ]) 
        
        #measurement noise
        z_pos_noise_std = 0.005
        z_yaw_noise_std = 0.03
        self.R = np.array([
            [z_pos_noise_std*z_pos_noise_std,0,0],
            [0,z_pos_noise_std*z_pos_noise_std,0],
            [0,0,z_yaw_noise_std*z_yaw_noise_std]
        ])

    def rotation(self, yaw):
        '''
        create 2D rotation matrix from given angle
        '''
        s_yaw = math.sin(yaw)
        c_yaw = math.cos(yaw)
                
        return np.array([
            [c_yaw, -s_yaw], 
            [s_yaw,  c_yaw]
        ])
    
    def normalizeYaw(self, y):
        '''
        normalizes the given angle to the interval [-pi, +pi]
        '''
        while(y > math.pi):
            y -= 2 * math.pi
        while(y < -math.pi):
            y += 2 * math.pi
        return y
    
    def visualizeState(self):
        # visualize position state
        plot_trajectory("kalman", self.state.position[0:2])
        plot_covariance_2d("kalman", self.sigma[0:2,0:2])
        
    def get_markers(self):
        '''
        place up to 30 markers in the world
        '''
        markers = [
             [0, 0], # marker at world position x = 0, y = 0
             [2, 0],  # marker at world position x = 2, y = 0
             [1.5, 0.5], [3.0, 0.5], [4.5, 0.5], [3.5, 2.0], [1.5, 3.5], 
             [3.0, 3.5], [4.5, 3.5], [4.0, 5.5], [5.5, 5.5], [7.0, 5.5], 
             [4.0, 7.0], [4.0, 8.5], [5.5, 8.5], [7.0, 8.5], [6.5, 11.0], 
             [8.0, 11.0], [9.5, 11.0], [9.5, 9.5], [9.5, 12.5]
        ]
        
        #TODO: Add your markers where needed
       
        return markers
        
    def predictState(self, dt, state, u_linear_velocity, u_yaw_velocity):
        '''
        predicts the next state using the current state and 
        the control inputs local linear velocity and yaw velocity
        '''
        state_prediction = State()
        state_prediction.position = state.position + dt * np.dot(self.rotation(state.yaw), u_linear_velocity)
        state_prediction.velocity = self.Kp * (self.state_desired.position - state.position) + self.Kd * (self.state_desired.velocity - state.velocity)
        state_prediction.yaw = state.yaw   + dt * u_yaw_velocity
        state_prediction.yaw = self.normalizeYaw(state_prediction.yaw)
        
        return state_prediction
    
    def calculatePredictStateJacobian(self, dt, state, u_linear_velocity, u_yaw_velocity):
        '''
        calculates the 3x3 Jacobian matrix for the predictState(...) function
        '''
        s_yaw = math.sin(state.yaw)
        c_yaw = math.cos(state.yaw)
        
        dRotation_dYaw = np.array([
            [-s_yaw, -c_yaw],
            [ c_yaw, -s_yaw]
        ])
        F = np.identity(3)
        F[0:2, 2] = dt * np.dot(dRotation_dYaw, u_linear_velocity)
        
        return F
    
    def predictCovariance(self, sigma, F, Q):
        '''
        predicts the next state covariance given the current covariance, 
        the Jacobian of the predictState(...) function F and the process noise Q
        '''
        return np.dot(F, np.dot(sigma, F.T)) + Q
    
    def calculateKalmanGain(self, sigma_p, H, R):
        '''
        calculates the Kalman gain
        '''
        return np.dot(np.dot(sigma_p, H.T), np.linalg.inv(np.dot(H, np.dot(sigma_p, H.T)) + R))
    
    def correctState(self, K, state_predicted, z, z_predicted):
        '''
        corrects the current state prediction using Kalman gain, the measurement and the predicted measurement
        
        :param K - Kalman gain
        :param x_predicted - predicted state 3x1 vector
        :param z - measurement 3x1 vector
        :param z_predicted - predicted measurement 3x1 vector
        :return corrected state as 3x1 vector
        '''
        residual = (z - z_predicted)
        residual[2] = self.normalizeYaw(residual[2])

        correction = np.dot(K, residual)
        state_predicted.position = state_predicted.position + correction[0:2]
        state_predicted.yaw = state_predicted.yaw + correction[2]
            
        return state_predicted
    
    def correctCovariance(self, sigma_p, K, H):
        '''
        corrects the sate covariance matrix using Kalman gain and the Jacobian matrix of the predictMeasurement(...) function
        '''
        return np.dot(np.identity(3) - np.dot(K, H), sigma_p)
    
    def predictMeasurement(self, state, marker_position_world, marker_yaw_world):
        '''
        predicts a marker measurement given the current state and the marker position and orientation in world coordinates 
        '''
        z_predicted = Pose2D(self.rotation(state.yaw), state.position).inv() * Pose2D(self.rotation(marker_yaw_world), marker_position_world);
        
        return np.array([[z_predicted.translation[0], z_predicted.translation[1], z_predicted.yaw()]]).T
    
    def calculatePredictMeasurementJacobian(self, state, marker_position_world, marker_yaw_world):
        '''
        calculates the 3x3 Jacobian matrix of the predictMeasurement(...) function using the current state and 
        the marker position and orientation in world coordinates
        
        :param x - current state 3x1 vector
        :param marker_position_world - x and y position of the marker in world coordinates 2x1 vector
        :param marker_yaw_world - orientation of the marker in world coordinates
        :return - 3x3 Jacobian matrix of the predictMeasurement(...) function
        '''
        s_yaw = math.sin(state.yaw)
        c_yaw = math.cos(state.yaw)
        
        dx = marker_position_world[0] - state.position[0];
        dy = marker_position_world[1] - state.position[1];
        
        return np.array([
            [-c_yaw, -s_yaw, -s_yaw * dx + c_yaw * dy],
            [ s_yaw, -c_yaw, -c_yaw * dx - s_yaw * dy],
            [     0,      0,                      -1]
        ])
        
    def state_callback(self, t, dt, linear_velocity, yaw_velocity):
        '''
        called when a new odometry measurement arrives approx. 200Hz
    
        :param t - simulation time
        :param dt - time difference this last invocation
        :param linear_velocity - x and y velocity in local quadrotor coordinate frame (independet of roll and pitch)
        :param yaw_velocity - velocity around quadrotor z axis (independet of roll and pitch)

        :return tuple containing linear x and y velocity control commands in local quadrotor coordinate frame (independet of roll and pitch), and yaw velocity
        '''
        self.state = self.predictState(dt, self.state, linear_velocity, yaw_velocity)
        
        F = self.calculatePredictStateJacobian(dt, self.state, linear_velocity, yaw_velocity)
        self.sigma = self.predictCovariance(self.sigma, F, self.Q);
        
        self.visualizeState()
    
        if (self.beacons[self.current_beacon].intersect(self.state.position[0], self.state.position[1])):
            self.beacons[self.current_beacon].activate()
            self.current_beacon += 1
            return np.zeros(2), 0
        
        if (self.current_beacon < len(self.beacons)):
            self.state_desired.position = np.array([self.beacons[self.current_beacon].x, self.beacons[self.current_beacon].y])

        limit = 10
        if self.state.velocity[0] > limit:
            self.state.velocity[0] = limit
        if self.state.velocity[0] < -limit:
            self.state.velocity[0] = -limit
        if self.state.velocity[1] > limit:
            self.state.velocity[1] = limit
        if self.state.velocity[1] < -limit:
            self.state.velocity[1] = -limit
        return self.state.velocity, -self.Kp_psi * self.state.yaw


    def measurement_callback(self, marker_position_world, marker_yaw_world, marker_position_relative, marker_yaw_relative):
        '''
        called when a new marker measurement arrives max 30Hz, marker measurements are only available if the quadrotor is
        sufficiently close to a marker
            
        :param marker_position_world - x and y position of the marker in world coordinates 2x1 vector
        :param marker_yaw_world - orientation of the marker in world coordinates
        :param marker_position_relative - x and y position of the marker relative to the quadrotor 2x1 vector
        :param marker_yaw_relative - orientation of the marker relative to the quadrotor
        '''
        z = np.array([[marker_position_relative[0], marker_position_relative[1], marker_yaw_relative]]).T
        z_predicted = self.predictMeasurement(self.state, marker_position_world, marker_yaw_world)
                
        H = self.calculatePredictMeasurementJacobian(self.state, marker_position_world, marker_yaw_world)
        K = self.calculateKalmanGain(self.sigma, H, self.R)
        
        self.state = self.correctState(K, self.state, z, z_predicted)
        self.sigma = self.correctCovariance(self.sigma, K, H)
        
        self.visualizeState()

    
class Pose2D:
    def __init__(self, rotation, translation):
        self.rotation = rotation
        self.translation = translation
        
    def inv(self):
        '''
        inversion of this Pose2D object
        
        :return - inverse of self
        '''
        inv_rotation = self.rotation.transpose()
        inv_translation = -np.dot(inv_rotation, self.translation)
        
        return Pose2D(inv_rotation, inv_translation)
    
    def yaw(self):
        from math import atan2
        return atan2(self.rotation[1,0], self.rotation[0,0])
        
    def __mul__(self, other):
        '''
        multiplication of two Pose2D objects, e.g.:
            a = Pose2D(...) # = self
            b = Pose2D(...) # = other
            c = a * b       # = return value
        
        :param other - Pose2D right hand side
        :return - product of self and other
        '''
        return Pose2D(np.dot(self.rotation, other.rotation), np.dot(self.rotation, other.translation) + self.translation)

class Beacon:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = False

    def activate(self):
        self.status = True

    def intersect(self, x, y):
        return (abs(x - self.x) < 0.3) and (abs(y - self.y) < 0.3)

class State:
    def __init__(self):
        self.position = np.zeros((2,1))
        self.velocity = np.zeros((2,1))
        self.yaw = 0
