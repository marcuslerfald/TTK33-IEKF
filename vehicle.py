from math import sin, cos
import numpy as np
    
class Vehicle:
    def __init__(self, x=np.array([0,0,0,0]), u=np.array([0,0]), T=0.01, measurement_stds=[0.1,0.1,0.01,0.02]):
        # x: x, y, theta, v
        # u: a, omega
        self.T = T
        self.measurement_stds = measurement_stds
        
        self.x_data = []
        self.u_data = []

        self.update_vehicle(x, u)

    def update_vehicle(self, x, u):
        self.x = x
        self.u = u
        self.x_data = np.vstack((self.x_data, x))
        self.u_data = np.vstack((self.u_data, u))

    def f(self, x, u):
        #Zero-order hold discretization. x_dot ≈ 1/dt(x_k+1 - x_k)
        #Ad = I + A*dt, Bd = B*dt
        B = np.array([
            [x[3] * cos(x[2]), 0],
            [x[3] * sin(x[2]), 0],
            [0, 1],
            [1, 0]
        ])

        Ad = np.eye(self.x.size) # A = 0
        Bd = B*self.T

        return Ad@self.x + Bd@u

    def F(self, x):
        return np.array([
            [1, 0, -x[3]*self.T*sin(x[2]), self.T*cos(x[2])],
            [0, x[3]*self.T*cos(x[2]), self.T*sin(x[2])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def full_observation(self, enable_noise=True):
        Cd = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        Dd = np.eye((2,2))
        G = Cd
        return Cd @ self.x + Dd @ self.u + enable_noise*np.random.normal(0, self.measurement_stds[:2]), G

    def rate_and_accel_observation(self, enable_noise=True):
        Cd = np.zeros((2,4))
        Dd = np.eye(2)
        G = Cd
        return Cd @ self.x + Dd @ self.u + enable_noise*np.random.normal(0, self.measurement_stds[2:]), G