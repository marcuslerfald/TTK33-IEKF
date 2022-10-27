from math import sin, cos
import numpy as np
    
class Vehicle:
    def __init__(self, x=np.array([0,0,0,0]), u=np.array([0,0]), T=0.01, measurement_stds=np.array([0.1,0.1,0.0001,0.0002])):
        # x: x, y, theta, v
        # u: a, omega
        self.T = T
        self.measurement_stds = measurement_stds

        self.x = x
        self.u = u
        self.x_data = np.empty((0,4), float)
        self.u_data = np.empty((0,2), float)

    def f(self, x, u, enable_noise=True):
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

        x_new = Ad @ x + Bd @ (u + enable_noise*np.random.normal(0, self.measurement_stds[-2:]))
        x_new[2] = x_new[2]%(2*np.pi) # Limit to 0~2pi
        return x_new

    def F(self, x):
        return np.array([
            [1, 0, -x[3]*self.T*sin(x[2]), self.T*cos(x[2])],
            [0, 1, x[3]*self.T*cos(x[2]), self.T*sin(x[2])],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def g(self, x, enable_noise=True):
        Cd = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return Cd @ x + enable_noise*np.random.normal(0, self.measurement_stds[:2]).T

    def G(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])