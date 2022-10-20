from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

class Vehicle:
    def __init__(self, x=0, y=0, theta=0, T=0.01):
        self.T = T
        
        self.x_data = []
        self.y_data = []
        self.theta_data = []

        plt.ion()
        fig = plt.figure()
        fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

        self.update_state()

    def update_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.plot()


    def vehicle_dynamics(self, v, omega):
        A = np.array([
            [0, 0, -sin(self.theta)],
            [0, 0, cos(self.theta)],
            [0, 0, 0]
        ])
        
        
        B = np.array([
            [cos(self.theta), 0],
            [sin(self.theta), 0],
            [0, 1]
        ])

        AdBdI = linalg.expm(np.array([[A, B], [np.zeros((3,3)), np.zeros((3,2))]]))
        Ad = AdBdI[:3,:3]
        Bd = AdBdI[3:,:3]

        state = np.array([self.x, self.y, self.theta]).T
        gain = np.array([v, omega]).T

        return Ad@state + Bd@gain