from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class Vehicle:
    def __init__(self, x=np.array([0,0,0]).T, T=0.01):
        self.T = T
        
        self.x_data = []

        plt.ion()
        fig = plt.figure()
        fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])

        self.update_state()

    def update_state(self, x):
        self.x = x
        self.plot()


    def vehicle_dynamics(self, u):
        A = np.array([
            [0, 0, -sin(self.x[2])],
            [0, 0, cos(self.x[2])],
            [0, 0, 0]
        ])
        
        
        B = np.array([
            [cos(self.x[2]), 0],
            [sin(self.x[2]), 0],
            [0, 1]
        ])

        # Trick from https://en.wikipedia.org/wiki/Discretization
        AdBdI = expm(np.array([[A, B], [np.zeros((3,3)), np.zeros((3,2))]]))
        Ad = AdBdI[:3,:3]
        Bd = AdBdI[3:,:3]

        return Ad@self.x + Bd@u