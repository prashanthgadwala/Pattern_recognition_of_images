import numpy as np
import matplotlib.pyplot as plt
from pip._internal import resolution


class Checker:
    def __init__(self, resolution, size):
        self.resolution = resolution
        self.size = size
        self.tile = np.zeros((self.size*2, self.size*2))
        self.output = np.zeros((self.resolution, self.resolution))
    def draw(self):
        self.tile[:self.size, :self.size] = 0
        self.tile[self.size:, :self.size] = 1
        self.tile[:self.size, self.size:] = 1
        self.tile[self.size:, self.size:] = 0
        self.output = np.tile(self.tile, (self.resolution // (self.size*2), self.resolution // (self.size*2)))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap = 'gray')
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((self.resolution, self.resolution), dtype='bool')


    def draw(self):
        X, Y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        distance = np.sqrt((X - self.position[0]) ** 2 + (Y - self.position[1]) ** 2)
        self.output = distance <= self.radius
        return self.output.copy()


    def show(self):
        plt.imshow(self.output, cmap = 'gray')
        plt.axis('off')
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution,3), dtype=np.float64)
    def draw(self):
        X, Y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        x_size = X / (self.resolution-1)
        y_size = Y / (self.resolution-1)

        self.output[:,:,0] = x_size
        self.output[:,:,1] = y_size
        self.output[:,:,2] = 1 - x_size
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap = 'gray')
        plt.axis('off')
        plt.show()
