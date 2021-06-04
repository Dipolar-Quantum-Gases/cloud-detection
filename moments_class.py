# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:08:28 2021

@author: hofer
"""

import numpy as np

class moments():
    def coordinates(self, data, x_coords=None, y_coords=None, flatten=False):
        if (x_coords is None) and (y_coords is None):
            x_coords = np.arange(len(data[0]))
            y_coords = np.arange(len(data))
            x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        
            if flatten:
                x_coords = np.ndarray.flatten(x_coords)
                y_coords = np.ndarray.flatten(y_coords)
                data = np.ndarray.flatten(data)
                
        return x_coords, y_coords, data
    
    
    def normalization(self, data):
        return np.sum(data)
    
    
    def first_moments(self, x_coords, y_coords, data, total):
        x0 = np.sum(x_coords * data) / total
        y0 = np.sum(y_coords * data) / total
        return x0, y0
    

    def second_moments(self, x_coords, y_coords, data, total, x0, y0):
        x_var = np.sum((x_coords - x0)**2 * data) / total
        y_var = np.sum((y_coords - y0)**2 * data) / total
        xy_var = np.sum((x_coords - x0) * (y_coords - y0) * data) / total
        
        return x_var, y_var, xy_var
    
    
    def angle(self, x_var, y_var, xy_var):
        if x_var!=y_var:
            return .5*np.arctan((2 * xy_var) / (x_var - y_var))
        else:
            return (xy_var / abs(xy_var)) * (np.pi / 4)
        
        
    def distribution_variances(self, x_var, y_var, xy_var):
    
        p1 = (x_var + y_var)
        
        if x_var != y_var:
            sgn = (x_var - y_var) / abs(x_var - y_var)
            p2 = sgn * ((x_var - y_var)**2 + 4 * xy_var**2)**.5
        else:
            p2 =2 * abs(xy_var)
        
        dist_x_var = 0.5 * (p1 + p2)
        dist_y_var = 0.5 * (p1 - p2)
        return dist_x_var, dist_y_var
    

    def get_moments(self, data, x_coords=None, y_coords=None):
        x_coords, y_coords, data=self.coordinates(data, x_coords, y_coords, flatten=True)
        total = self.normalization(data)
        x0, y0 = self.first_moments(x_coords, y_coords, data, total)
        x_var, y_var, xy_var = self.second_moments(x_coords, y_coords, data, total, x0, y0)
        phi = self.angle(x_var, y_var, xy_var)
        dist_x_var, dist_y_var = self.distribution_variances(x_var, y_var, xy_var)
        return x0, y0, x_var, y_var, xy_var, phi, dist_x_var, dist_y_var, x_coords, y_coords, data