# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:11:00 2021

@author: hofer
"""
import numpy as np
import moments_class as mom
import time

class mask_fit_class():   
    """Calculates the Gaussian parameters using image of mask

    Attributes:
        mom: class to do moment calculations
    """
    def __init__(self):
        self.mom = mom.moments()
    
    
    def get_coordinates(self, img):
        """gets coordinate maps corresponding to each pixel in the image
    
        Args:
            img: 2D numpy array
        Returns:
            X, Y: 2D numpy arrays of the x and y coordinates
        """  
        xlength = len(img[0])
        ylength = len(img)
        x = np.linspace(0, xlength - 1 , xlength)
        y = np.linspace(0, ylength - 1 , ylength)
        X, Y = np.meshgrid(x, y)
        return X, Y
     
    
    def calc_hist_offset(self, img, masks, threshold=.5, num_bins=1000):
        """gets coordinate maps corresponding to each pixel in the image
    
        Args:
            img: 2D numpy array
            mask: list of 2D numpy arrays which are mask for each cloud
            threshold: threshold value to make masks bool rather than float
        Returns:
            histoffset: offset for image
        """  
        bmasks = np.zeros(img.shape, dtype=bool) #empty 2D bool array 
        for k in range(0, len(masks)): #loop thru masks
            if str(masks[k].dtype) != 'bool': # if the masks are not bool already
                #get binary mask from float mask
                mask = self.get_binary_mask(masks, k, threshold=threshold) 
            else:
                mask = masks[k] #get the correct mask from list
            bmasks = np.logical_or(bmasks, mask)
        bmasks = np.logical_not(bmasks) #invert mask
        sdata = img[bmasks] #get image not inside 1/e^2 radius
        #create histogram bins which are bounded by min and max of data
        bins = np.linspace(np.amin(sdata), np.amax(sdata), num_bins) 
        hist, bins = np.histogram(sdata, bins = bins) #get histogram data
        histoffset = bins[np.argmax(hist)] #find bin with most counts
        return histoffset #this is the offset

    
    def calc_i0(self, p, wx, wy, offset): 
        """calculate 2D Gaussian amplitude using equation in paper
    
        Args:
            p: 2D numpy array
            wx: major radii
            wy: minor radii
            offset: intensity offset calculated from image
        Returns:
            amplitude: amplitude of 2D Gaussian
        """  
        offset = np.pi * wx * wy * offset #see equation in paper
        amplitude = (2 * (p - offset)) / (np.pi * wx * wy * (1 - np.exp(-2)))
        return amplitude
    
    
    def ocalc_i0(self, wx, wy, offset, img, mask): #calc amplitude
        """calculate 2D Gaussian amplitude
        First truncates at 1/e^2 radius then sums power inside to before using
        the equation in paper
    
        Args:
            wx: major radii
            wy: minor radii
            offset: intensity offset calculated from image
            img: 2D numpy array of experimental image
            mask: 2D numpy array of mask image
        Returns:
            amplitude: amplitude of 2D Gaussian
        """  
        masked_img = img * mask
        p = np.sum(masked_img)
        amplitude = self.calc_i0(p, wx, wy, offset)
        return amplitude


    def get_mask_fit(self, img, mask, X, Y, offset):
        """Extract Gaussian parameters for a cloud from mask and image
    
        Args:
            img: 2D numpy array of experimental image
            mask: 2D numpy array of mask image
            X: 2D numpy array of the x coordinates
            Y: 2D numpy arrays of the y coordinates
            offset: intensity offset calculated from image
        Returns:
            i0f, x0f, y0f, wxf, wyf, thetaf, offset: Gaussian parameters
        """  
        #1st and 2nd moment calcs on the mask
        x0f, y0f, x_var, y_var, xy_var, thetaf, dist_x_var, dist_y_var, x_coords, y_coords, datast = self.mom.get_moments(mask, X, Y)
        wxf = 2*dist_x_var**.5 #convert variances into wx and wy
        wyf = 2*dist_y_var**.5
        i0f = self.ocalc_i0(wxf, wyf, offset, img, mask) #calc the amplitude
        # thetaf += np.pi
        return [i0f, x0f, y0f, wxf, wyf, thetaf, offset]


    def get_mask(self, masks, i, threshold):
        """Extract Gaussian parameters for a cloud from mask and image
    
        Args:
            img: 2D numpy array of experimental image
            mask: 2D numpy array of mask image
            X: 2D numpy array of the x coordinates
            Y: 2D numpy arrays of the y coordinates
            offset: intensity offset calculated from image
        Returns:
            i0f, x0f, y0f, wxf, wyf, thetaf, offset: Gaussian parameters
        """  
        nmask = np.copy(masks[i])
        # nmask /= np.amax(nmask)
        nmask[nmask < threshold] = 0
        return nmask
    
    
    def get_binary_mask(self, masks, i, threshold):
        """Get binary mask based on threshold values for mask
    
        Args:
            masks: list with masks for each ROI
            i: which ROI to look at
            threshold: value between (0-1) at which mask is made binary
        Returns:
            nmask: boolean mask
        """  
        nmask = np.copy(masks[i][0]) #get correct mask
        # nmask /= np.amax(nmask)
        nmask[nmask < threshold] = 0 #set value below threshold to 0
        nmask[nmask >= threshold] = 1 #and values above to 1
        nmask = np.ndarray.astype(nmask, dtype=bool) #convert array to bool
        return nmask


    def filter_fits(self, i0, x0, y0, wx, wy, theta, offset):
      if wy > wx:
        wy_yemp = wy
        wy = wx
        wx = wy_yemp
        theta += np.pi / 2
      
      return i0, x0, y0, wx, wy, theta, offset
  
    
    def calc_mask_fits(self, img, masks, threshold=.5):
        """calculate the gaussian parameters from mask and img
    
        Args:
            img: 2D numpy array of experimental image
            mask: 2D numpy array of mask image
            threshold: value at which to threshold masks
        Returns:
            fit list: list of lists [[i01, i02], [x01, x02], [y01, y02]..] where 
                    the number denotes which region the parameter comes from
        """  
        X, Y = self.get_coordinates(img)
        offset = self.calc_hist_offset(img, masks) #calc offset for image
        num_objs = len(masks) #number of clouds in image
        # fit_list = [[] for i in range(0, 7)]
        mask_fits = []
        for i in range(0, num_objs): #loop through all the clouds in the image
            st = time.time()
            nmask = self.get_mask(masks, i, threshold=threshold) #get region mask
            mask_fit = self.get_mask_fit(img, nmask, X, Y, offset) #calc gauss params
            mask_fit = self.filter_fits(*mask_fit)
            mask_fits.append(mask_fit)
        return mask_fits