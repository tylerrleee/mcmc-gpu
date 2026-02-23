#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:57:24 2025

@author: niyashao
"""

###import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
import gstatsim as gs
import gstools as gstools
import skgstat as skg
from skgstat import models
from tqdm.auto import tqdm
from IPython import display
import math
import sys
import time

from . import Topography
from . import gstatsim_custom as gsim
from copy import deepcopy
import numbers

def move_cursor_to_line(line_number):
    """Move cursor to specific line for updating in place"""
    sys.stdout.write(f'\033[{line_number};0H')  # Move to line N, column 0
    sys.stdout.flush()

def clear_line():
    """Clear the current line"""
    sys.stdout.write('\033[2K')
    sys.stdout.flush()

# code adopted from gstatsim_custom by Michael
def _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.

    Returns:
        (out_grid, nst_trans, cond_msk, inds, vario, global_mean, stencil)
    """
    
    # get masks and gaussian transform data
    cond_msk = ~np.isnan(grid)
    #out_grid, nst_trans = gaussian_transformation(grid, cond_msk)
    out_grid = grid.copy()

    if sim_mask is None:
        sim_mask = np.full(xx.shape, True)

    # get index coordinates and filter with sim_mask
    # maybe this should be input rather than 
    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')
    inds = np.array([ii[sim_mask].flatten(), jj[sim_mask].flatten()]).T

    vario = deepcopy(variogram)

    # turn scalar variogram parameters into grid
    for key in vario:
        if isinstance(vario[key], numbers.Number):
            vario[key] = np.full(grid.shape, vario[key])

    # mean of conditioning data for simple kriging
    global_mean = np.mean(out_grid[cond_msk])

    # make stencil for faster nearest neighbor search
    if stencil is None:
        stencil, _, _ = gsim.neighbors.make_circle_stencil(xx[0,:], radius)


    return out_grid, cond_msk, inds, vario, global_mean, stencil

# code adopted from gstatsim_custom by Michael
def sgs(xx, yy, grid, variogram, radius=100e3, num_points=20, ktype='ok', sim_mask=None, quiet=False, stencil=None, rcond=None, seed=None):
    """
    Sequential Gaussian Simulation with ordinary or simple kriging using nearest neighbors found in an octant search.

    Args:
        xx (numpy.ndarray): 2D array of x-coordinates.
        yy (numpy.ndarray): 2D array of y-coordinates.
        grid (numpy.ndarray): 2D array of simulation grid. NaN everywhere except for conditioning data.
        variogram (dictionary): Variogram parameters. Must include, major_range, minor_range, sill, nugget, vtype.
        radius (int, float): Minimum search radius for nearest neighbors. Default is 100 km.
        num_points (int): Number of nearest neighbors to find. Default is 20.
        ktype (string): 'ok' for ordinary kriging or 'sk' for simple kriging. Default is 'ok'.
        sim_mask (numpy.ndarray or None): Mask True where to do simulation. Default None will do whole grid.
        quiet (book): Turn off progress bar when True. Default False.
        stencil (numpy.ndarray or None): Mask to use as 'cookie cutter' for nearest neighbor search.
            Default None a circular stencil will be used.
        seed (int, None, or numpy.random.Generator): If None, a fresh random number generator (RNG)
            will be created. If int, a RNG will be instantiated with that seed. If an instance of
            RNG, that will be used.

    Returns:
        (numpy.ndarray): 2D simulation
    """
    
    # check arguments
    gsim.interpolate._sanity_checks(xx, yy, grid, variogram, radius, num_points, ktype, sim_mask)

    # preprocess some grids and variogram parameters
    out_grid, cond_msk, inds, vario, global_mean, stencil = _preprocess(xx, yy, grid, variogram, sim_mask, radius, stencil)

    # make random number generator if not provided
    rng = gsim.utilities.get_random_generator(seed)

    # shuffle indices
    rng.shuffle(inds)

    ii, jj = np.meshgrid(np.arange(xx.shape[0]), np.arange(xx.shape[1]), indexing='ij')

    # iterate over indicies
    for k in range(inds.shape[0]):
        
        i, j = inds[k]

        nearest = np.array([])
        rad = radius
        stenc = stencil

        # check if grid cell needs to be simulated
        if cond_msk[i, j] == False:
            # make local variogram
            local_vario = {}
            for key in vario.keys():
                if key=='vtype':
                    local_vario[key] = vario[key]
                else:
                    local_vario[key] = vario[key][i,j]

            # find nearest neighbors, increasing search distance if none are found
            while nearest.shape[0] == 0:
                nearest = gsim.neighbors.neighbors(i, j, ii, jj, xx, yy, out_grid, cond_msk, rad, num_points, stencil=stenc)
                if nearest.shape[0] > 0:
                    break
                else:
                    rad += 100e3
                    stenc, _, _ = gsim.neighbors.make_circle_stencil(xx[0,:], rad)

            # solve kriging equations
            if ktype=='ok':
                est, var = gsim._krige.ok_solve((xx[i,j], yy[i,j]), nearest, local_vario, rcond)
            elif ktype=='sk':
                est, var = gsim._krige.sk_solve((xx[i,j], yy[i,j]), nearest, local_vario, global_mean, rcond)

            var = np.abs(var)

            # put value in grid
            # out_grid[i,j] = rng.normal(est, np.sqrt(var), 1)

            out_grid[i,j] = rng.normal(est, np.sqrt(var), 1)
            cond_msk[i,j] = True

    #sim_trans = nst_trans.inverse_transform(out_grid.reshape(-1,1)).squeeze().reshape(xx.shape)

    return out_grid


def spectral_synthesis_field(RF, shape, res=1.0):
    """
    Generate a 2D Gaussian random field using FFT-based spectral synthesis.
    This method uses a fast Fourier transform approach to produce spatially correlated
    random fields consistent with the variogram model stored in the RandField object.

    Args:
        RF (RandField): 
        shape (tuple[int,int]): 
        res (float, optional):

    Returns:
        np.ndarray:
            A 2D NumPy array of shape (ny, nx) representing a single random field 
            realization generated using the FFT-based spectral synthesis method.
            The field has zero mean and unit variance before scaling, then scaled 
            by the sampled vertical standard deviation (`scale`) and augmented 
            with Gaussian nugget noise. 
    """
    
    ny, nx = shape
    rng = RF.rng

    # Sample model parameters
    scale = rng.uniform(RF.scale_min, RF.scale_max) / 3.0
    nug = rng.uniform(0.0, RF.nugget_max)

    if not RF.isotropic:
        range_x = rng.uniform(RF.range_min_x, RF.range_max_x)
        range_y = rng.uniform(RF.range_min_y, RF.range_max_y)
    else:
        range_x = range_y = rng.uniform(RF.range_min_x, RF.range_max_x)

    model_name = RF.model_name
    if model_name == "Gaussian":
        len_x, len_y = range_x / np.sqrt(3), range_y / np.sqrt(3)
    elif model_name == "Exponential":
        len_x, len_y = range_x / 3.0, range_y / 3.0
    else:  # Matern
        len_x, len_y = range_x / 2.0, range_y / 2.0
        #len_x, len_y = range_x, range_y

    #print(len_x, len_y, '3')

    # Frequency grids
    kx = np.fft.fftfreq(nx, d=res) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=res) * 2 * np.pi
    kyv, kxv = np.meshgrid(ky, kx, indexing="ij")
    k = np.sqrt(kxv**2 + kyv**2) + 1e-10

    # Spectral power density
    if model_name == "Gaussian":
        a = np.sqrt((len_x * len_y))
        S = np.exp(-0.5 * (a * k) ** 2)
    elif model_name == "Exponential":
        a = np.sqrt((len_x * len_y))
        S = 1.0 / (1.0 + (a * k) ** 2) ** 1.5
    else:  # Matern (approximate)
        nu = RF.smoothness or 1.0
        a = np.sqrt((len_x * len_y))
        #S = 1.0 / (1.0 + (a * k) ** 2) ** (nu + 1)
        constant = (4 * np.pi * math.gamma(nu + 1) * (2 * nu)**nu) / (math.gamma(nu) * a**(2*nu))
        keppa = 2*nu/(a**2)
        S = constant * ((keppa + 4 * np.pi * k**2) ** (-nu - 1))

    # Complex white noise
    noise = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
    #noise = np.random.normal(size=(ny, nx)) + 1j * np.random.normal(size=(ny, nx))
    freq_field = noise * np.sqrt(S)

    # Inverse FFT
    field = np.fft.ifft2(freq_field).real
    field = (field - np.mean(field)) / (np.std(field) + 1e-12)

    # Apply scaling and nugget noise
    field = field * scale + rng.normal(0, np.sqrt(nug), size=(ny, nx))
    #field = field * scale + np.random.normal(0, np.sqrt(nug), size=(ny, nx))

    return field


def fit_variogram(data, coords, roughness_region_mask, maxlag, n_lags=50, samples=0.6, subsample=100000, data_for_trans = []):
    """
    This function computes an experimental variogram from the input data and fits
    several theoretical models (Gaussian, Exponential, Spherical, Matern) to it.
    By default, the nugget is fixed at 0.

    Args:
        data (np.ndarray): A column vector of shape (N, 1) containing the input
            data for variogram computation. Use `data.reshape((-1, 1))` if the
            input is a 1D array.
        coords (np.ndarray): An array of shape (N, 2) with spatial coordinates
            corresponding to the data. Each row represents a location, with the
            first column as the x-coordinate and the second as the y-coordinate.
        roughness_region_mask (np.ndarray): A binary mask of shape (M, N)
            identifying the region for roughness evaluation (1 for included,
            0 for excluded).
        maxlag (float): The maximum lag distance to consider for the variogram.
        n_lags (int, optional): The number of lag bins for the experimental
            variogram. Defaults to 50.
        samples (float, optional): The proportion of data pairs to use when
            computing the experimental variogram. Defaults to 0.6.
        subsample (int, optional): The number of data points to use for the
            quantile transformation. Defaults to 100,000.
        data_for_trans (np.ndarray, optional): A column vector of data used to
            compute the quantile transformation, if different from `data`.

    Returns:
        tuple: A tuple containing the following:
            nst_trans (sklearn.preprocessing.QuantileTransformer): The fitted
                quantile transformer used for data normalization.
            transformed_data (np.ndarray): The quantile-transformed data used in
                the variogram calculation.
            params (list[dict]): A list of dictionaries, one for each fitted
                model (Gaussian, Exponential, Spherical, Matern). Each dict
                contains the 'range', 'sill', and 'nugget' parameters.
            fig (matplotlib.figure.Figure): A figure object that plots the
                experimental variogram against the fitted theoretical models.
    """
    if len(data_for_trans)==0:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=152,subsample=subsample).fit(data)
    else:
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal",random_state=152,subsample=subsample).fit(data_for_trans)
        
    transformed_data = nst_trans.transform(data)
    
    coords = coords[roughness_region_mask==1]
    values = transformed_data[roughness_region_mask==1].flatten()
    
    test1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                    maxlag=maxlag, normalize=False, model='gaussian',samples=samples)
    test2 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='exponential',samples=samples)
    test3 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='spherical',samples=samples)
    test4 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, 
                       maxlag=maxlag, normalize=False, model='matern',samples=samples)

    tp1 = test1.parameters
    tp2 = test2.parameters
    tp3 = test3.parameters
    tp4 = test4.parameters

    print('range, sill, and nugget for gaussian variogram is ', tp1)
    print('for exponential variogram is ', tp2)
    print('for spherical variogram is ', tp3)
    print('range, sill, smoothness, and nugget for for matern variogram is ', tp4)
    
    # extract experimental variogram values
    xdata1 = test1.bins
    ydata1 = test1.experimental
    xdata2 = test2.bins
    ydata2 = test2.experimental
    xdata3 = test3.bins
    ydata3 = test3.experimental
    xdata4 = test4.bins
    ydata4 = test4.experimental

    # evaluate models
    xi = np.linspace(0, xdata2[-1], n_lags) 
    y_gauss = [models.gaussian(h, tp1[0], tp1[1], tp1[2]) for h in xi]
    y_exp = [models.exponential(h, tp2[0], tp2[1], tp2[2]) for h in xi]
    y_sph = [models.spherical(h, tp3[0], tp3[1], tp3[2]) for h in xi]
    y_mtn = [models.matern(h, tp4[0], tp4[1], tp4[2], tp4[3]) for h in xi] # variogram parameter is [range, sill, shape, nugget] for matern model.

    # plot variogram model
    fig = plt.figure(figsize=(6,4))
    plt.plot(xi, y_gauss,'b--', label='Gaussian variogram model')
    plt.plot(xi, y_exp,'b-', label='Exponential variogram model')
    plt.plot(xi, y_sph,'b*-', label='Spherical variogram model')
    plt.plot(xi, y_mtn,'b-.', label='Matern variogram model')
    plt.plot(xdata1, ydata1,'o', markersize=4, color='green', label='Experimental variogram gaussian',alpha=0.4)
    plt.plot(xdata2, ydata2,'o', markersize=4, color='orange', label='Experimental variogram exponential',alpha = 0.4)
    plt.plot(xdata3, ydata3,'o', markersize=4, color='pink', label='Experimental variogram spherical', alpha = 0.4)
    plt.plot(xdata4, ydata4,'o', markersize=4, color='burlywood', label='Experimental variogram matern', alpha = 0.4)
    plt.title('Variogram for synthetic data')
    plt.xlabel('Lag [m]'); plt.ylabel('Semivariance')  
    plt.legend(loc='lower right', fontsize=8)
    
    return nst_trans, transformed_data, [tp1, tp2, tp3, tp4], fig

# helper function used to initiate a new large scale chain 
# from parameter dictionary of an existing large scale chain
def init_lsc_chain_by_instance(param_dict):
    p = param_dict
    chain = chain_crf(param_dict['xx'],param_dict['yy'],param_dict['initial_bed'],param_dict['surf'],param_dict['velx'],param_dict['vely'],param_dict['dhdt'],
                      param_dict['smb'],param_dict['cond_bed'],param_dict['data_mask'],param_dict['grounded_ice_mask'],param_dict['resolution'])
    chain.update_in_region = p['update_in_region']
    chain.region_mask = p['region_mask']
    chain.sigma_mc = p['sigma_mc']
    #chain.sigma_data = p['sigma_data']
    #chain.map_func = p['map_func']
    #chain.diff_func = None
    chain.block_type = p['block_type']
    chain.crf_data_weight = p['crf_data_weight']
    chain.rng = np.random.default_rng(seed=p['rng_seed'])
    chain.rng_seed = p['rng_seed']
    chain.mc_region_mask = p['mc_region_mask']
    #chain.data_region_mask = p['data_region_mask']
    chain.sample_loc = deepcopy(p['sample_loc'])
        
    return chain

# helper function used to initiate a new RandField instance 
# from parameter dictionary of an existing RandField instance
def initiate_RF_by_instance(param_dict):
    p = param_dict
    rf = RandField(p['range_min_x'],p['range_max_x'],p['range_min_y'],p['range_max_y'],
                   p['scale_min'],p['scale_max'],p['nugget_max'],p['model_name'],
                   p['isotropic'],smoothness=p['smoothness'])
    rf.spectral = p['spectral']
    rf.min_block_x = p['min_block_x']
    rf.min_block_y = p['min_block_y']
    rf.max_block_x = p['max_block_x']
    rf.max_block_y = p['max_block_y']
    rf.steps = p['steps']
    rf.pairs = p['pairs']
    rf.logistic_param = p['logistic_param']
    rf.max_dist = p['max_dist']
    rf.resolution = p['resolution']
    rf.edge_masks = p['edge_masks']
    rf.rng = np.random.default_rng(seed=p['rng_seed'])
    return rf

# helper function used to initiate a new small scale chain 
# from parameter dictionary of an existing small scale chain
def init_msc_chain_by_instance(param_dict):
       
    p = param_dict
    chain = chain_sgs(param_dict['xx'],param_dict['yy'],param_dict['initial_bed'],param_dict['surf'],param_dict['velx'],param_dict['vely'],param_dict['dhdt'],
                      param_dict['smb'],param_dict['cond_bed'],param_dict['data_mask'],param_dict['grounded_ice_mask'],param_dict['resolution'])
    chain.update_in_region = p['update_in_region']
    chain.region_mask = p['region_mask']
    chain.sigma_mc = p['sigma_mc']
    chain.mc_region_mask = p['mc_region_mask']
    #chain.sigma_data = p['sigma_data']
    #chain.map_func = p['map_func']
    #chain.diff_func = None
    chain.block_min_x = p['block_min_x']
    chain.block_min_y = p['block_min_y']
    chain.block_max_x = p['block_max_x']
    chain.block_max_y = p['block_max_y']
    chain.do_transform = p['do_transform']
    chain.nst_trans = deepcopy(p['nst_trans'])
    chain.trend = p['trend']
    chain.detrend_map = p['detrend_map']
    chain.vario_type = p['vario_type']
    chain.vario_param = deepcopy(p['vario_param'])
    chain.sgs_param = deepcopy(p['sgs_param'])
    chain.rng = np.random.default_rng(seed=p['rng_seed'])
    chain.rng_seed = p['rng_seed']
    chain.sample_loc = deepcopy(p['sample_loc'])
    #chain.data_region_mask = p['data_region_mask']
        
    return chain
        

class RandField:
    """Generates 2D random fields based on specified variogram models.

    This class creates random fields with defined spatial statistics. It can produce both unconditional fields and conditional fields, where conditioning is achieved using a weighting scheme based on the distance to known data points.

    Before generating fields, the `set_block_sizes` and `set_weight_param` methods must be called to initialize the simulation and conditioning parameters.

    Attributes:
        range_min_x (float): Minimum spatial correlation range in x-direction. The range in x-direction for each realization is randomly sampled between `range_min_x` and `range_max_x`. Similarly, the range in y-direction is randomly sampled between `range_min_y` and `range_max_y`.
        range_max_x (float): Maximum spatial correlation range in x-direction.
        range_min_y (float): Minimum spatial correlation range in y-direction.
        range_max_y (float): Maximum spatial correlation range in y-direction.
        scale_min (float): Minimum vertical scaling (standard deviation).
        scale_max (float): Maximum vertical scaling (standard deviation).
        nugget_max (float): Maximum nugget effect.
        model_name (str): The variogram model type ('Gaussian', 'Exponential', or 'Matern').
        isotropic (bool): Flag for isotropic (True) or anisotropic (False) fields.
        smoothness (float): Smoothness parameter for the Matern model.
        rng (np.random.Generator): The random number generator instance.
        pairs (list[tuple]): List of (block_width, block_height) tuples generated by `set_block_sizes`.
        logistic_param (list[float]): Parameters [L, x0, k, offset] for the logistic weighting function.
        max_dist (float): Maximum distance used for scaling the logistic weighting mask.
        resolution (float): Spatial resolution of the grid for conditioning.
        edge_masks (np.ndarray): Precomputed masks for applying conditioning weights at edges.
    """
    
    def __init_func(self):
        print("Before using the `RandField` object in an MCMC chain or for field generation, \n call function `set_block_sizes` to initialize block size ranges; \n call function `set_weight_param` to set up conditional weighting parameters; \n call function 'set_generation_method' to set up method used to generate random fields.")

    def __init__(self,range_min_x,range_max_x,range_min_y,range_max_y,scale_min,scale_max,nugget_max,model_name,isotropic,smoothness = None, rng_seed = None):
        """Initializes the RandField object.

        This method sets up the parameters for generating random fields based on specified spatial correlation ranges, variogram model type, and scaling. Anisotropy and nugget effects can be included. 
        A NumPy random number generator is initialized using `np.random.default_rng()`, and seeds to this generator is optional. 
        The maximum range values must be greater than or equal to the corresponding minimums, and a `smoothness` value is required if `model_name` is 'Matern'.

        Args:
            range_min_x (float): Minimum spatial correlation range in the x-direction. The actual range is randomly sampled between `range_min_x` and `range_max_x`.
            range_max_x (float): Maximum spatial correlation range in the x-direction.
            range_min_y (float): Minimum spatial correlation range in the y-direction. The actual range is randomly sampled between `range_min_y` and `range_max_y`.
            range_max_y (float): Maximum spatial correlation range in the y-direction.
            scale_min (float): Minimum vertical scaling (standard deviation) of the random field.
            scale_max (float): Maximum vertical scaling (standard deviation) of the random field.
            nugget_max (float): Maximum nugget effect in the variogram.
            model_name (str): The variogram model type used for defining spatial covariance ('Gaussian', 'Exponential', or 'Matern').
            isotropic (bool): If True, enforces isotropic spatial correlation. If False, generates fields with random anisotropy, where the strength depends on the ratio between x- and y-direction ranges.
            smoothness (float, optional): The smoothness parameter for the Matern variogram model. Required if `model_name='Matern'`.
            rng_seed (int, optional): The seed for the NumPy random number generator. Notice that rng_seed will automatically be overriden by rng in the MCMC chain in the run() function. If None, a random seed is used. If a seed is used, the RandField object will produce a fixed sequence of conditional / unconditional random fields. 
        """
            
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
 
        self.rng = rng
            
        if (range_max_x < range_min_x) or (range_max_y < range_min_y):
            print('the maximum range must be greater to equal to the minimum range')
        
        self.range_max_x = range_max_x
        self.range_max_y = range_max_y
        self.range_min_x = range_min_x
        self.range_min_y = range_min_y
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.nugget_max = nugget_max
        if (model_name != 'Gaussian') and (model_name != 'Exponential') and (model_name != 'Matern'):
            raise Exception('please put in a valid model_name, including Gaussian, Exponential, and Matern')
        if (model_name == 'Matern') and (smoothness == None):
            raise Exception('a smoothness value must be defined if model name is Matern')
        self.smoothness = smoothness
        self.model_name = model_name
        self.isotropic = isotropic
        
        self.__init_func()
        
    def set_generation_method(self,spectral):
        """
        Define the generation method. If spectral is False, then use gstools RandMeth generator. Otherwise, use spectral systhesis to generate random field
        
        Args:
            spectral (bool): Whether use spectral synthesis to generate random field
        """
        
        self.spectral = spectral
           
    def set_block_sizes(self,min_block_x,max_block_x,min_block_y,max_block_y,steps=5):
        """
        Defines the minimum and maximum block dimensions in x and y directions and determines the number of discrete block sizes between them. This method does not return a value; the resulting block size pairs are stored in the ``pairs`` attribute.
        
        Args:
            min_block_x (int): The minimum block width.
            max_block_x (int): The maximum block width.
            min_block_y (int): The minimum block height.
            max_block_y (int): The maximum block height.
            steps (int, optional): The number of size intervals between the minimum and maximum values. Defaults to 5.
        """
        
        self.min_block_x = min_block_x
        self.min_block_y = min_block_y
        self.max_block_x = max_block_x
        self.max_block_y = max_block_y
        self.steps = steps

        self.pairs = self.get_block_sizes()
    
    def set_weight_param(self, logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution):
        """
        Set logistic function parameters for computing conditioning weights.
    
        The logistic function is defined as ``f(x) = (L / (1 + exp(-k(x - x0)))) - offset``, where ``x`` is the distance to the nearest conditioning data point. The resulting weight equals 1 at conditioning data locations and decays logistically with distance 'x'.
    
        Args:
            logis_func_L (float): The upper limit (L) of the logistic function.
            logis_func_x0 (float): The distance value (x0) where the function reaches its midpoint.
            logis_func_k (float): The growth rate (k) of the logistic curve.
            logis_func_offset (float): A constant offset subtracted from the logistic function's output.
            max_dist (float): The maximum distance used for scaling the weighting mask.
            resolution (float): The spatial resolution of the grid.
        """
        
        if not hasattr(self, 'pairs'):
            raise Exception('It seems like the set_block_sizes has not been called yet before calling set_weight_param')
        
        self.logistic_param = [logis_func_L, logis_func_x0, logis_func_k, logis_func_offset]
        self.max_dist = max_dist
        self.resolution = resolution
        self.edge_masks = self.get_edge_masks()
        
    
    def get_block_sizes(self):
        """
        Internal function. Calculate the discrete block sizes from set_block_size()

        Returns:
            pairs (np.ndarray) : An array of shape (2, N) containing the discrete block sizes. The first row holds the widths and the second row holds the heights.
        """
        
        width = np.linspace(self.min_block_x,self.max_block_x,self.steps,dtype=int)
        height = np.linspace(self.min_block_y,self.max_block_y,self.steps,dtype=int)
        w,h = np.meshgrid(width,height)
        pairs = np.array([(w//2*2).flatten(),(h//2*2).flatten()])
        
        return pairs
    
    def get_edge_masks(self):
        """
        Generate block-edge conditioning masks for each block size.
        For each block defined in ``pairs``, this method computes a distance-based logistic weighting mask along the block edges. The resulting masks are used for conditioning random fields near block boundaries.
        
        Returns:
            edge_masks (list : List of 2D arrays representing edge conditioning masks for each block size.
        """
        
        if not hasattr(self, 'pairs'):
            raise Exception('It seems like the set_block_sizes has not been called yet before calling get_edge_mask')
        
        edge_masks = []
        pairs = self.pairs
        res = self.resolution

        for i in range(pairs.shape[1]):
            bwidth = pairs[:,i][0]
            bheight = pairs[:,i][1]
            
            xx,yy=np.meshgrid(range(bwidth),range(bheight))
            xx = xx*res 
            yy = yy*res
            
            # make a mask of the block boundaries
            cond_msk_edge = np.zeros((bheight,bwidth))
            cond_msk_edge[0,:]=1
            cond_msk_edge[bheight-1,:]=1
            cond_msk_edge[:,0]=1
            cond_msk_edge[:,bwidth-1]=1
            
            # calculate the distance to block boundaries
            dist_edge = RandField.min_dist(np.where(cond_msk_edge==0, np.nan, 1), xx, yy)
            # re-scale the distance by the maximum correlation distance
            dist_rescale_edge = RandField.rescale(dist_edge, self.max_dist)
            # calculate the logistic function
            dist_logi_edge = RandField.logistic(dist_rescale_edge, self.logistic_param[0], self.logistic_param[1], self.logistic_param[2]) - self.logistic_param[3]
            edge_masks.append(dist_logi_edge)

        return edge_masks
    
    def get_random_field(self,X,Y,n=1):
        """Generates random field realizations.

        This method samples variogram parameters (e.g., range, sill, nugget) from the ranges specified in the object's attributes.
        It then creates one or more spatial random fields on the provided grid using the selected covariance model.

        Args:
            X (np.ndarray): A 1D array of the grid's x-coordinates.
            Y (np.ndarray): A 1D array of the grid's y-coordinates.
            n (int, optional): The number of random field realizations to generate. Defaults to 1.

        Returns:
            np.ndarray: A 3D array of shape (n, len(Y), len(X)) containing the generated random fields.
        """
        
        rng = self.rng
        
        _mean=0
        _var=1
        
        scale  = rng.uniform(low=self.scale_min, high=self.scale_max, size=1)[0]/3
        nug = rng.uniform(low=0.0, high=self.nugget_max, size=1)[0]
        
        if not self.isotropic:
            range1 = rng.uniform(low=self.range_min_x, high=self.range_max_x, size=1)[0]
            range2 = rng.uniform(low=self.range_min_y, high=self.range_max_y, size=1)[0]
            angle = rng.uniform(low=0, high=180, size=1)[0]
        else:
            range1 = rng.uniform(low=self.range_min_x, high=self.range_max_x, size=1)[0]
            range2 = range1
            angle = 0.0
            
        if self.model_name == 'Gaussian':
            model = gstools.Gaussian(dim=2, var = _var,
                                len_scale = [range1/np.sqrt(3),range2/np.sqrt(3)],
                                angles = angle*np.pi/180,
                                nugget = nug)
        elif self.model_name == 'Exponential':
            model = gstools.Exponential(dim=2, var = _var,
                        len_scale = [range1/3,range2/3],
                        angles = angle*np.pi/180,
                        nugget = nug)
        elif self.model_name == 'Matern':
            smoothness = self.smoothness
            model = gstools.Matern(dim=2, var=_var,
                        len_scale = [range1/2,range2/2],
                        angles = angle*np.pi/180,
                        nugget = nug,
                        nu = smoothness)
        else:
            print('error model name')
            return

        fields = np.zeros((n,len(Y),len(X)))
        for i in range(n):
            # Covariance model field generation
            srf = gstools.SRF(model)
            fields[i,:,:] = srf.structured([X, Y]).T*scale + _mean
            
            # Spectral synthesis field generation
            #fields[i,:,:] = spectral_synthesis_field(self, (len(Y), len(X)), res=self.resolution)

        return fields[0,:,:]
    
    def min_dist(hard_mat, xx, yy):
        """
        Compute the minimum Euclidean distance to non-NaN points.
        
        Notes: This is an internal helper function and not intended for direct use.
        """
        dist = np.zeros(xx.shape)
        xx_hard = np.where(np.isnan(hard_mat), np.nan, xx)
        yy_hard = np.where(np.isnan(hard_mat), np.nan, yy)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                dist[i,j] = np.nanmin(np.sqrt(np.square(yy[i,j]-yy_hard)+np.square(xx[i,j]-xx_hard)))
        return dist

    def rescale(x, maxdist):
        """
        Rescale distance values by the specified maximum distance.
        
        Notes: This is an internal helper function and not intended for direct use.
        """
        return np.where(x>maxdist,1,(x/maxdist))

    def logistic(x, L, x0, k):
        """
        Evaluate the logistic function with given parameters.
        
        Notes: is an internal helper function and not intended for direct use.
        """
        return L/(1+np.exp(-k*(x-x0)))
    
    def get_crf_weight(self,xx,yy,cond_data_mask):
        """
        This method generates weights for a conditional random field using a mask showing locations of the conditioning data. 

        Args:
            xx (np.ndarray): A 2D array of the grid's x-coordinates.
            yy (np.ndarray): A 2D array of the grid's y-coordinates.
            cond_data_mask (np.ndarray): A 2D array showing locations of conditioning data. (1 = have data, 0 = do not have data)

        Returns:
            weight (np.ndarray): 2D array. The final conditioning weights.
            dist (np.ndarray): 2D array. The minimal distance to the closest conditioning data.
            dist_rescale (np.ndarray): 2D array. The distance array scaled such that `self.max_dist` maps to 1.
            dist_logi (np.ndarray): 2D array. The raw output of the logistic function applied to the rescaled distances.
        """
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        # calculate the distance to block boundaries
        dist = RandField.min_dist(np.where(cond_data_mask==0, np.nan, 1), xx, yy)
        # re-scale the distance by the maximum correlation distance
        dist_rescale = RandField.rescale(dist, max_dist)
        # calculate the logistic function
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi

    def get_crf_weight_from_dist(self,xx,yy,dist):
        """
        This method generates weights for a conditional random field using a distance array that has already been computed. 
        It is useful for large domains where calculating distances on-the-fly is computationally expensive, allowing the distance array to be loaded from a file instead.

        Args:
            xx (np.ndarray): A 2D array of the grid's x-coordinates.
            yy (np.ndarray): A 2D array of the grid's y-coordinates.
            dist (np.ndarray): A 2D array containing the pre-calculated distance to the nearest conditioning point for each grid cell.

        Returns:
            weight (np.ndarray): 2D array. The final conditioning weights.
            dist (np.ndarray): 2D array. The original, unmodified distance array passed to the function.
            dist_rescale (np.ndarray): 2D array. The distance array scaled such that `self.max_dist` maps to 1.
            dist_logi (np.ndarray): 2D array. The raw output of the logistic function applied to the rescaled distances.
        """
        logistic_param = self.logistic_param
        max_dist = self.max_dist
        dist_rescale = RandField.rescale(dist, max_dist)
        dist_logi = RandField.logistic(dist_rescale, logistic_param[0], logistic_param[1], logistic_param[2]) - logistic_param[3]

        weight = dist_logi - np.min(dist_logi)
        return weight, dist, dist_rescale, dist_logi
    
    def get_rfblock(self):
        """
        Generate a random field block based on the information stored in the RandField object.
        The block size is randomly selected from values defined in set_block_sizes() function. 
        The random field has a logistic decay to the block's edges
        
        Returns:
            f (2D numpy array): a random field sample shaped by logistic mask.
        """
        
        res = self.resolution
        
        # randomly choose a size from the list
        block_size_i = self.rng.integers(low=0, high=self.pairs.shape[1], size=1)[0]
        block_size = self.pairs[:,block_size_i]
        
        #generate field
        x_uniq = np.arange(0,block_size[0]*res,res)
        y_uniq = np.arange(0,block_size[1]*res,res)

        #in-case of a weird bug
        while True:
            ## TODO: have to modify this for n>1
            #f = self.get_random_field(x_uniq, y_uniq)
            if self.spectral == True:
                f = spectral_synthesis_field(self, (len(y_uniq), len(x_uniq)), res=self.resolution)
            else:
                f = self.get_random_field(x_uniq, y_uniq)
                
            #f = f[0,:,:]
            if (np.sum(np.isnan(f))) != 0:
                print('f have nan')
                continue
            else:
                break
            
        return f*self.edge_masks[block_size_i]
    
class chain:
    """
    A base class for setting up a Markov chain for MCMC topography sampling.

    This class handles the initialization of data and the configuration of the loss function for its child classes (e.g., crf_chain, sgs_chain).
    
    Parameters:
        xx, yy, initial_bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution: As defined in the initialization function
        update_in_region (bool): If 'True', the block update will only happen within the region. If 'False', the block update will happen for the entire map
        region_mask (numpy.ndarray): The 2D mask defining the region where block update occur (1 = inside the region, 0 = outside the region).
        mc_region_mask (numpy.ndarray): The 2D mask defining the region where the mass conservation loss is calculated (1 = inside the region, 0 = outside the region).
        data_region_mask (numpy.ndarray): The 2D mask defining the region where the data misfit loss is calculated (1 = inside the region, 0 or np.nan = outside the region).
        sigma_mc (float): standard deviation for the mass conservation loss, when the 'map_func' is 'sumsquare' (a.k.a. the mass flux residuals have gaussian distributions)
        sigma_mc (float): standard deviation for the data misfit loss, when the 'diff_func' is 'sumsquare' (a.k.a. the data misfits have gaussian distributions)
        map_func (string): A string for choosing the function to calculate for mass conservation loss.
        diff_func (string): A string for choosing the function to calculate for data misfit loss.
        loss_function_list (list): A list of two functions used for mass conservation loss and data misfit loss respective.
    """

    def __init_func__(self):
        """
        A placeholder function for printing out notes and instruction on setting-up the chain

        No returns.
        """
        
        return
    
    def __init__(self, xx, yy, initial_bed, surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution):
        """Initializes the MCMC chain with input data and model geometry.

        Args:
            xx (np.ndarray): A 2D array of grid x-coordinates.
            yy (np.ndarray): A 2D array of grid y-coordinates.
            initial_bed (np.ndarray): The initial bed topography guess.
            surf (np.ndarray): The ice surface elevation.
            velx (np.ndarray): The ice surface velocity in the x-direction.
            vely (np.ndarray): The ice surface velocity in the y-direction.
            dhdt (np.ndarray): The rate of surface height change.
            smb (np.ndarray): The surface mass balance.
            cond_bed (np.ndarray): Conditioning radar bed measurements. Grid cells without data should be marked with NaN.
            data_mask (np.ndarray): A binary mask where 1 indicates the presence of conditioning data.
            grounded_ice_mask (np.ndarray): A binary mask where 1 indicates grounded ice.
            resolution (float): The spatial resolution of the grid in meters.

        Raises:
            Exception: If input arrays do not have matching shapes.
        """
        
        self.xx = xx
        self.yy = yy
        self.initial_bed = initial_bed
        self.surf = surf
        self.velx = velx
        self.vely = vely
        self.dhdt = dhdt
        self.smb = smb
        self.cond_bed = cond_bed
        self.data_mask = data_mask
        self.grounded_ice_mask = grounded_ice_mask
        self.resolution = resolution
        self.loss_function_list = []
        self.sample_loc = None
        
        if (initial_bed.shape!=surf.shape) or (initial_bed.shape!=velx.shape) or (initial_bed.shape!=vely.shape) or (initial_bed.shape!=dhdt.shape) or (initial_bed.shape!=smb.shape) or (initial_bed.shape!=cond_bed.shape) or (initial_bed.shape!=data_mask.shape):
            raise Exception('the shape of bed, surf, velx, vely, dhdt, smb, radar_bed, data_mask need to be same')
        
        self.__init_func__()
        
    def set_update_region(self, update_in_region, region_mask = []):
        """
        Defines a spatial mask to constrain where topography updates can occur.

        Args:
            update_in_region (bool): If True, updates are restricted to the area defined by `region_mask`.
            region_mask (np.ndarray): A binary mask specifying the update region.

        Raises:
            ValueError: If `region_mask` does not match the shape of the grid in the initialization.
        """
        
        self.update_in_region = update_in_region
        
        if update_in_region == False:
            print('the update blocks is set to be randomly generated for any locations inside the entire map')
            self.region_mask = np.full(self.xx.shape, 1)
        else:
            if region_mask.shape != self.xx.shape:
                raise ValueError('the region_mask input is invalid. It has to be a 2D numpy array with the shape of the map')
                self.region_mask = None
            else:
                print('the update blocks is set to be randomly generated for any locations inside the given region')
                self.region_mask = region_mask
    
    def __meanabs(data,mask):
        """
        A loss function, return the mean of absolute values of data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the mean of absolute data inside the mask
        """
        return np.nanmean(np.abs(data[mask==1]))
    
    def __meansq(data,mask):
        """
        A loss function, return the mean of squared data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the mean of squared data inside the mask
        """
        return np.nanmean(np.square(data[mask==1]))
    
    def __sumabs(data,mask):
        """
        A loss function, return the sum of absolute values of data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the sum of the absolute data value inside the mask
        """
        return np.nansum(np.abs(data[mask==1]))
    
    def __sumsq(data,mask):
        """
        A loss function, return the sum of squared data inside the mask
        
        Args:
            data (np.ndarray): a 2D numpy array for calculating the loss
            mask (np.ndarray): a 2D numpy array of booleans specifying locations to calculate the loss. (1 = calculate loss, 0 = do not calculate the loss)
            
        Returns:
            the sum of the squared data inside the mask
        """
        return np.nansum(np.square(data[mask==1]))
    
    def __return0(data,mask):
        return 0

# =============================================================================
#     # currently unusable for parallelization
#     def set_loss_type(self, map_func = None, diff_func = None, sigma_mc = -1, sigma_data = -1, massConvInRegion = True, dataDiffInRegion = False):
#         """
#         Configure loss function used in MCMC chain
#         Currently enable either of losses (or both) to be used: mass flux residuals and misfit to radar measurements.
#         The function has no return. Its effect can be checked in chain object's 'loss_function_list' attribute
#         
#         Args:
#             map_func (str, optional): The aggregation function for the mass flux residuals ('meanabs', 'meansquare', 'sumabs', 'sumsquare'). 'sumsquare' corresponds to a Gaussian likelihood. If None, this loss component is ignored.
#             diff_func (str, optional): The aggregation function for the radar data misfit ('meanabs', 'meansquare', 'sumabs', 'sumsquare'). 'sumsquare' corresponds to a Gaussian likelihood. If None, this loss component is ignored.
#             sigma_mc (float): The standard deviation for the mass flux residuals likelihood. This is required if `map_func` is not None.
#             sigma_data (float): The standard deviation for the data misfit likelihood. This is required if `diff_func` is not None.
#             massConvInRegion (bool): If True, calculates the mass conservation loss only within the specified `region_mask`.
#             dataDiffInRegion (bool): If True, calculates the data misfit loss only within the specified `region_mask`.
# 
#         Raises:
#             ValueError: If function names are invalid or required `sigma` values are not provided.
#         """
# =============================================================================

    def set_loss_type(self, sigma_mc = -1, massConvInRegion = True):
        """
        Configure loss function used in MCMC chain
        For multiprocessing, the loss function is fixed by using sumed square loss (representing Gaussian distribution) for mass conservation residual and using no data loss
       
        Args:
            sigma_mc (float): The standard deviation for the mass flux residuals likelihood. This is required if `map_func` is not None.
            massConvInRegion (bool): If True, calculates the mass conservation loss only within the specified `region_mask`.

        Raises:
            ValueError: If function names are invalid or required `sigma` values are not provided.
        """
    
# =============================================================================
#         function_list = []
#         
#         if (map_func == None) and (diff_func == None):
#             raise ValueError('please specify either one of or both of map_func and diff_func. The chain need at least one loss function. If plan to set up custom function later, please also put in valid values in map_func and/or data_func as a filler')
#         if ((map_func != None) and (sigma_mc <= 0)) or ((diff_func != None) and (sigma_data <= 0)):
#             raise ValueError('please make sure sigma is correctly set for either sigma_mc and/or sigma_data (sigma >= 0)')
#     
# =============================================================================
        if massConvInRegion:
            self.mc_region_mask = self.region_mask
        else:
            self.mc_region_mask = np.full(self.xx.shape,1)
        
# =============================================================================
#         if dataDiffInRegion:
#             self.data_region_mask = self.region_mask
#         else:
#             self.data_region_mask = np.full(self.xx.shape,1)
# =============================================================================
            
# =============================================================================
#     
#         if map_func == 'meanabs':
#             function_list.append(chain.__meanabs)
#         elif map_func == 'meansquare':
#             function_list.append(chain.__meansq)
#         elif map_func == 'sumabs':
#             function_list.append(chain.__sumabs)
#         elif map_func == 'sumsquare':
#             function_list.append(chain.__sumsq)
#         elif map_func == None:
#             function_list.append(chain.__return0)
#         else:
#             raise Exception("the map_func argument is not set to correct value.")
#             
#             
#         if diff_func == 'meanabs':
#             function_list.append(chain_crf.__meanabs)
#         elif diff_func == 'meansquare':
#             function_list.append(chain_crf.__meansq)
#         elif diff_func == 'sumabs':
#             function_list.append(chain_crf.__sumabs)
#         elif diff_func == 'sumsquare':
#             function_list.append(chain_crf.__sumsq)
#         elif diff_func == None:
#             function_list.append(chain_crf.__return0)
#         else:
#             raise Exception("the diff_func argument is not set to correct value.")
#       
# =============================================================================
        self.sigma_mc = sigma_mc
        #self.sigma_data = sigma_data
        #self.map_func = map_func
        #self.diff_func = diff_func
        #self.loss_function_list = function_list

    
    def loss(self, massConvResidual, dataDiff):
        """Computes the value of the loss function for a candidate topography.

        Args:
            massConvResidual (np.ndarray): A 2D array of the mass flux residuals field.
            dataDiff (np.ndarray): A 2D array of the difference between the candidate topography and observed bed elevation. For multiprocessing, can use any placeholder for dataDiff

        Returns:
            total_loss (float): The combined, weighted loss value.
            loss_mc (float): The loss component from mass conservation.
            loss_data (float): The loss component from data misfit.
        """
        
        #f1 = self.loss_function_list[0]
        #f2 = self.loss_function_list[1]
            
        # TODO: is it inappropriate to use sum when the two loss have unequal number of grid cells?
        #loss_mc = f1(massConvResidual, self.mc_region_mask) / (2*self.sigma_mc**2)
        #loss_data = f2(dataDiff, (self.data_mask==1)&(self.data_region_mask==1)) / (2*self.sigma_data**2)
        
        loss_mc = np.nansum(np.square(massConvResidual[self.mc_region_mask == 1]))  / (2*self.sigma_mc**2)
        loss_data = 0
        
        return loss_mc + loss_data, loss_mc, loss_data
    
    def set_random_generator(self, rng_seed = None):
        """
        Set the random generator for the chain to maintain replicability
        Notice that once set_random_generator is called, the random generator for all following call of "run()" will use the continuous sequence of randomness defined by the random generator here. In other words, no other object of random generator will be created unless the set_random_generator() is called again.
        For large scale chain's run() function, the random generator of the random field will be replaced by the current random generator of the chain
        Default: to run a complete chain, call set_random_generator only once before the chain started.

        Args:
            rng_seed (None, int, or numpy.random._generator.Generator): set the random number generator for the chain, either by assigning a random default rng (rng_seed = None), a random generator with seed of rng_seed (rng_seed has type int), or a random generator defined by rng_seed (rng_seed is a generator itself).
        """
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
            chain.seed = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
            
        self.rng = rng
        
    def set_sample_points_locations(self, loc):
        '''
        Set the x and y coordinates, where the bed elevation will be recorded and returned by MCMC chain

        Parameters
        ----------
        loc (np.ndarray) : a 2D array of shape (n, 2) of x and y coordinates, where the bed elevation values through the chain is collected. 

        Returns
        -------
        None.

        '''
        self.sample_loc = loc
      
class chain_crf(chain):
    """
    Inherit the chain class. Used for creating random field-based MCMC chains. Can choose between conditional or unconditional random fields.
    
    Parameters in addition to chain's parameters:
        block_type (String): Specifying uncondtional or conditional random field, 'CRF_weight' or 'RF'.
        crf_data_weight (numpy.ndarray): A 2D array of the data weight to ensure that the updates will not perturb the conditioning data.
    """ 
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_region(update_in_region, region_mask)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')
        print('an RandField object also need to be created correctly and passed in set_crf_data_weight(RF) and in run(n_iter, RF)')
        return
    
    def set_update_type(self, block_type):
        """
        Set types of the perturbation blocks. 
        For the current version of the algorithm, can choose from unconditional random field ('RF') or conditional random field ('CRF') created by logistic weighting (CRF_weight)
        The function has no returns. Its effect can be checked in the object's parameter 'block_type'
        
        Args:
            block_type (str): 'CRF_weight' or 'RF'.
        
        Raises:
            ValueError: If block_type is invalid.
        """
            
        if block_type == 'CRF_rbf':
            print('The update block is set to conditional random field generated by rbf method (not implemented yet)')
        elif block_type == 'CRF_weight':
            print('The update block is set to conditional random field generated by calculating weights with logistic function')
        elif block_type == 'RF':
            print('The update block is set to Random Field')
        else:
            raise ValueError('The block_type argument should be one of the following: CRF_weight, CRF_rbf, RF')
        
        self.block_type = block_type
            
        return
    
    def set_crf_data_weight(self, RF):
        """
        Calculate and store conditioning random field weights from conditioning data.
        The function has no returns. Its effect can be checked in the object's parameter 'crf_data_weight'
        
        Args:
            RF (RandField): A RandField object with the logistic function parameters configured.
        """
        
        crf_weight, dist, dist_rescale, dist_logi = RF.get_crf_weight(self.xx,self.yy,self.data_mask)
        self.crf_data_weight = crf_weight


    def run(self, n_iter, RF, only_save_last_bed=False, info_per_iter = 1000, plot=True, progress_bar=True):
        """Runs the MCMC sampling chain to generate topography realizations.

        Args:
            n_iter (int): The total number of MCMC iterations to perform.
            RF (RandField): An initialized `RandField` object used to generate the topography perturbations.
            rng_seed (int, optional): A seed for the random number generator to ensure reproducibility. Defaults to None.
            only_save_last_bed (bool): If True, only the final topography is returned. If False, the topography from every iteration is saved, which requires more memory.
            info_per_iter (int): The iteration interval for printing progress updates, such as loss and acceptance rate, to the console.

        Returns:
            bed_cache (np.ndarray): A 4D array of saved topographies if `only_save_last_bed` is False, or a 2D array of the final topography if True.
            loss_mc_cache (np.ndarray): A 1D array of the mass conservation loss at each iteration. If no mass conservation loss is set, will return zeros.
            loss_data_cache (np.ndarray): A 1D array of the data misfit loss at each iteration. If no data misfit loss is set, will return zeros.
            loss_cache (np.ndarray): A 1D array of the total loss (mass conservation loss + data misfit loss) at each iteration.
            step_cache (np.ndarray): A 1D boolean array indicating whether the proposal was accepted at each iteration.
            resampled_times (np.ndarray): A 2D array counting how many times each grid cell was part of an accepted proposal.
            blocks_cache (np.ndarray): A 2D array logging the location and size `[row, col, height, width]` of the proposed update block at each iteration.
        """
        # synchronize the random generator with RF object
        rng = self.rng
        
        if not isinstance(RF, RandField):
            raise TypeError('The arugment "RF" has to be an object of the class RandField')
        
        # initialize storage
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        loss_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        if not only_save_last_bed:
            bed_cache = np.zeros((n_iter, self.xx.shape[0], self.xx.shape[1]))
        blocks_cache = np.full((n_iter, 4), np.nan)
        resampled_times = np.zeros(self.xx.shape)
        
        # if the user request to return bed elevation in some sampling locations
        if not (self.sample_loc is None):
            sample_values = np.zeros((self.sample_loc.shape[0], n_iter))
            
            # convert sample_loc from x and y locations to i and j indexes
            sample_loc_ij = np.zeros(self.sample_loc.shape, dtype=np.int16)
            for k in range(self.sample_loc.shape[0]):
                sample_i,sample_j = np.where((self.xx == self.sample_loc[k,0]) & (self.yy == self.sample_loc[k,1]))
                sample_loc_ij[k,:] = [int(sample_i[0]), int(sample_j[0])]
                
            sample_values[:,0] = self.initial_bed[sample_loc_ij[:,0],sample_loc_ij[:,1]]
        
        bed_c = self.initial_bed

        resolution = self.resolution
        
        # initialize loss
        mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        data_diff = bed_c - self.cond_bed
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)

        loss_cache[0] = loss_prev
        loss_data_cache[0] = loss_prev_data
        loss_mc_cache[0] = loss_prev_mc
        step_cache[0] = False
        if not only_save_last_bed:
            bed_cache[0] = bed_c
        
        #crf_weight = self.crf_data_weight

        if plot:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12,5))
            (line_loss,) = ax_loss.plot([], [], color='tab:blue', label='Loss')
            (line_acc,)  = ax_acc.plot([], [], color='tab:green', label='Acceptance Rate')
            #NOTE use get_mass_conservation_residual on BedMachine data
            # bm_loss = 
            # ax_loss.axhline(bm_loss, ls='--', label='BedMachine loss') 
            
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("MCMC Loss")

            ax_acc.set_xlabel("Iteration")
            ax_acc.set_ylabel("Acceptance Rate (%)")
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title("MCMC Acceptance Rate")

            ax_loss.legend()
            ax_acc.legend()
            
            display_handle = display.display(fig, display_id=True)
            plt.tight_layout()

        # Track acceptance rate
        accepted_count = 0
        acceptance_rates = []

        if progress_bar == True:
            chain_id = getattr(self, 'chain_id', 0)
            seed = getattr(self, 'seed', 'Unknown')
            tqdm_position = getattr(self, 'tqdm_position', 0)

            iterator = tqdm(range(1, n_iter),
                            desc=f'Chain {chain_id} | Seed {seed}',
                            position=tqdm_position,
                            leave=True)
        else:
            iterator = range(1,n_iter)

            chain_id = getattr(self, 'chain_id', 'Unknown')
            output_line = getattr(self, 'tqdm_position', 0) + 2 # Reserve first line for header
            seed = getattr(self, 'seed', 'Unknown')
            
        iter_start_time = time.time()
        #pbar = tqdm(range(1,n_iter))
        for i in iterator:
                        
            f = RF.get_rfblock()
            block_size = f.shape
            
            # determine the location of the block
            if self.update_in_region:
                while True:
                    indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                    indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                    if self.region_mask[indexx,indexy] == 1:
                        break
            else:
                indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                
            #record block
            blocks_cache[i,:]=[indexx,indexy,block_size[0],block_size[1]]

            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = np.max((0,int(indexx-block_size[0]/2)))
            bxmax = np.min((bed_c.shape[0],int(indexx+block_size[0]/2)))
            bymin = np.max((0,int(indexy-block_size[1]/2)))
            bymax = np.min((bed_c.shape[1],int(indexy+block_size[1]/2)))
            
            #find the index of the block side in the coordinate of the block
            mxmin = np.max([block_size[0]-bxmax,0])
            mxmax = np.min([bed_c.shape[0]-bxmin,block_size[0]])
            mymin = np.max([block_size[1]-bymax,0])
            mymax = np.min([bed_c.shape[1]-bymin,block_size[1]])
            
            #perturb
            if self.block_type == 'CRF_weight': # weighted random field perturbation
                perturb = f[mxmin:mxmax,mymin:mymax]*self.crf_data_weight[bxmin:bxmax,bymin:bymax]
            else: #random field perturbation
                perturb = f[mxmin:mxmax,mymin:mymax]

            bed_next = bed_c.copy()
            bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb
            
            if self.update_in_region:
                bed_next = np.where(self.region_mask, bed_next, bed_c)
            else:
                bed_next = np.where(self.grounded_ice_mask, bed_next, bed_c)

            # Define Padded Block to solve gradient & mass conservation residual (MSR)
            pad = 1 
            c_xmin = np.max([0, bxmin - pad])              # neighbor to the left boundary
            c_xmax = np.min([bed_c.shape[0], bxmax + pad]) # neighbor to the right boundary
            c_ymin = np.max([0, bymin - pad])              # neighbor to the lower boundary
            c_ymax = np.min([bed_c.shape[1], bymax + pad]) # neighbor to the upper boundary

            # Define the BLOCK index to compute MSR -- which needs neighbors for np.gradient
            local_bed = bed_next[c_xmin:c_xmax, c_ymin:c_ymax]
            local_surf = self.surf[c_xmin:c_xmax, c_ymin:c_ymax]
            local_velx = self.velx[c_xmin:c_xmax, c_ymin:c_ymax]
            local_vely = self.vely[c_xmin:c_xmax, c_ymin:c_ymax]
            local_dhdt = self.dhdt[c_xmin:c_xmax, c_ymin:c_ymax]
            local_smb  = self.smb[c_xmin:c_xmax, c_ymin:c_ymax]

            local_mc_res = Topography.get_mass_conservation_residual(local_bed, local_surf, local_velx, local_vely, local_dhdt, local_smb, resolution)   
            mc_res_candidate = mc_res.copy()
            
            # Our TARGET slice index
            valid_x_start = bxmin - c_xmin
            valid_x_end = valid_x_start + (bxmax - bxmin)
            valid_y_start = bymin - c_ymin
            valid_y_end = valid_y_start + (bymax - bymin)
            mc_res_candidate[bxmin:bxmax, bymin:bymax] = local_mc_res[valid_x_start:valid_x_end, valid_y_start:valid_y_end]

            data_diff = bed_next - self.cond_bed
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res_candidate,data_diff)
           
            #make sure no bed elevation is greater than surface elevation
            block_thickness = self.surf[bxmin:bxmax,bymin:bymax] - bed_next[bxmin:bxmax,bymin:bymax]
            
            if self.update_in_region:
                block_region_mask = self.region_mask[bxmin:bxmax,bymin:bymax]
            else:
                block_region_mask = self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]
            
            if np.sum((block_thickness<=0)[block_region_mask==1]) > 0:
                loss_next = np.inf

            if loss_prev > loss_next:
                acceptance_rate = 1
            else:
                acceptance_rate = min(1,np.exp(loss_prev-loss_next))
            
            u = rng.random()
            if (u <= acceptance_rate):
                bed_c = bed_next.copy()
                
                mc_res = mc_res_candidate # Update global residual if new slice is accepted
                loss_prev = loss_next
                loss_prev_mc = loss_next_mc
                loss_cache[i] = loss_next
                loss_mc_cache[i] = loss_next_mc
                loss_prev_data = loss_next_data
                loss_data_cache[i] = loss_next_data
                
                step_cache[i] = True
                if self.update_in_region:
                    resampled_times[bxmin:bxmax,bymin:bymax] += self.region_mask[bxmin:bxmax,bymin:bymax]
                else:
                    resampled_times[bxmin:bxmax,bymin:bymax] += self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]

                accepted_count += 1
                
            else:
                loss_mc_cache[i] = loss_prev_mc
                loss_cache[i] = loss_prev
                loss_data_cache[i] = loss_prev_data
                step_cache[i] = False
            
            if not only_save_last_bed:
                bed_cache[i,:,:] = bed_c
                
            if not (self.sample_loc is None):
                sample_values[:,i] = bed_c[sample_loc_ij[:,0],sample_loc_ij[:,1]]

            if progress_bar:
                # Update tqdm progress bar
                iterator.set_postfix({
                    'chain_id'  :   chain_id,
                    'seed'      :   seed,
                    'mc loss'   :   f'{loss_mc_cache[i]:.3e}',
                    'data loss' :   f'{loss_data_cache[i]:.3e}',
                    'loss'      :   f'{loss_cache[i]:.3e}',
                    'acceptance rate'   :   f'{np.sum(step_cache)/(i+1):.6f}'
                })
            else:
                if i%info_per_iter == 0 or i == 1 or i == n_iter - 1:
                    move_cursor_to_line(output_line)
                    clear_line()
                    progress = i / (n_iter - 1) * 100
                    elapsed = time.time() - iter_start_time
                    iter_per_sec = i / elapsed if elapsed > 0 else 0
                    print(f'Chain {chain_id}: {progress:.1f}% | i: {i} | mc loss: {loss_mc_cache[i]:.3e} | loss: {loss_cache[i]:.3e} | acc: {np.sum(step_cache)/(i+1):.4f} | it/s: {iter_per_sec:.2f} | seed: {str(seed)[:6]}', end='')
                    sys.stdout.flush()

            # Calculate acceptance rate for plot
            total_acceptance = (accepted_count / (i + 1)) * 100
            acceptance_rates.append(total_acceptance)

            if plot:
                if i < 5000:
                    update_interval = 100
                else:
                    update_interval = info_per_iter

                if i % update_interval == 0:
                    # Update loss line
                    line_loss.set_data(range(i + 1), loss_cache[:i + 1])
                    ax_loss.relim()
                    ax_loss.autoscale_view()

                    # Update acceptance rate line
                    line_acc.set_data(range(len(acceptance_rates)), acceptance_rates)
                    ax_acc.set_ylim(0, 100)
                    ax_acc.relim()
                    ax_acc.autoscale_view()

                    display_handle.update(fig)
                
        if not only_save_last_bed:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            if not (self.sample_loc is None):
                return bed_c, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return bed_c, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache

class chain_sgs(chain):
    """
    Inherit the chain class. Used for creating sequential gaussian simulation blocks-based MCMC chains.
    
    Parameters in addition to chain's parameters:
        do_transform (bool): If true, normalize radar measurements with normal score transformation to generate subglacial topography. If false, directly use Sequential Gaussian Simulation on un-normalized subglacial topography.
        nst_trans (scikit-learn.preprocessing.QuantileTransformer): The normal score transformation for the (detrended/not detrended) subglacial topography.
        trend: (numpy.ndarray): A 2D array representing the trend of the subglacial topography
        detrend_map (bool): If 'True', the subglacial topography will be de-trended using parameter 'trend'. If 'False', the topography will not be de-trended.
        vario_type (string): The type of variogram model used for SGS ('Gaussian', 'Exponential', 'Spherical', or 'Matern').
        vario_param (list): A list of parameters defining the variogram model, set by the `set_variogram` method. [azimuth, nugget, major range, minor range, sill, variogram type, smoothness].
        sgs_param (list): A list of parameters controlling the SGS behavior, set by the `set_sgs_param` method. [number of nearest neighbors, searching radius, randomly drop out conditioning data (True or False), dropout rate]
        block_min_x, block_max_x, block_min_y, block_max_y (int): the minimum and maximum width and height of the update block
    """ 
    
    def __init_func__(self):
        print('before running the chain, please set where the block update will be using the object\'s function set_update_in_region(region_mask) and set_update_region(update_in_region)')
        print('please also set up the sgs parameters using set_sgs_param(self, block_size, sgs_param)')
        print('then please set up the loss function using either set_loss_type or set_loss_func')
        
    def set_normal_transformation(self, nst_trans, do_transform = True):
        """
        Set the normal score transformation object (from scikit-learn package) used to normalize the bed elevation.
        The function has no returns. Its effect can be checked in the object's parameter 'nst_trans'
        
        Args:
            nst_trans (QuantileTransformer): A fitted scikit-learn transformer used to normalize input data.
        
        Note:
            This transformation must be fit beforehand (e.g., via `MCMC.fit_variogram`).
        """
        self.do_transform = do_transform
        if do_transform:
            self.nst_trans = nst_trans
        else:
            self.nst_trans = None
      
    def set_trend(self, trend = None, detrend_map = True):
        """
        Set the long-wavelength trend component of the bed topography.
        Notice that detrend topography means that the SGS simulation will only simulate the short-wavelength topography residuals that is not a part of the trend
        The function has no returns. Its effect can be checked in the object's parameter 'trend' and 'detrend_map'
        
        Args:
            trend (np.ndarray): A 2D array, representing the topographic trend.
            detrend_map (bool): If True, remove trend before transforming the bed elevation and add it back after inverse transform.
        
        Raises:
            ValueError: If detrend_map is True but trend has invalid shape.
        """
        
        if detrend_map == True:
            if len(trend)!=len(self.xx) or trend.shape!=self.xx.shape:
                raise ValueError('if detrend_map is set to True, then the trend of the topography, which is a 2D numpy array, must be provided')
            else:
                self.trend = trend
        else:
            self.trend = None
        self.detrend_map = detrend_map
    
    def set_variogram(self, vario_type, vario_range, vario_sill, vario_nugget, isotropic = True, vario_smoothness = None, vario_azimuth = None):
        """
        Specify variogram model and its parameters for SGS interpolation.
        The function has no returns. Its effect can be checked in the object's parameter 'vario_type' and 'vario_param'
        
        Args:
            vario_type (str): Variogram model type. One of 'Gaussian', 'Exponential', 'Spherical', 'Matern'.
            vario_range (float or list): Correlation range(s). One value for isotropic; list of two for anisotropic.
            vario_sill (float): Variogram sill (variance).
            vario_nugget (float): Nugget effect.
            isotropic (bool): Whether the variogram is isotropic. Default is True.
            vario_smoothness (float): Smoothness parameter for Matern model (required if `vario_type` is 'Matern').
            vario_azimuth (float): Azimuth angle for anisotropic variograms in degrees. Units is degrees (360 maximum)
        
        Raises:
            ValueError: If required parameters are missing or in the wrong format.
        """
            
        if (vario_type == 'Gaussian') or (vario_type == 'Exponential') or (vario_type == 'Spherical'):
            print('the variogram is set to type', vario_type)
        elif vario_type == 'Matern':
            if (vario_smoothness == None) or (vario_smoothness <= 0):
                raise ValueError('vario_smoothness argument should be a positive float when the vario_type is Matern')
            else:
                print('the variogram is set to type', vario_type)
        else:
            raise ValueError('vario_type argument should be one of the following: Gaussian, Exponential, Spherical, or Matern')
        
        self.vario_type = vario_type
        
        if isotropic:
            vario_azimuth = 0
            self.vario_param = [vario_azimuth, vario_nugget, vario_range, vario_range, vario_sill, vario_type, vario_smoothness]
        else:
            if (len(vario_range) == 2):
                print('set to anistropic variogram with major range and minor range to be ', vario_range)
                self.vario_param = [vario_azimuth, vario_nugget, vario_range[0], vario_range[1], vario_sill, vario_type, vario_smoothness]
            else:
                raise ValueError ("vario_range need to be a list with two floats to specifying for major range and minor range of the variogram when isotropic is set to False")
    
    def set_sgs_param(self, sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on = False, dropout_rate = 0):
        """
        Set parameters for Sequential Gaussian Simulation (SGS). Details please see implementation of SGS in GStatSim
        The function has no returns. Its effect can be checked in the object's parameter 'sgs_param
        
        Args:
            sgs_num_nearest_neighbors (int): Number of nearest neighbors used in simulation.
            sgs_searching_radius (float): Radius (in meters) to search for neighbors.
            sgs_rand_dropout_on (bool): Whether to randomly drop conditioning points in simulation block.
            dropout_rate (float): Proportion of conditioning data to drop if dropout is enabled (between 0 and 1).
        """
        
        if sgs_rand_dropout_on == False:
            dropout_rate = 0
            print('because the sgs_rand_dropout_on is set to False, the dropout_rate is automatically set to 0')
            
        self.sgs_param = [sgs_num_nearest_neighbors, sgs_searching_radius, sgs_rand_dropout_on, dropout_rate]
    
    def set_block_sizes(self, block_min_x, block_max_x, block_min_y, block_max_y):
        """
        Set minimum and maximum block sizes (in grid cells) for SGS updates.
        The function has no returns. Its effect can be checked in the object's parameter 'block_min_x', 'block_max_x', 'block_min_y', 'block_max_y'
        
        Args:
            block_min_x (int): Minimum width of block in x-direction. Unit in grid cells
            block_max_x (int): Maximum width of block in x-direction.
            block_min_y (int): Minimum height of block in y-direction.
            block_max_y (int): Maximum height of block in y-direction.
        """
        self.block_min_x = block_min_x
        self.block_min_y = block_min_y
        self.block_max_x = block_max_x
        self.block_max_y = block_max_y
    
    def set_random_generator(self, rng_seed = None):
        """
        Set the random generator for the chain to maintain replicability
        Notice that once set_random_generator is called, the random generator for all following call of "run()" will use the continuous sequence of randomness defined by the random generator here. In other words, no other object of random generator will be created unless the set_random_generator() is called again.
        Default: to run a complete chain, call set_random_generator only once before the chain started.

        Args:
            rng_seed (None, int, or numpy.random._generator.Generator): set the random number generator for the chain, either by assigning a random default rng (rng_seed = None), a random generator with seed of rng_seed (rng_seed has type int), or a random generator defined by rng_seed (rng_seed is a generator itself).
        """
        if rng_seed is None:
            rng = np.random.default_rng()
        elif isinstance(rng_seed, int):
            rng = np.random.default_rng(seed=rng_seed)
        elif isinstance(rng_seed, np.random._generator.Generator):
            rng = rng_seed
        else:
            raise ValueError('Seed should be an integer, a NumPy random Generator, or None')
            
        self.rng = rng

    def run(self, n_iter, only_save_last_bed=False, info_per_iter=100, plot=True, progress_bar=True):
        """
        Run the MCMC chain using block-based SGS updates, with the new gstatsim_custom code
        
        Args:
            n_iter (int): Number of iterations in the MCMC chain.
            only_save_last_bed: If true, the function will only return one subglacial topography at the end of iterations. If false, the function will return all subglacial topography in every iteration.
            info_per_iter (int): for every this number of iterations, the information regarding current loss values and acceptance rate will be printed out.
        
        Returns:
            bed_cache (np.ndarray): A 3D array showing subglacial topography at each iteration, or only the last topography.
            loss_mc_cache (np.ndarray): A 1D array of mass conservation loss at each iteration. If the mass conservation loss is not used, return array of 0
            loss_data_cache (np.ndarray): A 1D array of data misfit loss at each iteration. If the data misfit loss is not used, return array of 0
            loss_cache (np.ndarray): A 1D array of total loss at each iteration.
            step_cache (np.ndarray): A 1D array of boolean indicating if the step was accepted.
            resampled_times (np.ndarray): A 2D array of number of times each pixel was updated.
            blocks_cache (np.ndarray): A 1D array of info on block proposals at each iteration, (x coordinate for the center of the block, y coordinate for the center of the block, block size in x-direction, block size in y-direction).
        """
            
        rng = self.rng
        
        rows = self.xx.shape[0]
        cols = self.xx.shape[1]
        
        loss_cache = np.zeros(n_iter)
        loss_mc_cache = np.zeros(n_iter)
        loss_data_cache = np.zeros(n_iter)
        step_cache = np.zeros(n_iter)
        if not only_save_last_bed:
            bed_cache = np.zeros((n_iter, rows, cols))
        blocks_cache = np.full((n_iter, 4), np.nan)
        resampled_times = np.zeros(self.xx.shape)
        
        # if the user request to return bed elevation in some sampling locations
        if not (self.sample_loc is None):
            sample_values = np.zeros((self.sample_loc.shape[0], n_iter))
            
            # convert sample_loc from x and y locations to i and j indexes
            sample_loc_ij = np.zeros(self.sample_loc.shape, dtype=np.int16)
            for k in range(self.sample_loc.shape[0]):
                sample_i,sample_j = np.where((self.xx == self.sample_loc[k,0]) & (self.yy == self.sample_loc[k,1]))
                sample_loc_ij[k,:] = [int(sample_i[0]), int(sample_j[0])]
                
            sample_values[:,0] = self.initial_bed[sample_loc_ij[:,0],sample_loc_ij[:,1]]

        if self.detrend_map:
            bed_c = (self.initial_bed - self.trend).copy()
            cond_bed_c = (self.cond_bed - self.trend).copy()
        else:
            bed_c = self.initial_bed.copy()
            cond_bed_c = self.cond_bed.copy()
       
        if self.do_transform:
            nst_trans = self.nst_trans
            z = nst_trans.transform(bed_c.reshape(-1,1))
            z_cond_bed = nst_trans.transform(cond_bed_c.reshape(-1,1))
        else:
            z = bed_c.copy().reshape(-1,1)
            z_cond_bed = cond_bed_c.copy().reshape(-1,1)
    
        z_cond_bed = z_cond_bed.reshape(self.xx.shape)

        resolution = self.resolution
        
        # initialize loss
        if self.detrend_map == True:
            mc_res = Topography.get_mass_conservation_residual(bed_c + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        else:
            mc_res = Topography.get_mass_conservation_residual(bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
        
        data_diff = bed_c - cond_bed_c
        loss_prev, loss_prev_mc, loss_prev_data = self.loss(mc_res,data_diff)
    
        loss_cache[0] = loss_prev
        loss_mc_cache[0] = loss_prev_mc
        loss_data_cache[0] = loss_prev_data
        step_cache[0] = False
        if not only_save_last_bed:
            bed_cache[0] = bed_c
    
        rad = self.sgs_param[1]
        neighbors = self.sgs_param[0]
        
        if self.vario_param[5] == 'Matern':
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
                's' : self.vario_param[6]
            }
        else:
            vario = {
                'azimuth' : self.vario_param[0],
                'nugget' : self.vario_param[1],
                'major_range' : self.vario_param[2],
                'minor_range' : self.vario_param[3],
                'sill' :  self.vario_param[4],
                'vtype' : self.vario_param[5],
            }

        # plotting for real-time result update
        if plot:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12,5))
            (line_loss,) = ax_loss.plot([], [], color='tab:blue', label='Loss')
            (line_acc,)  = ax_acc.plot([], [], color='tab:green', label='Acceptance Rate')
            #NOTE use get_mass_conservation_residual on BedMachine data
            # bm_loss = 
            # ax_loss.axhline(bm_loss, ls='--', label='BedMachine loss') 
            
            ax_loss.set_xlabel("Iteration")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("MCMC Loss")

            ax_acc.set_xlabel("Iteration")
            ax_acc.set_ylabel("Acceptance Rate (%)")
            ax_acc.set_ylim(0, 100)
            ax_acc.set_title("MCMC Acceptance Rate")

            ax_loss.legend()
            ax_acc.legend()
            
            display_handle = display.display(fig, display_id=True)
            plt.tight_layout()

        # Track acceptance rate
        accepted_count = 0
        acceptance_rates = []

        if progress_bar == True:
            chain_id = getattr(self, 'chain_id', 0)
            seed = getattr(self, 'seed', 'Unknown')
            tqdm_position = getattr(self, 'tqdm_position', 0)

            iterator = tqdm(range(1,n_iter),
                            desc=f'Chain {chain_id} | Seed {seed}',
                            position=tqdm_position,
                            leave=True) 
        else:
            iterator = range(1,n_iter)

            chain_id = getattr(self, 'chain_id', 'Unknown')
            output_line = getattr(self, 'tqdm_position', 0) + 2 # Reserve first line for header
            seed = getattr(self, 'seed', 'Unknown')

        iter_start_time = time.time()        
        for i in range(n_iter):
    
            while True:
                indexx = rng.integers(low=0, high=bed_c.shape[0], size=1)[0]
                indexy = rng.integers(low=0, high=bed_c.shape[1], size=1)[0]
                if self.region_mask[indexx,indexy] == 1:
                    break
    
            block_size_x = rng.integers(low=self.block_min_x, high=self.block_max_x, size=1)[0]
            block_size_y = rng.integers(low=self.block_min_y, high=self.block_max_y, size=1)[0]
    
            blocks_cache[i,:]=[indexx,indexy,block_size_x,block_size_y]
    
            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = np.max((0,int(indexx-block_size_x/2)))
            bxmax = np.min((bed_c.shape[0],int(indexx+block_size_x/2)))
            bymin = np.max((0,int(indexy-block_size_y/2)))
            bymax = np.min((bed_c.shape[1],int(indexy+block_size_y/2)))
    
            if self.do_transform == True:
                bed_tosim = nst_trans.transform(bed_c.reshape(-1,1)).reshape(self.xx.shape)
            else:
                bed_tosim = bed_c.copy()
    
            bed_tosim[bxmin:bxmax,bymin:bymax] = z_cond_bed[bxmin:bxmax,bymin:bymax].copy()
            sim_mask = np.full(self.xx.shape, False)
            sim_mask[bxmin:bxmax,bymin:bymax] = True
            newsim = sgs(self.xx, self.yy, bed_tosim, vario, rad, neighbors, sim_mask = sim_mask, seed=rng)
    
            if self.do_transform == True:
                bed_next = nst_trans.inverse_transform(newsim.reshape(-1,1)).reshape(rows,cols)
            else:
                bed_next = newsim.copy()
            
            if self.detrend_map == True:
                mc_res = Topography.get_mass_conservation_residual(bed_next + self.trend, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            else:
                mc_res = Topography.get_mass_conservation_residual(bed_next, self.surf, self.velx, self.vely, self.dhdt, self.smb, resolution)
            
            data_diff = bed_next - cond_bed_c
            loss_next, loss_next_mc, loss_next_data = self.loss(mc_res,data_diff)
            
            if self.detrend_map == True:
                thickness = self.surf - (bed_next + self.trend)
            else:
                thickness = self.surf - bed_next
            
            if np.sum((thickness<=0)[self.grounded_ice_mask==1]) > 0:
                loss_next = np.inf
            
            if loss_prev > loss_next:
                acceptance_rate = 1
                
            else:
                acceptance_rate = min(1,np.exp(loss_prev-loss_next))
    
            u = rng.random()
            
            if (u <= acceptance_rate):
                bed_c = bed_next
                loss_cache[i] = loss_next
                loss_mc_cache[i] = loss_next_mc
                loss_data_cache[i] = loss_next_data
                step_cache[i] = True
                
                loss_prev = loss_next
                loss_prev_mc = loss_next_mc
                loss_prev_data = loss_next_data
                resampled_times[bxmin:bxmax,bymin:bymax] += 1
            else:
                loss_cache[i] = loss_prev
                loss_mc_cache[i] = loss_prev_mc
                loss_data_cache[i] = loss_prev_data
                step_cache[i] = False
    
            if not only_save_last_bed:
                if self.detrend_map == True:
                    bed_cache[i,:,:] = bed_c + self.trend
                else:
                    bed_cache[i,:,:] = bed_c
                    
            if not (self.sample_loc is None):
                sample_values[:,i] = bed_c[sample_loc_ij[:,0],sample_loc_ij[:,1]]

            if progress_bar:
                # Update tqdm progress bar
                iterator.set_postfix({
                    'chain_id'  :   chain_id,
                    'seed'      :   seed,
                    'mc loss'   :   f'{loss_mc_cache[i]:.3e}',
                    'data loss' :   f'{loss_data_cache[i]:.3e}',
                    'loss'      :   f'{loss_cache[i]:.3e}',
                    'acceptance rate'   :   f'{np.sum(step_cache)/(i+1):.6f}'
                })
            else:
                if i%info_per_iter == 0 or i == 1 or i == n_iter - 1:
                    move_cursor_to_line(output_line)
                    clear_line()
                    progress = i / (n_iter - 1) * 100
                    elapsed = time.time() - iter_start_time
                    iter_per_sec = i / elapsed if elapsed > 0 else 0
                    print(f'Chain {chain_id}: {progress:.1f}% | i: {i} | mc loss: {loss_mc_cache[i]:.3e} | loss: {loss_cache[i]:.3e} | acc: {np.sum(step_cache)/(i+1):.4f} | it/s: {iter_per_sec:.2f} | seed: {str(seed)[:6]}', end='')
                    sys.stdout.flush()

            # Calculate acceptance rate for plot
            total_acceptance = (accepted_count / (i + 1)) * 100
            acceptance_rates.append(total_acceptance)

            if plot:
                if i < 5000:
                    update_interval = 100
                else:
                    update_interval = info_per_iter

                if i % update_interval == 0:
                    # Update loss line
                    line_loss.set_data(range(i + 1), loss_cache[:i + 1])
                    ax_loss.relim()
                    ax_loss.autoscale_view()

                    # Update acceptance rate line
                    line_acc.set_data(range(len(acceptance_rates)), acceptance_rates)
                    ax_acc.set_ylim(0, 100)
                    ax_acc.relim()
                    ax_acc.autoscale_view()

                    display_handle.update(fig)
    
        if self.detrend_map == True:
            last_bed = bed_c + self.trend
        else:
            last_bed = bed_c
    
        if not only_save_last_bed:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
        else:
            if not (self.sample_loc is None):
                return bed_cache, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache, sample_values
            else:
                return last_bed, loss_mc_cache, loss_data_cache, loss_cache, step_cache, resampled_times, blocks_cache
