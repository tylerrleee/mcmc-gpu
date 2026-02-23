
"""
Created on 17/02/2026

Refactored from Topography.py -- niyashao
PyTorch GPU (MPS / CUDA) conversion -- tylerrleee

@author: @tylerrleee
"""

###import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from pyproj import CRS, Transformer
import verde as vd
import os
import csv
import gstatsim as gs
from PIL import Image, ImageFilter

import torch
# ==============
# Torch Helpers
# ==============
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



def to_tensor(arr, device: torch.device = None, dtype = torch.float32) -> torch.Tensor:
    """Convert a numpy array (or anything array-like) to a torch tensor on *device*."""

    if device is None:
        device = DEVICE

    if isinstance(arr, torch.Tensor):
        return arr.to(device = device, 
                      dtype  = dtype)
    return torch.tensor(np.asarray(arr,
                                    dtype = np.float64), 
                                    dtype = dtype, 
                                    device=device)

def to_numpy(t) -> np.ndarray:
    """Convert a torch tensor back to a numpy ndarray (CPU, float32)."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _interpolate(interp_method, fromx, fromy, data, tox, toy, k):
    """Interpolate using the verde library (CPU).  Inputs/outputs are *numpy*
    - Verde uses SciPy/Scikit-learn -- hence no GPU support """

    # np.ndarray check
    fromx, fromy, data = to_numpy(fromx), to_numpy(fromy), to_numpy(data)
    tox, toy = to_numpy(tox), to_numpy(toy)

    if interp_method == 'spline':
        interp = vd.Spline()
    elif interp_method == 'linear':
        interp = vd.Linear()
    elif interp_method == 'kneighbors':
        interp = vd.KNeighbors(k=k)
    else:
        raise ValueError('the interp_method is not correctly defined, exit the function')

    interp.fit((fromx, fromy), data)
    result = interp.predict((tox, toy))
    return result
    

"""load surface mass balance data from the smb data from https://doi.org/10.5194/tc-12-1479-2018 and https://www.projects.science.uu.nl/iceclimate/publications/data/2018/vwessem2018_tc/RACMO_Yearly/

Args:
    dataset_path (str): The file location of the dataset
    xx (2D numpy array of floats): The x-coordinate of the map
    yy (2D numpy array of floats): The y-coordinate of the map
    interp_method (str): The interpolation methods, can choose from 'spline', 'linear', or 'kneighbors'. Details please check python package 'verde'
    k (int): default = 1. Should only be used when interp_method = 'kneighbors', where k define number of neighbors
    smb_time (int): default = 2015. Should be a value between range 1979 and 2016 (inclusive). 
    
Returns:
    smb (2D array of floats): the interpolated annual average SMB for year 'smb_time'
    fig: figure of interpolated SMB and uninterpolated original data
"""

def load_smb_racmo(dataset_path, xx, yy, res, time=2015, interp_method='linear', k=1):
    # check if smb_time is in the range
    if (time > 2016) or (time < 1979):
        raise ValueError("invalid value for time variable")
    
    try:
        ds = xr.open_dataset(dataset_path)
    except FileNotFoundError:
        print("Error: File not found at path: ", dataset_path)
    
    # transform dataset to polar stereographic coordinate
    ### NOTE: if changed dataset, need to change this method
    crs_rotated = CRS('-m 57.295779506 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +lon_0=10.0')
    polar = CRS.from_epsg(3031)
    transformer = Transformer.from_crs(crs_rotated, polar)
    lonlon, latlat = np.meshgrid(ds.rlon.values, ds.rlat.values)
    xx2, yy2 = transformer.transform(lonlon, latlat)
    
    # restrict the domain of interpolation
    msk = (xx2 > xx.min() - res*200) \
            & (xx2 < xx.max() + res*200) \
            & (yy2 > yy.min() - res*200) \
            & (yy2 < yy.max() + res*200)
    ix = xx2[msk]
    iy = yy2[msk]
    max_time = 2016
    time_int = int(time - max_time - 1)
    iz = ds.isel(time=time_int)['smb'].values.squeeze()[msk]
    
    # assume unit is water equivalent mm / yr, correct units to m/yr
    # iz assumes it is a standard Python array
    iz = iz / 920.0
    
    preds_smb = _interpolate(interp_method, ix, iy, iz, xx.flatten(), yy.flatten(), k)
    preds_smb = preds_smb.reshape(xx.shape)

    preds_smb_t = to_tensor(preds_smb)
    iz_tensor   = to_tensor(iz)

    # Create a stack where NaN values are 0
    max_stack = torch.stack([
        torch.nanmax(preds_smb_t), torch.nanmax(iz_tensor)
    ])
    min_stack = torch.stack([
        torch.nanmin(preds_smb_t), torch.nanmin(iz_tensor)
    ])

    # scale bar values
    vmax = torch.max(max_stack)
    vmin = torch.min(min_stack)

    #print(np.max(preds_smb), np.max(iz),np.min(preds_smb), np.min(iz))

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'wspace': -0.1})
    
    im = ax1.pcolormesh(xx, yy, preds_smb, vmin=vmin, vmax=vmax)
    ax1.axis('scaled')
    ax1.set_title(interp_method + ' interpolation')
    plt.colorbar(im, ax=ax1, pad=0.03, aspect=40, label='m/yr')

    im = ax2.scatter(ix, iy, c=iz, s=150, vmin=vmin, vmax=vmax)
    ax2.axis('scaled')
    ax2.set_title('SMB data')
    ax2.set_yticks([])

    plt.colorbar(im, ax=ax2, pad=0.03, aspect=40)
    plt.close()

    return preds_smb, fig


"""load surface elevation change data from https://nsidc.org/data/nsidc-0782/versions/1

Args:
    dataset_path (str): The file location of the dataset
    xx (2D numpy array of floats): The x-coordinate of the map
    yy (2D numpy array of floats): The y-coordinate of the map
    interp_method (str): The interpolation methods, can choose from 'spline', 'linear', or 'kneighbors'. Details please check python package 'verde'
    k (int): Should only be used when interp_method = 'kneighbors', where k define number of neighbors
    begin_year (int): The begin year of the time-averaged period; must be in format of yyyy, where yyyy is four digits of year between 1950 and 2020
    end_year (int): The end year of the time-averaged period; must be in format of yyyy, where yyyy is four digits of year between 1950 and 2020; must be at least one larger than begin_year
    month (int): Which month's data will be used in averaging, for example, 5 means that surface height change between 2014/05 to 2016/05 are counted in the yearly average outcome 
Returns:
    dhdt (2D array of floats): the interpolated annual average change of the surface height between begin_year and end_year
    fig: figure of interpolated surface height change and uninterpolated original data
"""
def load_dhdt(dataset_path, xx, yy, res, interp_method='linear', k=1,
              begin_year=2014, month=5, end_year=2016):
    try:
        ds2 = xr.open_dataset(dataset_path)
    except FileNotFoundError:
        print("Error: File not found at path: ", dataset_path)

        
    ds2 = ds2.sel(
        x=(ds2.x > xx.min() - res*20) & (ds2.x < xx.max() + res*20),
        y=(ds2.y > yy.min() - res*20) & (ds2.y < yy.max() + res*20)
    )

    if month < 1 or month > 11:
        raise ValueError()
    if begin_year < 1950 or begin_year > 2020:
        raise ValueError()
    if end_year < begin_year + 1:
        raise ValueError()

    month_str    = str(month).zfill(2)
    month_p1_str = str(month + 1).zfill(2)
    ref   = ds2.sel(time=slice(str(begin_year)+'-'+month_str+'-01',
                               str(begin_year)+'-'+month_p1_str+'-01'))
    later = ds2.sel(time=slice(str(end_year)+'-'+month_str+'-01',
                               str(end_year)+'-'+month_p1_str+'-01'))

    num_years = int(end_year) - int(begin_year)

    # Torch based DHDT calculation
    later_t = to_tensor(later['height_change'].values)
    ref_t   = to_tensor(ref['height_change'].values)
    dhdt_t  = (later_t - ref_t) / num_years
    dhdt    = to_numpy(dhdt_t)

    xx2, yy2 = np.meshgrid(ds2.x.values, ds2.y.values)

    preds_h = _interpolate(interp_method, xx2.flatten(), yy2.flatten(),
                           dhdt.flatten(), xx.flatten(), yy.flatten(), k)
    preds_h = preds_h.reshape(xx.shape)

    dhdt_t2 = to_tensor(dhdt)
    v_max  = float(torch.nanmax(dhdt_t2))
    v_min  = float(torch.nanmin(dhdt_t2))
    absmax = max(abs(v_max), abs(v_min))
    
    fig, ax = plt.subplots(figsize=(6, 5))
    p = ax.pcolormesh(xx/1000, yy/1000, preds_h, vmin=-1*absmax, vmax=absmax, cmap='RdBu_r')
    plt.axis('scaled')
    ax.set_title('regridded surface height change rate')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    cbar = fig.colorbar(p, pad=0.03, aspect=40)
    cbar.set_label('m')
    plt.close()
    
    return preds_h, fig


"""load surface elevation change data from Todo

Args:
    dataset_path (str): The file location of the dataset
    xx (2D numpy array of floats): The x-coordinate of the map
    yy (2D numpy array of floats): The y-coordinate of the map
    interp_method (str): The interpolation methods, can choose from 'spline', 'linear', or 'kneighbors'. Details please check python package 'verde'
    k (int): Should only be used when interp_method = 'kneighbors', where k define number of neighbors
Returns:
    velx (2D numpy array of floats): the interpolated veloxity in x direction
    vely (2D numpy array of floats): the interpolated veloxity in y direction
    velx_err (2D numpy array of floats): the interpolated veloxity error in x direction
    vely_err (2D numpy array of floats): the interpolated veloxity error in xy direction
"""
def load_vel_measures(dataset_path, xx, yy, res, interp_method='linear', k=1):
    ds2 = xr.open_dataset(dataset_path)

    ds2 = ds2.sel(
        x=(ds2.x > xx.min() - res*20) & (ds2.x < xx.max() + res*20),
        y=(ds2.y > yy.min() - res*20) & (ds2.y < yy.max() + res*20)
    )
    
    xx2, yy2 = np.meshgrid(ds2.x.values, ds2.y.values)
    
    velx_err_raw = ds2['ERRX'].values
    vely_err_raw = ds2['ERRY'].values
    velx_raw     = ds2['VX'].values
    vely_raw     = ds2['VY'].values

    # Interpolate && reshape to match longitude 
    velx_err = _interpolate(interp_method, xx2.flatten(), yy2.flatten(),
                            velx_err_raw.flatten(), xx.flatten(), yy.flatten(), k).reshape(xx.shape)
    vely_err = _interpolate(interp_method, xx2.flatten(), yy2.flatten(),
                            vely_err_raw.flatten(), xx.flatten(), yy.flatten(), k).reshape(xx.shape)
    velx     = _interpolate(interp_method, xx2.flatten(), yy2.flatten(),
                            velx_raw.flatten(), xx.flatten(), yy.flatten(), k).reshape(xx.shape)
    vely     = _interpolate(interp_method, xx2.flatten(), yy2.flatten(),
                            vely_raw.flatten(), xx.flatten(), yy.flatten(), k).reshape(xx.shape)

    
    ds = [velx, vely, velx_err, vely_err]
    titles = ['velocity in x-direction', 'velocity in y-direction', 
              'the error in velocity in x-direction', 'the error in velocity in y-direction']
    fig, axs = plt.subplots(2, 2, figsize=(10.5,7), sharey=True, sharex=True,
                           gridspec_kw={'wspace':-0.01})
    
    for ax, d, t in zip(axs.ravel(), ds, titles):
        im = ax.pcolormesh(xx, yy, d)
        ax.set_title(t)
        ax.axis('scaled')
        plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.close()
       
    return velx, vely, velx_err, vely_err, fig


"""load bedmachine

Args:
    dataset_path (str): The file location of the dataset
    xx (2D numpy array of floats): The x-coordinate of the map
    yy (2D numpy array of floats): The y-coordinate of the map
    interp_method (str): The interpolation methods, can choose from 'spline', 'linear', or 'kneighbors'. Details please check python package 'verde'
    k (int): Should only be used when interp_method = 'kneighbors', where k define number of neighbors
Returns:
    bm_mask (2D numpy array of floats): the mask in BedMachine, recording the type of ice (open ocean, ice free, grounded, floating, etc.). Details see BedMachine document. Interpolated using nearest neighbor interpolation
    bm_source (2D numpy array of floats): the source method used for BedMachine, recording the type of ice (open ocean, ice free, grounded, floating, etc.). Details see BedMachine document. Interpolated using nearest neighbor interpolation
    bm_bed (2D numpy array of floats): the interpolated BedMachine bed elevation.
    bm_surface (2D numpy array of floats): the interpolated BedMachien ice surface elevation.
    bm_errbed (2D numpy array of floats): the interpolated BedMachine bed error.
    fig: visualization of the interpolated data
"""

def load_bedmachine(dataset_path, xx, yy, res, interp_method='linear', k=1):
    dsbm = xr.open_dataset(dataset_path)
    dsbm = dsbm.sel(
        x=(dsbm.x > xx.min() - res*20) & (dsbm.x < xx.max() + res*20),
        y=(dsbm.y > yy.min() - res*20) & (dsbm.y < yy.max() + res*20)
    )

    bm_mask     = dsbm['mask'].values
    bm_source   = dsbm['source'].values
    bm_surface  = dsbm['surface'].values
    bm_bed      = dsbm['bed'].values
    bm_errbed   = dsbm['errbed'].values
    
    xx2, yy2 = np.meshgrid(dsbm.x.values, dsbm.y.values)
    
    print('NOTICE! The categorical data in bedmachine will automatically be interpolated using nearest neighbor interpolation method')
    
    bm_mask = _interpolate('kneighbors', xx2.flatten(), yy2.flatten(), bm_mask.flatten(), xx.flatten(), yy.flatten(), k)
    bm_mask = bm_mask.reshape(xx.shape)

    bm_source = _interpolate('kneighbors', xx2.flatten(), yy2.flatten(), bm_source.flatten(), xx.flatten(), yy.flatten(), k)
    bm_source = bm_source.reshape(xx.shape)
    
    bm_bed = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_bed.flatten(), xx.flatten(), yy.flatten(), k)
    bm_bed = bm_bed.reshape(xx.shape)
    
    bm_surface = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_surface.flatten(), xx.flatten(), yy.flatten(), k)
    bm_surface = bm_surface.reshape(xx.shape)
    
    bm_errbed = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_errbed.flatten(), xx.flatten(), yy.flatten(), k)
    bm_errbed = bm_errbed.reshape(xx.shape)
    
    ds = [bm_mask, bm_source, bm_bed, bm_surface, bm_errbed]
    titles = ['BedMachine mask', 'BedMachine source', 'BedMachine bed',
               'BedMachine surface', 'BedMachine bed error']
    
    fig, axs = plt.subplots(3, 2, figsize=(10.5,7), sharey=True, sharex=True,
                           gridspec_kw={'wspace':-0.01})
    
    for ax, d, t in zip(axs.ravel(), ds, titles):
        im = ax.pcolormesh(xx, yy, d)
        ax.set_title(t)
        ax.axis('scaled')
        plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.close()
    
    return bm_mask, bm_source, bm_bed, bm_surface, bm_errbed, fig

"""load bedmap

Args:
    dataset_path (str): The file location of the dataset
    xx (2D numpy array of floats): The x-coordinate of the subglacial topography
    yy (2D numpy array of floats): The y-coordinate of the subglacial topography
    res (int): resolution of the subglacial topography
    interp_method (str): The interpolation methods, can choose from 'spline', 'linear', or 'kneighbors'. Details please check python package 'verde'
    k (int): Should only be used when interp_method = 'kneighbors', where k define number of neighbors

Returns:
    bm_mask (2D numpy array of floats): the mask in BedMachine, recording the type of ice (open ocean, ice free, grounded, floating, etc.). Details see BedMachine document. Interpolated using nearest neighbor interpolation
    bm_source (2D numpy array of floats): the source method used for BedMachine, recording the type of ice (open ocean, ice free, grounded, floating, etc.). Details see BedMachine document. Interpolated using nearest neighbor interpolation
    bm_bed (2D numpy array of floats): the interpolated BedMachine bed elevation.
    bm_surface (2D numpy array of floats): the interpolated BedMachien ice surface elevation.
    bm_errbed (2D numpy array of floats): the interpolated BedMachine bed error.
    fig: visualization of the interpolated data
"""

def load_bedmap(dataset_path,xx,yy,res,interp_method='linear',k=1):
    dsbm = xr.open_dataset(dataset_path)
    dsbm = dsbm.sel(x=(dsbm.x > xx.min() - res*20) & (dsbm.x < xx.max() + res*20), y=(dsbm.y > yy.min() - res*20) & (dsbm.y < yy.max() + res*20))
    
    bm_surf = dsbm['surface_topography'].values
    bm_bed  = dsbm['bed_topography'].values
    bm_bed_uncertainty = dsbm['bed_uncertainty'].values
    bm_mask = dsbm['mask'].values
    
    xx2, yy2 = np.meshgrid(dsbm.x.values, dsbm.y.values)
    
    print('NOTICE! The region mask in bedmap will automatically be interpolated using nearest neighbor interpolation method')
    
    bm_mask = _interpolate('kneighbors', xx2.flatten(), yy2.flatten(), bm_mask.flatten(), xx.flatten(), yy.flatten(), k)
    bm_mask = bm_mask.reshape(xx.shape)
    
    bm_bed = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_bed.flatten(), xx.flatten(), yy.flatten(), k)
    bm_bed = bm_bed.reshape(xx.shape)
    
    bm_surf = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_surf.flatten(), xx.flatten(), yy.flatten(), k)
    bm_surf = bm_surf.reshape(xx.shape)
    
    bm_bed_uncertainty = _interpolate(interp_method, xx2.flatten(), yy2.flatten(), bm_bed_uncertainty.flatten(), xx.flatten(), yy.flatten(), k)
    bm_bed_uncertainty = bm_bed_uncertainty.reshape(xx.shape)
    
    ds = [bm_mask, bm_surf, bm_bed, bm_bed_uncertainty]
    titles = ['BedMap mask', 'BedMap surface', 'BedMap bed','BedMap bed uncertainty']
    
    fig, axs = plt.subplots(2, 2, figsize=(10.5,7), sharey=True, sharex=True,
                           gridspec_kw={'wspace':-0.01})
    
    for ax, d, t in zip(axs.ravel(), ds, titles):
        im = ax.pcolormesh(xx, yy, d)
        ax.set_title(t)
        ax.axis('scaled')
        plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.close()
        
    return bm_mask, bm_surf, bm_bed, bm_bed_uncertainty, fig

def _thickToEle(x, x_axis, y_axis, surface):
    bmx = (np.round(x['x'])*1000)
    bmy = (np.round(x['y'])*1000)

    sid = np.intersect1d(np.where(x_axis==bmx),np.where(y_axis==bmy))

    if sid.shape[0] == 1:
        return surface[sid][0] - x['land_ice_thickness (m)']
    else:
        return -9999

"""load bed elevation measurement from bedmap2 or bedmap3 dataset

Args:
    folder_path (str): The location of the folder that contain csv files of radar measurement data retrieved from bedmap2 and bedmap3
    xx (2D numpy array of floats): The x-coordinate of the map
    yy (2D numpy array of floats): The y-coordinate of the map
    output_csv (str): the location and name of the output csv file
    include_only_thickness_data (bool): if True, convert data that bed = -9999, but surface != -9999 and thickness != -9999
Returns:
    df (pandas dataframe): the dataframe that contains the bed elevation measurements that are included, based on parameter include_only_thickness_data
    df_out (pandas dataframe): the dataframe that are excluded from the df
    fig: figure as a overview of the dataset
"""
# TODO: geoid correction, make include_only_thickness_data useful
def load_radar(folder_path, output_csv, include_only_thickness_data = False):

    if not os.path.isdir(folder_path):
        raise FileNotFoundError('the folder_path provided is not a directory')
    
    filename_list = os.listdir(folder_path)
    df_list = [] #a list of dataframe containing radar data, each dataframe contains radar data from one file
    mf = open(folder_path + "/radar_metadata.txt", "a")
    for filename in filename_list:
        
        if filename[-4:] != '.csv':
            continue
        
        with open(folder_path + '/' + filename) as fp:
            mf.write(filename + '\n')
            
            reader = csv.reader(fp)
            for j in range(18):
                headers = next(reader)        # The header row is now consumed
                mf.write('\t'.join(headers) + '\n')
    
        df = pd.read_csv(folder_path + '/' + filename,skiprows=18)
        df['file'] = filename
        df_list.append(df)
        mf.write('\n')
    mf.close()
    
    print('the following files are loaded: ')
    print(filename_list)
    print('the metadata for each radar compaign is saved in ', folder_path+"/radar_metadata.txt")
    
    df = pd.concat(df_list)
    
    source_crs = 'epsg:4326'
    target_crs = 'epsg:3031'
    lonlat_to_xy = Transformer.from_crs(source_crs,target_crs)
    x,y = lonlat_to_xy.transform(df['latitude (degree_north)'], df['longitude (degree_east)'])
    df['x'] = x.tolist()
    df['y'] = y.tolist()
    
    df_bedmap3 = df[df['file'].str[-7:-4] == 'BM3'].copy()
    df_bedmap2 = df[df['file'].str[-7:-4] == 'BM2'].copy()
       
    df_out = df[df['bedrock_altitude (m)'] == -9999].copy()
    df = df[df['bedrock_altitude (m)'] != -9999]  
    df = df.reset_index()
    df = df.rename(columns={"bedrock_altitude (m)": "bed"})
    
    # remove unnecessary columns and rows
    del df['trajectory_id']
    del df['trace_number']
    del df['longitude (degree_east)']
    del df['latitude (degree_north)']
    del df['date']
    del df['time_UTC']
    del df['two_way_travel_time (m)']
    del df['aircraft_altitude (m)']
    del df['along_track_distance (m)']
    del df['land_ice_thickness (m)']
    del df['index']
    print('There are in total', df.shape[0], 'datapoints')
    
    df.to_csv(output_csv,index=False, header=True)
    print('output csv file saved as ', output_csv)
    
    # Convert to tensor
    bed_t = to_tensor(df['bed'].values)
    v_min = float(torch.min(bed_t))
    v_max = float(torch.max(bed_t))
     
    df_sparse1 = df_bedmap3[df_bedmap3.index % 10 == 1]
    df_sparse2 = df_bedmap2[df_bedmap2.index % 10 == 1]
    df_sparse3 = df_out[df_out.index % 10 == 1]
    df_sparse4 = df[df.index % 10 == 1]
    
    ds = [df_sparse1['bedrock_altitude (m)'], df_sparse2['bedrock_altitude (m)'], df_sparse3['bedrock_altitude (m)'], df_sparse4['bed']]
    xs = [df_sparse1['x'],df_sparse2['x'],df_sparse3['x'],df_sparse4['x']]
    ys = [df_sparse1['y'],df_sparse2['y'],df_sparse3['y'],df_sparse4['y']]
    titles = ['bedmap3', 'bedmap2', 'the excluded measurements', 'the final bed elevation']
    
    fig, axs = plt.subplots(2, 2, figsize=(8.5,7), sharey=True, sharex=True,
                           gridspec_kw={'wspace':-0.01})
    
    for ax, d, x, y, t in zip(axs.ravel(), ds, xs, ys, titles):
        im = ax.scatter(x, y, c = d, vmin = v_min, vmax = v_max, marker = '.', s = .5)
        ax.set_title(t)
        ax.axis('scaled')
        plt.colorbar(im, ax=ax, pad=0.03, aspect=40)
    plt.close()
    
    return df, df_out, fig

# TODO: method title
# the method is adopted from GStatSim python package
"""grid compiled radar data into square grid with given resolution. 
Notice that this function is a slight modification to the grid_data function in gstatsim python package.
The extra argument xmin, xmax, ymin, ymax ensures the grid location even when no radar data is available at those locations

Args:
    df (pandas dataframe): the DataFrame containing all the radar data
    x_name, y_name, z_name (strings): the label of the column in df representing x coordinate, y coordinate and bed elevation
    res (int): resolution of the square grid, with unit of meters
    xmin, xmax, ymin, ymax (ints): the boundary of domain, with unit of meters
Returns:
    df_grid (pandas dataframe): the DataFrame containing all the gridded radar data. This gridding is produced by averaging all radar data within each grid cell. The dataframe also record number of radar point measurements used in each grid. If no data available at the grid cell, the 'Z' column has nan value.
    grid_matrix (numpy array): a 2D numpy array of gridded radar data
    rows (int): number of rows in the gridded map
    cols (int): number of columns in the gridded map
"""
def grid_data(df, x_name, y_name, z_name, res, xmin, xmax, ymin, ymax):
    
    df = df.rename(columns = {x_name: "X", y_name: "Y", z_name: "Z"})

    grid_coord, cols, rows = gs.Gridding.make_grid(xmin, xmax, ymin, ymax, res) 

    np_data   = df[['X', 'Y', 'Z']].to_numpy()
    np_resize = np.copy(np_data)
    origin     = np.array([xmin, ymin])
    resolution = np.array([res, res])
    
    # shift and re-scale the data by subtracting origin and dividing by resolution
    np_resize[:, :2] = np.rint((np_resize[:, :2] - origin) / resolution)


    data_t   = to_tensor(np_data, device=device)
    resize_t = to_tensor(np_resize, device=device)

    grid_sum_t   = torch.zeros((rows, cols), dtype=torch.float64, device=device)
    grid_count_t = torch.zeros((rows, cols), dtype=torch.float64, device=device)

    xindex = resize_t[:, 1].long()
    yindex = resize_t[:, 0].long()
    # Constraints for each
    valid_constraints  = (xindex < rows) & (yindex < cols) & (xindex >= 0) & (yindex >= 0)

    xindex_v = xindex[valid_constraints]
    yindex_v = yindex[valid_constraints]
    z_v      = data_t[:, 2][valid_constraints]

    flat_idx = xindex_v * cols + yindex_v
    grid_sum_flat   = grid_sum_t.view(-1)
    grid_count_flat = grid_count_t.view(-1)
    grid_sum_flat.scatter_add_(0, flat_idx, z_v)
    grid_count_flat.scatter_add_(0, flat_idx, torch.ones_like(z_v))

    # Nonzero Division Check
    grid_matrix_t = torch.where( 
        grid_count_t > 0,
        grid_sum_t / grid_count_t,
        torch.tensor(float('nan'), dtype=torch.float32, device=device)
    )

    grid_matrix = to_numpy(grid_matrix_t)
    grid_sum    = to_numpy(grid_sum_t)
    grid_count  = to_numpy(grid_count_t)

    grid_array = grid_matrix.reshape(rows * cols)
    grid_sum_f = grid_sum.reshape(rows * cols)
    grid_count_f = grid_count.reshape(rows * cols)

    grid_total = np.array([grid_coord[:, 0], grid_coord[:, 1],
                           grid_sum_f, grid_count_f, grid_array])
    df_grid = pd.DataFrame(grid_total.T, columns=['X', 'Y', 'Sum', 'Count', 'Z'])
    grid_matrix = np.flipud(grid_matrix)

    return df_grid, grid_matrix, rows, cols


"""
Args:
    geoid_dataset_path (str): The file location of the geoid file
    xx (2D numpy array of floats): The x-coordinate of the target map (usually is also the subglacial topography map)
    yy (2D numpy array of floats): The y-coordinate of the target map
    res (int): target resolution
    
Returns:
    geoid (2D numpy array of floats): the geoid height anomaly
"""
def convert_geoid(geoid_file_path, xx, yy, res):
    df_geoid = pd.read_csv(geoid_file_path,skiprows=36,header=None,sep=r"\s+",names=['lon','lat','anomalyHeight'])
    latlon = CRS.from_epsg(4326)
    polar = CRS.from_epsg(3031)
    
    res = np.abs(xx[0,0] - xx[1,1])
    
    transformer = Transformer.from_crs(latlon, polar)
    xx2, yy2 = transformer.transform(df_geoid.lat.values, df_geoid.lon.values)
    xl_g = xx2[(xx2 < np.max(xx)+res*20)&(xx2 > np.min(xx)-res*20)&(yy2 < np.max(yy)+res*20)&(yy2 > np.min(yy)-res*20)]
    yl_g = yy2[(xx2 < np.max(xx)+res*20)&(xx2 > np.min(xx)-res*20)&(yy2 < np.max(yy)+res*20)&(yy2 > np.min(yy)-res*20)]
    h_g = df_geoid[(xx2 < np.max(xx)+res*20)&(xx2 > np.min(xx)-res*20)&(yy2 < np.max(yy)+res*20)&(yy2 > np.min(yy)-res*20)].anomalyHeight.values
    
    interp = vd.Linear()
    interp.fit(((xl_g, yl_g)), h_g)
    geoid = interp.predict((xx.flatten(),yy.flatten()))

    return geoid   
    
    
""" find the high velocity region as a mask and a boundary map. 
The idea is to first find the high-velocity, grounded ice region; then smooth the boundary of the mask; then expand the boundary outward

Args:
    velx (2D numpy array of float): velocity in the x-direction
    vely (2D numpy array of float): velocity in the y-direction
    velmag_threshold (float): the threshold of the velocity of the high velocity region
    grounded_ice_mask (2D numpy array of int): mask of grounded ice; when 0, means floating ice / open ocean / land without ice; when 1, means grounded ice
    ocean_mask (2D numpy array of int): mask of floating ice or open coean; when 0, means floating ice or open ocean; when 1, means grounded ice
    distance_max (float): how much distance to expand the smoothed boundary outward
    xx (2D numpy array of float): The x-coordinate of the map
    yy (2D numpy array of float): The y-coordinate of the map
Returns:
    mask_final: the final high velocity region mask
    TODO: test
"""
def get_highvel_boundary(velx, vely, velmag_threshold, grounded_ice_mask, ocean_mask, 
                         distance_max, xx, yy, smooth_mode=10, batch_size=10000):

    velx_t = to_tensor(velx, device=device)
    vely_t = to_tensor(vely, device=device)
    grounded_ice_mask_t  = to_tensor(grounded_ice_mask.astype(float), device=device)
    ocean_mask_t   = to_tensor(ocean_mask.astype(float), device=device)

    velmag_t = torch.sqrt(velx_t**2 + vely_t**2)
    mask_t   = ((grounded_ice_mask_t > 0) & (velmag_t >= velmag_threshold)) | (ocean_mask_t > 0)

    # Smooth with PIL (CPU) – convert mask to uint8 image
    mask_np  = to_numpy(mask_t).astype(np.uint8)
    image    = Image.fromarray(mask_np * 255)
    image    = image.filter(ImageFilter.ModeFilter(size=smooth_mode))
    mask_mat = (np.array(image) / 255).astype(int)

    # Tensor conversion
    xx_t = to_tensor(xx, device=device)
    yy_t = to_tensor(yy, device=device)
    mask_mat_t = to_tensor(mask_mat, device=device)

    constraint_mask = (mask_mat_t == 1) & (grounded_ice_mask_t == 1)

    # shape: (N_constraints, 2)
    constraint_pts = torch.stack([
        xx_t[constraint_mask], 
        yy_t[constraint_mask]
    ], dim=1)

    # Handle edge case: No constraint points found
    if constraint_pts.size(0) == 0:
        return np.zeros_like(grounded_ice_mask, dtype=bool)
    
    # Flatten the grid into a list of coordinates
    # shape: (H*W, 2)
    grid_pts = torch.stack([xx_t.flatten(), yy_t.flatten()], dim=1)

    # Batch processing
    n_grid = grid_pts.size(0)
    min_dists = torch.empty(n_grid, device=device, dtype=torch.float32)
    
    # Process grid in chunks
    # This loop ensures only allocate (Batch_Size * N_constraints) memory at each iteration
    for i in range(0, n_grid, batch_size):
        # Current batch of grid points: (Batch, 2)
        batch_grid = grid_pts[i : i + batch_size]
        
        # Compute distance matrix: (Batch, N_constraints)
        #  p-norm distance (default p=2 ; Euclidean)
        dists = torch.cdist(batch_grid, constraint_pts)
        
        # Find min distance for each pixel in this batch: (Batch,)
        min_vals, indices = torch.min(dists, dim=1)
        
        # Store results
        min_dists[i : i + batch_size] = min_vals

    # 5. Reshape and Finish
    mask_dist_t = min_dists.reshape(xx.shape)    

    # Mask constraints : with max distance && doesn't include grounded ice
    mask_final_t = (mask_dist_t < distance_max) & (grounded_ice_mask_t > 0)
    mask_final_np = to_numpy(mask_final_t).astype(bool)

    return mask_final_np

""" compute the mass conservation residual from topography

Args:
    bed (2D numpy array of float): the elevation of the subglacial topography, in units of meters
    surf (2D numpy array of float): the elevation of the ice surface, in unit of meters
    velx (2D numpy array of float): velocity in the x-direction, in unit of meters per year
    vely (2D numpy array of float): velocity in the y-direction, in unit of meters per year
    dhdt (2D numpy array of float): rate of changes of surface elevation, in unit of meters per year
    smb (2D numpy array of float): annual surface mass balance, in unit of ice-equivalent meters per year
    resolution (int): resolution of the grid.
Returns:
    res: mass conservation residual given the input parameters
    
Note: need to consider uncertainties due to 
1. firn thickness - ice-equivalent conversion error
2. error due to unmatched data collection time
3. numerical error due to the np.gradient function
4. error in all these day and correlation in the error
"""


def get_mass_conservation_residual_torch(bed, surf, velx, vely, dhdt, smb, res):    
    thick_t = surf - bed

    # torch.gradient returns a list; axis 1 = x (columns), axis 0 = y (rows)
    (dx_t,) = torch.gradient(velx * thick_t, spacing=(float(res),), dim=1)
    (dy_t,) = torch.gradient(vely * thick_t, spacing=(float(res),), dim=0)

    res_t = dx_t + dy_t + dhdt - smb
    return res_t

def get_mass_conservation_residual(bed, surf, velx, vely, dhdt, smb, resolution):
    thick = surf - bed
    
    dx = np.gradient(velx*thick, resolution, axis=1)
    dy = np.gradient(vely*thick, resolution, axis=0)
    
    res = dx + dy + dhdt - smb
    
    return res

# TODO <============
def filter_data_by_std(df_in, rf_bed, cond_bed, num_of_std, xx, yy, shallow, dfmaskname = 'bedmachine_mask'):
    
    df = df_in.copy()
    
    # plot the radar difference, give std
    fig = plt.figure(figsize=(8,6*3))
    gs = fig.add_gridspec(3,1,height_ratios = [2,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    
    rfradardiff = rf_bed - cond_bed
    stdrf = np.std(rfradardiff[~np.isnan(rfradardiff)])
    
    print('the standard deviation of difference to conditioning data is', stdrf)
    
    fig_bed = ax1.pcolormesh(xx/1000,yy/1000,rf_bed,cmap='gist_earth',vmin=-2500,vmax=2000)
    fig_diff = ax1.pcolormesh(xx/1000, yy/1000,rfradardiff,vmin=-1000, vmax=1000,cmap='RdBu')
    plt.colorbar(fig_bed,ax=ax1, aspect=40, label='m',orientation='horizontal')
    plt.colorbar(fig_diff,ax=ax1, aspect=40, label='m',orientation='horizontal')
    ax1.set_xlabel('X [km]')
    ax1.set_ylabel('Y [km]')
    ax1.set_title('final topography minus radar data')
    ax1.axis('scaled')
    
    ax3.pcolormesh(xx/1000, yy/1000,  (rfradardiff<stdrf*num_of_std)&(rfradardiff>-stdrf*num_of_std), cmap='YlGn')
    ax3.set_xlabel('X [km]')
    ax3.set_ylabel('Y [km]')
    ax3.set_title('if exclude positive and negative radardiff')
    ax3.axis('scaled')

    ax2.pcolormesh(xx/1000, yy/1000,  (rfradardiff>-stdrf*num_of_std), cmap='RdPu')
    ax2.set_xlabel('X [km]')
    ax2.set_ylabel('Y [km]')
    ax2.set_title('if only exclude negative radardiff (bed<rf)')
    ax2.axis('scaled')
    
    #exclude data
    df['bedQCrf'] = [np.nan]*df.shape[0]
    df['bedrf'] = rf_bed.flatten()
    num_excluded_data = 0
    for index, row in df.iterrows():
        if ((row[dfmaskname] == 3) | (row[dfmaskname] == 0)): #if in ice shelf
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        # elif (row['bedmachine_mask'] == 0): #or sea floor
        #     df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        elif pd.isna(row['bed']):
            continue
        elif (row['bed'] < row['bedrf'] + stdrf*num_of_std) and (row['bed'] > row['bedrf'] - stdrf*num_of_std) and (~shallow):
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        elif (row['bed'] < row['bedrf'] + stdrf*1.5) and (shallow):
            df.loc[index,'bedQCrf'] = df.loc[index,'bed']
        else:
            num_excluded_data += 1
            
    print('the exclusion rate is',num_excluded_data / df[df['bed'].isnull()==False].shape[0])
            
    return df
