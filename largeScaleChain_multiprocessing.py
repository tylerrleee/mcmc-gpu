import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gstatsMCMC import Topography
from gstatsMCMC import MCMC
import gstatsim as gs
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
from copy import deepcopy
import time
import multiprocessing as mp
from pathlib import Path
import os
import sys

def largeScaleChain_mp(n_chains,n_workers,largeScaleChain,rf,initial_beds,rng_seeds,n_iters):
    '''
    function to run multiple large scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    largeScaleChain (MCMC.chain_crf): an existing large scale chain that has already been set-up
    rf (MCMC.RandField): an existing RandField instance that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    rng_seeds (list): a list of int used to initialize the random number generator of each chain
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''

    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')
    
    tic = time.time()
    
    params = []
    example_chain = largeScaleChain.__dict__ #retrive parameters from the existing chain / RandField
    example_RF = rf.__dict__

   # modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):   
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]

        run_param = {} # a dictionary of parameters passed in the run() function
        run_param['n_iter'] = n_iters[i]
        run_param['only_save_last_bed']=True # some display parameters are fixed.
        run_param['info_per_iter']=1000
        run_param['plot']=False
        run_param['progress_bar']=False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 2 # 2 lines for header
        run_param['seed'] = rng_seeds[i]

        params.append([deepcopy(chain_param),deepcopy(example_RF),deepcopy(run_param)])

    # Print header and reserve space for progress bars
    print('Running MCMC chains...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # force output into the terminal
    
    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(lsc_run_wrapper, params)

    # Move cursor below chain outputs before printing the timing
    print('\n' * (n_chains + 2))

    toc = time.time()
    print(f'Completed in {toc-tic:.2f} seconds')
    
    return result

def lsc_run_wrapper(param_chain, param_rf, param_run):
    # a function used to initialize chain by input parameters and run the chains

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    chain = MCMC.init_lsc_chain_by_instance(param_chain)
    rf1 = MCMC.initiate_RF_by_instance(param_rf)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = old_stdout

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    chain.seed = param_run.get('seed', 'Unknown')

    result = chain.run(
        n_iter=param_run['n_iter'], 
        RF=rf1, 
        only_save_last_bed=param_run['only_save_last_bed'], 
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )

    return result
    
def smallScaleChain_mp(n_chains,n_workers,smallScaleChain,initial_beds,rng_seeds,n_iters):
    '''
    function to run multiple small scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    smallScaleChain (MCMC.chain_sgs): an existing small scale chain that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    rng_seeds (list): a list of int used to initialize the random number generator of each chain
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''

    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')
    
    tic = time.time()
    
    params = []
    example_chain = smallScaleChain.__dict__ #retrive parameters from the existing chain

    # modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):   
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]

        run_param = {}
        run_param['n_iter'] = n_iters[i]
        run_param['only_save_last_bed']=True # some display parameters are fixed.
        run_param['info_per_iter']=1000
        run_param['plot']=False
        run_param['progress_bar']=False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 2 # 2 lines for header
        run_param['seed'] = rng_seeds[i]

        params.append([deepcopy(chain_param),deepcopy(run_param)])

    # Print header and reserve space for progress bars
    print('Running MCMC chains...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # force output into the terminal
    
    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(msc_run_wrapper, params)

    # Move cursor below chain outputs before printing total time
    print('\n' * (n_chains + 2))
    
    toc = time.time()
    print(f'Completed in {toc-tic:.2f} seconds')
    
    return result
    
def msc_run_wrapper(param_chain, param_rf, param_run):
    # a function used to initialize chain by input parameters and run the chains

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    chain = MCMC.init_msc_chain_by_instance(param_chain)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = old_stdout

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    chain.seed = param_run.get('seed', 'Unknown')

    result = chain.run(
        n_iter=param_run['n_iter'], 
        only_save_last_bed=param_run['only_save_last_bed'], 
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )

    return result

if __name__=='__main__':

    # load compiled bed elevation measurements
    df = pd.read_csv('./Data/KohlerPopeSmith.csv') #FIXME df = pd.read_csv('DenmanDataGridded.csv')
    
    rng_seed = 23198104
    
    # create a grid of x and y coordinates
    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)
    
    xmin = np.min(x_uniq)
    xmax = np.max(x_uniq)
    ymin = np.min(y_uniq)
    ymax = np.max(y_uniq)
    
    cols = len(x_uniq)
    rows = len(y_uniq)
    
    resolution = 500
    
    xx, yy = np.meshgrid(x_uniq, y_uniq)
    
    # load other data
    dhdt = df['dhdt'].values.reshape(xx.shape)
    smb = df['smb'].values.reshape(xx.shape)
    velx = df['velx'].values.reshape(xx.shape)
    vely = df['vely'].values.reshape(xx.shape)
    bedmap_mask = df['bedmap_mask'].values.reshape(xx.shape)
    bedmachine_thickness = df['bedmachine_thickness'].values.reshape(xx.shape)
    bedmap_surf = df['bedmap_surf'].values.reshape(xx.shape)
    highvel_mask = df['highvel_mask'].values.reshape(xx.shape)
    bedmap_bed = df['bedmap_bed'].values.reshape(xx.shape)
    
    bedmachine_bed = bedmap_surf - bedmachine_thickness
    
    # create conditioning data
    # bed elevation measurement in grounded ice region, and bedmachine bed topography elsewhere
    cond_bed = np.where(bedmap_mask == 1, df['bed'].values.reshape(xx.shape), bedmap_bed)
    df['cond_bed'] = cond_bed.flatten()
    
    # create a mask of conditioning data
    data_mask = ~np.isnan(cond_bed)
    
    # normalize the conditioning bed data, saved to df['Nbed']
    data = df['cond_bed'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=1000, output_distribution="normal",random_state=rng_seed,subsample=None).fit(data)
    transformed_data = nst_trans.transform(data)
    df['Nbed'] = transformed_data
    
    # randomly drop out 50% of coordinates. Decrease this value if you have a lot of data and it takes a long time to run
    df_sampled = df.sample(frac=0.5, random_state=rng_seed)
    df_sampled = df_sampled[df_sampled["cond_bed"].isnull() == False]
    df_sampled = df_sampled[df_sampled["bedmap_mask"]==1]
    
    # compute experimental (isotropic) variogram
    coords = df_sampled[['x','y']].values
    values = df_sampled['Nbed']

    '''FIXME    
    maxlag = 80000      # maximum range distance
    n_lags = 60         # num of bins (try decreasing if this is taking too long)
    
    # compute variogram
    V1 = skg.Variogram(coords, values, bin_func='even', 
                       n_lags=n_lags, maxlag=maxlag, normalize=False, 
                       model='matern')
    
    # extract variogram values
    xdata = V1.bins
    ydata = V1.experimental
    
    # Notice: because we randomly drop out some data, the calculation of V1_p won't always obtain the same result
    # To ensure reproducibility of your work, please use a consistent set of V1_p throughout different chains
    V1_p = V1.parameters
    '''
    V1_p = [np.float64(38566.30452359015), np.float64(1.3915876949924022), np.float64(0.7156807016487273), 0]
    
    # load bed generated by Sequential Gaussian Simulation
    sgs_bed = np.loadtxt('./Data/sgs_bed_kps.txt') #FIXME sgs_bed = np.loadtxt('sgs_bed_denman.txt')
    thickness = bedmap_surf - sgs_bed
    sgs_bed = np.where((thickness<=0)&(bedmap_mask==1), bedmap_surf-1, sgs_bed)
    
    grounded_ice_mask = (bedmap_mask == 1)
    
    # initialize a large scale chain to be used as an example to initialize other large scale chain
    largeScaleChain = MCMC.chain_crf(xx, yy, sgs_bed, bedmap_surf, velx, vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
    
    largeScaleChain.set_update_region(True,highvel_mask)
    
    mc_res_bm = Topography.get_mass_conservation_residual(bedmachine_bed,bedmap_surf,velx,vely,dhdt,smb,resolution)
    
    # in multiprocessing, we choose to only use mass flux residual loss in squared sum (Gaussian distribution)
    largeScaleChain.set_loss_type(sigma_mc=5, massConvInRegion=True)
    
    #range_max and range_min changes topographies features' lateral scale
    #by default, I set range_max to variogram range
    range_max_x = 50e3 #in terms of meters in lateral dimension, regardless of resolution of the map
    range_max_y = 50e3
    range_min_x = 10e3
    range_min_y = 10e3
    scale_min = 50 #in terms of meters in vertical dimension, how much you want to multiply the perturbation by
    scale_max = 150
    nugget_max = 0
    random_field_model = 'Matern' # currently only supporting 'Gaussian' or 'Exponential'
    isotropic = True
    smoothness = V1_p[2]
    
    # initialize a RandField instance to be used for all large scale chains
    rf1 = MCMC.RandField(range_min_x, range_max_x, range_min_y, range_max_y, scale_min, scale_max, nugget_max, random_field_model, isotropic, smoothness = smoothness)
    
    min_block_x = 50
    max_block_x = 80
    min_block_y = 50
    max_block_y = 80
    rf1.set_block_sizes(min_block_x, max_block_x, min_block_y, max_block_y)
    
    logis_func_L = 2
    logis_func_x0 = 0
    logis_func_k = 6
    logis_func_offset = 1
    max_dist = V1_p[0] # set to the distance between two points on the map where the correlation vanish / is minimal
    # this controls how fast the perturbation magnitude decay to zero when close to a radar data
    # a large max_dist give slower decay
    rf1.set_weight_param(logis_func_L, logis_func_x0, logis_func_k, logis_func_offset, max_dist, resolution)
    
    # spectral synthesis can generate the random field significantly faster
    rf1.set_generation_method(spectral=True)
    
    ## The function set_crf_data_weight calculate the weight for conditioning to the radar measurements
    ## Calculate this and re-save data weight again if you change weight parameters in 'set_weight_param()'
    # largeScaleChain.set_crf_data_weight(rf1)
    # largeScaleChain.set_update_type('CRF_weight')
    # np.savetxt('data_weight_denman.txt', largeScaleChain.crf_data_weight)
    
    # load and set the data weight for the chain
    largeScaleChain.crf_data_weight = np.loadtxt('./Data/data_weight.txt') #FIXME largeScaleChain.crf_data_weight = np.loadtxt('data_weight_denman.txt')
    largeScaleChain.set_update_type('CRF_weight')
    
    largeScaleChain.set_random_generator(rng_seed = rng_seed)
    
    n_iter = 5000

    n_chains = 4
    n_workers = 4
    
    initial_beds = np.array([sgs_bed, sgs_bed, sgs_bed, sgs_bed])
    
    with open(Path('./Data/200_seeds.txt'), 'r') as f: #FIXME with open(Path('../200_seeds.txt'), 'r') as f:
        lines = f.readlines()
    
    rng_seeds = []
    for line in lines:
        rng_seeds.append(int(line.strip()))
        
    n_iters = [n_iter]*n_chains

    result = largeScaleChain_mp(n_chains, n_workers, largeScaleChain, rf1, initial_beds, rng_seeds, n_iters)
    
    #beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used  = largeScaleChain.run(n_iter=n_iter, RF=rf1, only_save_last_bed=False)
