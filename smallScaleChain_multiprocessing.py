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
import scipy as sp

def largeScaleChain_mp(n_chains,n_workers,largeScaleChain,rf,initial_beds,rng_seeds,n_iters,output_path='./Data/output'):
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
    output_path (str): Path to the folder where the user wants to save results

    Returns
    -------
    result: a list of results from all the chains runned.

    '''
    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')
    
    tic = time.time()
    
    params = []

    # Retrive parameters from the existing chain / RandField
    example_chain = largeScaleChain.__dict__ 
    example_RF = rf.__dict__

   # Modify some of the parameters based on the input rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):   
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]

        run_param = {} # A dictionary of parameters passed in the run() function
        run_param['n_iter'] = n_iters[i]
        run_param['only_save_last_bed']=True # Some display parameters are fixed.
        run_param['info_per_iter']=1000
        run_param['plot']=False
        run_param['progress_bar']=False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 2 # 2 lines for header
        run_param['seed'] = rng_seeds[i]
        run_param['output_path'] = str(Path(output_path) / 'LargeScaleChain')

        params.append([deepcopy(chain_param),deepcopy(example_RF),deepcopy(run_param)])

    # Print header and reserve space for progress bars
    print('Running MCMC chains...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # Force output into the terminal
    
    # The multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(lsc_run_wrapper, params)

    # Move cursor below chain outputs before printing the timing
    print('\n' * (n_chains + 2))

    toc = time.time()
    print(f'Completed in {toc-tic:.2f} seconds')
    
    return result

def lsc_run_wrapper(param_chain, param_rf, param_run):
    '''
    A function used to initialize chain by input parameters and run the chains

    Parameters
    ----------
    param_chain (dict): Dictionary containing parameters needed to initialize chain
    param_rf (dict): Dictionary containing parameters needed to initialize random field
    param_run (dict): Dictionary containing parameters needed to run chain

    Returns
    -------
    result (tuple): A tuple containing the results of the run

    '''

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    chain = MCMC.init_lsc_chain_by_instance(param_chain)
    rf1 = MCMC.initiate_RF_by_instance(param_rf)

    # Restore stdout after initialization
    sys.stdout.close()
    sys.stdout = old_stdout

    # Setup output path
    output_path = param_run.get('output_path', './Data/LargeScaleChain')
    seed = param_run['seed']
    n_iter = param_run['n_iter']
    seed_folder = Path(output_path) / f'{str(seed)[:6]}'

    # Check for existing bed files (to resume progress)
    existing_beds = list(seed_folder.glob('bed_*.txt'))
    cumulative_iters = 0
    previous_results = None
    files_to_delete = []

    # Prepare to merge/concatenate existing files with new results
    if existing_beds:
        bed_file = existing_beds[0] # Existing bed file
        
        # Extract iteration count from filename
        filename = bed_file.stem  # Gets 'bed_100k' from 'bed_100k.txt'
        iter_str = filename.split('_')[1].replace('k', '')  # Gets '100' from 'bed_100k'
        iter_count = int(iter_str)
        cumulative_iters = iter_count * 1000  # Convert back to actual iterations
        
        # Load the most recent bed file
        most_recent_bed = np.loadtxt(bed_file)
        
        # Update the chain's initial bed
        chain.initial_bed = most_recent_bed
        
        # Load all previous result files
        previous_results = {
            'loss_mc': np.loadtxt(seed_folder / f'loss_mc_{iter_count}k.txt'),
            'loss_data': np.loadtxt(seed_folder / f'loss_data_{iter_count}k.txt'),
            'loss': np.loadtxt(seed_folder / f'loss_{iter_count}k.txt'),
            'steps': np.loadtxt(seed_folder / f'steps_{iter_count}k.txt'),
            'resampled_times': np.loadtxt(seed_folder / f'resampled_times_{iter_count}k.txt'),
            'blocks_used': np.loadtxt(seed_folder / f'blocks_used_{iter_count}k.txt')
        }
        
        # Mark files for deletion
        files_to_delete = [
            seed_folder / f'bed_{iter_count}k.txt',
            seed_folder / f'loss_mc_{iter_count}k.txt',
            seed_folder / f'loss_data_{iter_count}k.txt',
            seed_folder / f'loss_{iter_count}k.txt',
            seed_folder / f'steps_{iter_count}k.txt',
            seed_folder / f'resampled_times_{iter_count}k.txt',
            seed_folder / f'blocks_used_{iter_count}k.txt'
        ]

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    chain.seed = param_run.get('seed', 'Unknown')

    # Run the chain
    result = chain.run(
        n_iter=param_run['n_iter'], 
        RF=rf1, 
        only_save_last_bed=param_run['only_save_last_bed'], 
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )
    
    # Unpack results
    beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = result

    # Combine with previous results if they exist
    if previous_results is not None:
        # Append new results to previous results
        loss_mc = np.concatenate([previous_results['loss_mc'], loss_mc])
        loss_data = np.concatenate([previous_results['loss_data'], loss_data])
        loss = np.concatenate([previous_results['loss'], loss])
        steps = np.concatenate([previous_results['steps'], steps])
        resampled_times = previous_results['resampled_times'] + resampled_times
        blocks_used = np.vstack([previous_results['blocks_used'], blocks_used])
    
    # Calculate new cumulative iteration count
    cumulative_iters += n_iter
    iteration_label = f'{cumulative_iters // 1000}k'
    
    # Save all outputs with updated iteration label
    np.savetxt(seed_folder / f'bed_{iteration_label}.txt', beds)
    np.savetxt(seed_folder / f'loss_mc_{iteration_label}.txt', loss_mc)
    np.savetxt(seed_folder / f'loss_data_{iteration_label}.txt', loss_data)
    np.savetxt(seed_folder / f'loss_{iteration_label}.txt', loss)
    np.savetxt(seed_folder / f'steps_{iteration_label}.txt', steps)
    np.savetxt(seed_folder / f'resampled_times_{iteration_label}.txt', resampled_times)
    np.savetxt(seed_folder / f'blocks_used_{iteration_label}.txt', blocks_used)
    
    # Delete old files after successfully saving new ones
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()

    return result

    
def smallScaleChain_mp(n_chains, n_workers, smallScaleChain, initial_beds, ssc_rng_seeds, lsc_rng_seed, n_iters, output_path='./Data/output'):
    '''
    function to run multiple small scale chain using multiprocessing

    Parameters
    ----------
    n_chains (int): the number of chains this multiprocessing choose to run
    n_workers (int): the number of processes the multiprocessing function create. Must be less than the number of CPUs you have
    smallScaleChain (MCMC.chain_sgs): an existing small scale chain that has already been set-up
    initial_beds (list): a list of subglacial topography (each of which are a 2D numpy array) used to initialize each chain
    ssc_rng_seeds (list): a list of int used to initialize the random number generator of each chain
    lsc_rng_seed (int): rng seed for the parent lsc that will be used to find where to save results
    n_iters (int): a list of number of iterations runned for each chain

    Returns
    -------
    result: a list of results from all the chains runned.

    '''
    # Clear the console for the progress bars
    os.system('cls' if os.name == 'nt' else 'clear')

    tic = time.time()

    params = []
    # retrive parameters from the existing chain
    example_chain = smallScaleChain.__dict__

    # modify some of the parameters based on the input ssc_rng_seeds, initial_beds, and n_iters
    for i in range(n_chains):
        chain_param = deepcopy(example_chain)
        chain_param['rng_seed'] = ssc_rng_seeds[i]
        chain_param['initial_bed'] = initial_beds[i]

        run_param = {}
        run_param['n_iter'] = n_iters[i]
        # some display parameters are fixed.
        run_param['only_save_last_bed'] = True
        run_param['info_per_iter'] = 10
        run_param['plot'] = False
        run_param['progress_bar'] = False
        run_param['chain_id'] = i
        run_param['tqdm_position'] = i + 2 # 2 lines for header
        run_param['ssc_seed'] = ssc_rng_seeds[i]
        run_param['lsc_seed'] = lsc_rng_seed
        run_param['output_path'] = str(Path(output_path) / 'LargeScaleChain' / str(lsc_rng_seed)[:6] / 'SmallScaleChain')
        params.append([deepcopy(chain_param), deepcopy(run_param)])

    # Print header and reserve space for progress bars
    print('Running MCMC chains...')
    print('\n' * (n_chains + 1))
    sys.stdout.flush() # force output into the terminal

    # the multiprocessing step
    with mp.Pool(n_workers) as pool:
        result = pool.starmap(msc_run_wrapper, params)

    # Move cursor below chain outputs before printing the timing
    print('\n' * (n_chains + 2))

    toc = time.time()
    print(f'Completed in {toc-tic} seconds')

    return result


def msc_run_wrapper(param_chain, param_run):
    '''
    A function used to initialize chain by input parameters and run the chains

    Parameters
    ----------
    param_chain (dict): Dictionary containing parameters needed to initialize chain
    param_rf (dict): Dictionary containing parameters needed to initialize random field
    param_run (dict): Dictionary containing parameters needed to run chain

    Returns
    -------
    result (tuple): A tuple containing the results of the run

    '''

    # Suppress initialization prints from workers
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    chain = MCMC.init_msc_chain_by_instance(param_chain)

    # Restore stdout after initialization
    sys.stdout.close()
    sys.stdout = old_stdout

    # Setup output path
    output_path = param_run.get(
        'output_path', 
        './Data/LargeScaleChain/'+str(param_run['lsc_seed'])[:6]+'/SmallScaleChain'
        )
    seed = param_run['ssc_seed']
    n_iter = param_run['n_iter']
    seed_folder = Path(output_path) / f'{str(seed)[:6]}'

    # Check for existing bed files (to resume progress)
    existing_beds = list(seed_folder.glob('bed_*.txt'))
    cumulative_iters = 0
    previous_results = None
    files_to_delete = []

    # Prepare to merge/concatenate existing files with new reults
    if existing_beds:
        bed_file = existing_beds[0] # Existing bed file

        # Extract iteration count from filename
        filename = bed_file.stem # Gets 'bed_100k' from 'bed_100k.txt'
        iter_str = filename.split('_')[1].replace('k', '')
        iter_count = int(iter_str)
        cumulative_iters = iter_count * 1000 # Convert back to actual iterations

        # Load the most recent bed file
        most_recent_bed = np.loadtxt(bed_file)

        # Update the chain's intiial bed
        chain.initial_bed = most_recent_bed

        # Load all previous result files
        previous_results = {
            'loss_mc': np.loadtxt(seed_folder / f'loss_mc_{iter_count}k.txt'),
            'loss_data': np.loadtxt(seed_folder / f'loss_data_{iter_count}k.txt'),
            'loss': np.loadtxt(seed_folder / f'loss_{iter_count}k.txt'),
            'steps': np.loadtxt(seed_folder / f'steps_{iter_count}k.txt'),
            'resampled_times': np.loadtxt(seed_folder / f'resampled_times_{iter_count}k.txt'),
            'blocks_used': np.loadtxt(seed_folder / f'blocks_used_{iter_count}k.txt')
        }

        # Mark files for deletion
        files_to_delete = [
            seed_folder / f'bed_{iter_count}k.txt',
            seed_folder / f'loss_mc_{iter_count}k.txt',
            seed_folder / f'loss_data_{iter_count}k.txt',
            seed_folder / f'loss_{iter_count}k.txt',
            seed_folder / f'steps_{iter_count}k.txt',
            seed_folder / f'resampled_times_{iter_count}k.txt',
            seed_folder / f'blocks_used_{iter_count}k.txt'
        ]

    # Store positioning info
    chain.chain_id = param_run.get('chain_id', 'Unknown')
    chain.tqdm_position = param_run.get('tqdm_position', 0)
    chain.seed = param_run.get('ssc_seed', 'Unkown')

    # Run the chain
    result = chain.run(
        n_iter=param_run['n_iter'], 
        only_save_last_bed=param_run['only_save_last_bed'],
        info_per_iter=param_run['info_per_iter'], 
        plot=param_run['plot'], 
        progress_bar=param_run['progress_bar']
        )
    
    # Unpack results
    beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used = result

    # Combine with previous results if they exist
    if previous_results is not None:
        # Append new results to previous results
        loss_mc = np.concatenate([previous_results['loss_mc'], loss_mc])
        loss_data = np.concatenate([previous_results['loss_data'], loss_data])
        loss = np.concatenate([previous_results['loss'], loss])
        steps = np.concatenate([previous_results['steps'], steps])
        resampled_times = previous_results['resampled_times'] + resampled_times
        blocks_used = np.vstack([previous_results['blocks_used'], blocks_used])

    # Calculate new cumulative iteration count
    cumulative_iters += n_iter
    iteration_label = f'{cumulative_iters // 1000}k'

    # Save all outputs with updated iteration label
    np.savetxt(seed_folder / f'bed_{iteration_label}.txt', beds)
    np.savetxt(seed_folder / f'loss_mc_{iteration_label}.txt', loss_mc)
    np.savetxt(seed_folder / f'loss_data_{iteration_label}.txt', loss_data)
    np.savetxt(seed_folder / f'loss_{iteration_label}.txt', loss)
    np.savetxt(seed_folder / f'steps_{iteration_label}.txt', steps)
    np.savetxt(seed_folder / f'resampled_times_{iteration_label}.txt', resampled_times)
    np.savetxt(seed_folder / f'blocks_used_{iteration_label}.txt', blocks_used)

    # Delete old files after successfully saving new ones
    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()

    return result


if __name__ == '__main__':
    # Set file paths here
    #NOTE use r string literals in case backslashes are used
    glacier_data_path = Path(r'DenmanDataGridded.csv')
    seed_file_path = Path(r'../200_seeds.txt')
    output_path = Path(r'./Data/Denman')

    n_iter = 100
    lsc_seed_idx = 0 # Which lsc are we starting from?
    ssc_start_idx = 0 # Min of 0
    ssc_end_idx = 5 # Max of 19
    #NOTE n_chains is calculated by subtracting the starting index from the ending index
    n_workers = 12

    # load compiled bed elevation measurements
    df = pd.read_csv(glacier_data_path)

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

    # notice that this didn't recover bedmachine bed for ocean bathymetry or sub-ice-shelf bathymetry
    bedmachine_bed = bedmap_surf - bedmachine_thickness

    # create conditioning data
    # bed elevation measurement in grounded ice region, and bedmachine bed topography elsewhere
    cond_bed = np.where(
        bedmap_mask == 1, df['bed'].values.reshape(xx.shape), bedmap_bed)
    df['cond_bed'] = cond_bed.flatten()

    # create a mask of conditioning data
    data_mask = ~np.isnan(cond_bed)

    # Read all seeds
    with open(seed_file_path, 'r') as f:
        lines = f.readlines()

    rng_seeds = []
    for line in lines:
        rng_seeds.append(int(line.strip()))

    ssc_rng_seeds = rng_seeds[20*lsc_seed_idx:20*(lsc_seed_idx+1)]
    ssc_rng_seeds = ssc_rng_seeds[ssc_start_idx:ssc_end_idx]
    lsc_rng_seed = rng_seeds[lsc_seed_idx]

    #NOTE Cut off any extra chains that don't have a ssc seed tied to them to find n_chains
    n_chains = len(ssc_rng_seeds) 

    lsc_path = output_path / 'LargeScaleChain' / str(lsc_rng_seed)[:6]

    initial_bed = np.loadtxt(list(lsc_path.glob('bed_*.txt'))[0])
    thickness = bedmap_surf - initial_bed
    # make sure every topography in the grounded ice region is below ice surface
    initial_bed = np.where((thickness <= 0) & (
        bedmap_mask == 1), bedmap_surf-1, initial_bed)

    # sigma here control the smoothness of the trend
    trend = sp.ndimage.gaussian_filter(initial_bed, sigma=10)

    # normalize the conditioning bed data, saved to df['Nbed']
    df['cond_bed_residual'] = df['cond_bed'].values-trend.flatten()
    data = df['cond_bed_residual'].values.reshape(-1, 1)
    # data used to evaluate the distribution. We use all data in the initial bed
    data_for_distribution = (initial_bed - trend).reshape((-1, 1))
    nst_trans = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                    subsample=None, random_state=rng_seed).fit(data_for_distribution)
    # normalize all data in df['cond_bed_residual']
    transformed_data = nst_trans.transform(data)
    df['Nbed_residual'] = transformed_data

    # randomly drop out 50% of coordinates. Decrease this value if you have a lot of data and it takes a long time to run
    df_sampled = df.sample(frac=0.5, random_state=rng_seed)
    df_sampled = df_sampled[df_sampled["cond_bed_residual"].isnull() == False]
    df_sampled = df_sampled[df_sampled["bedmap_mask"] == 1]

    # compute experimental (isotropic) variogram
    coords = df_sampled[['x', 'y']].values
    values = df_sampled['Nbed_residual']

    maxlag = 20000      # maximum range distance
    # num of bins (try decreasing if this is taking too long)
    n_lags = 70

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

    grounded_ice_mask = (bedmap_mask == 1)

    # initialize the small scale chain to be used as an example to initialize other small scale chain
    smallScaleChain = MCMC.chain_sgs(xx, yy, initial_bed, bedmap_surf, velx,
                                     vely, dhdt, smb, cond_bed, data_mask, grounded_ice_mask, resolution)
    # set the update region
    smallScaleChain.set_update_region(True, highvel_mask)

    # get mass flux residuals for bedmachien as a reference
    mc_res_bm = Topography.get_mass_conservation_residual(
        bedmachine_bed, bedmap_surf, velx, vely, dhdt, smb, resolution)

    # in multiprocessing, we choose to only use mass flux residual loss in squared sum (Gaussian distribution)
    smallScaleChain.set_loss_type(sigma_mc=5, massConvInRegion=True)

    # set up the block sizes
    min_block_x = 5
    max_block_x = 20
    min_block_y = 5
    max_block_y = 20
    smallScaleChain.set_block_sizes(
        min_block_x, max_block_x, min_block_y, max_block_y)

    # set up normal score transformation, the trend, the variogram parameters, and the sgs paramters
    smallScaleChain.set_normal_transformation(nst_trans, do_transform=True)

    smallScaleChain.set_trend(trend=trend, detrend_map=True)

    smallScaleChain.set_variogram(
        'Matern', V1_p[0], V1_p[1], 0, isotropic=True, vario_smoothness=V1_p[2])

    smallScaleChain.set_sgs_param(48, 30e3, sgs_rand_dropout_on=False)

    # set up the random generator used in the chain
    # in multiprocessing, the random generator in here will be replaced by rng_seeds later
    smallScaleChain.set_random_generator(rng_seed=rng_seed)

    # fill in a list of initial_beds to be used for each chain
    # the list length should be equal to number of chains
    initial_beds = np.array([initial_bed] * n_chains) # np.repeart(initial_bed, n_chains)

    # number of iterations used to run each chain
    n_iters = [n_iter]*n_chains

    result = smallScaleChain_mp(n_chains, n_workers, smallScaleChain, initial_beds, ssc_rng_seeds, lsc_rng_seed, n_iters)

    # beds, loss_mc, loss_data, loss, steps, resampled_times, blocks_used  = smallScaleChain.run(n_iter=100, info_per_iter=10, only_save_last_bed=False)
