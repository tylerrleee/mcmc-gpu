from gstatsMCMC.MCMC import *
from gstatsMCMC import Topography
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def spectral_synthesis_field_torch(RF, shape, res=1.0, device=None, dtype=torch.float32):
    """
    Generate a 2D random field using FFT-based spectral synthesis — PyTorch version

    Functionally identical to spectral_synthesis_field()
    -  frequency-domain and field operations run on `device`. The numpy RNG (RF.rng) 
        is kept for scalar parameter sampling so chain reproducibility is preserved

    Args:
        RF        : RandField instance with model_name, scale_min/max, nugget_max,
                    range_min/max_x/y, isotropic, smoothness, rng attributes.
        shape     : (ny, nx) — output grid shape.
        res       : Grid spacing in metres (default 1.0).
        device    : torch.device to run on.  Falls back to RF.device if None.
        dtype     : Floating-point dtype.  Must be float32 on MPS.

    Returns:
        torch.Tensor: shape (ny, nx) on `device`, same semantics as the NumPy version.
    """

    ny, nx   = shape
    rng      = RF.rng                                      # numpy Generator — kept for reproducibility
    dev      = device 

    scale = float(rng.uniform(RF.scale_min, RF.scale_max)) / 3.0
    nug   = float(rng.uniform(0.0, RF.nugget_max))

    if not RF.isotropic:
        range_x = float(rng.uniform(RF.range_min_x, RF.range_max_x))
        range_y = float(rng.uniform(RF.range_min_y, RF.range_max_y))
    else:
        range_x = range_y = float(rng.uniform(RF.range_min_x, RF.range_max_x))

    model_name = RF.model_name
    if model_name == "Gaussian":
        len_x = range_x / math.sqrt(3)
        len_y = range_y / math.sqrt(3)
    elif model_name == "Exponential":
        len_x = range_x / 3.0
        len_y = range_y / 3.0
    else:  # Matern
        len_x = range_x / 2.0
        len_y = range_y / 2.0

    # Frequency grids - On GPU
    kx = torch.fft.fftfreq(nx, d=1.0, device=dev, dtype=dtype) * (2.0 * math.pi / res)
    ky = torch.fft.fftfreq(ny, d=1.0, device=dev, dtype=dtype) * (2.0 * math.pi / res)

    # indexing="ij" -> kyv[i,j] = ky[i], kxv[i,j] = kx[j]  (matches np.meshgrid indexing="ij")
    kyv, kxv = torch.meshgrid(ky, kx, indexing="ij")
    k = torch.sqrt(kxv ** 2 + kyv ** 2) + 1e-10    

    # Spectral power density
    a = math.sqrt(len_x * len_y)
    if model_name == "Gaussian":
        S = torch.exp(-0.5 * (a * k) ** 2)

    elif model_name == "Exponential":
        S = 1.0 / (1.0 + (a * k) ** 2) ** 1.5

    else:  # Matern
        nu      = float(RF.smoothness) if RF.smoothness else 1.0
        # Scalar constants computed in Python 
        constant = (
            (4.0 * math.pi * math.gamma(nu + 1.0) * (2.0 * nu) ** nu)
            / (math.gamma(nu) * a ** (2.0 * nu))
        )
        keppa = 2.0 * nu / (a ** 2)
        S = constant * ((keppa + 4.0 * math.pi * k ** 2) ** (-nu - 1.0))

    # Complex white noise
    noise_real_np = rng.normal(size=(ny, nx)).astype(np.float32)
    noise_imag_np = rng.normal(size=(ny, nx)).astype(np.float32)

    noise_real = torch.from_numpy(noise_real_np).to(device=dev, dtype=dtype)
    noise_imag = torch.from_numpy(noise_imag_np).to(device=dev, dtype=dtype)


    noise      = torch.complex(noise_real, noise_imag)
    S_complex  = S.to(dtype=torch.complex64 if dtype == torch.float32 else torch.complex128)

    #noise = np.random.normal(size=(ny, nx)) + 1j * np.random.normal(size=(ny, nx))
    freq_field = noise * torch.sqrt(S_complex) 

    # Inverse FFT
    field = torch.fft.ifft2(freq_field).real  # (ny, nx) float32, on device

    # Standardise: zero mean, unit variance
    field = (field - field.mean()) / (field.std() + 1e-12)

    # Apply scaling and nugget noise
    field = field * scale 

    nugget_np = rng.normal(0.0, math.sqrt(nug), size=(ny, nx)).astype(np.float32)
    nugget_t  = torch.from_numpy(nugget_np).to(device=dev, dtype=dtype)
    field     = field + nugget_t

    return field  # (ny, nx) tensor on device

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

class chain_crf_gpu(chain_crf):
    def __init_func__(self):
        return
    
    def _to_tensor(self, device = None):
        """
        Converts all Numpy array parameters initialized in the parent 'chain' class 
        and the child 'chain_crf' class to PyTorch tensors (float32)

        """        
        def convert(arr, dataType = torch.float32):
            """ 
            Helper ; convert ndarray to tensor w/ checks
            
            Default dtype: torch.float32 
            - Pytorch support for Apple Mac Silicon *MPS* only works w/ float32
            """
            if arr is not None and isinstance(arr, np.ndarray):
                return torch.tensor(arr, dtype = dataType, device = self.device)
            return arr

        # Convert parent class (chain) base arrays
        # Reference __init__ for parameters details

        self.xx             = convert(self.xx)
        self.yy             = convert(self.yy)
        self.initial_bed    = convert(self.initial_bed)
        self.surf           = convert(self.surf)
        self.velx           = convert(self.velx)
        self.vely           = convert(self.vely)
        self.dhdt           = convert(self.dhdt)
        self.smb            = convert(self.smb)
        self.cond_bed       = convert(self.cond_bed)
        self.data_mask      = convert(self.data_mask)
        self.grounded_ice_mask = convert(self.grounded_ice_mask)
        
        # self.resolution is intentionally left as a standard float
        # torch.gradient() expects a float/tuple of floats for the spacing argument
        if isinstance(self.resolution, (int, float)):
            self.resolution = float(self.resolution)

        # Convert dynamically assigned parent class (chain) arrays
        if hasattr(self, 'region_mask'):
            self.region_mask = convert(self.region_mask)
        if hasattr(self, 'mc_region_mask'):
            self.mc_region_mask = convert(self.mc_region_mask)
        if hasattr(self, 'sample_loc'):
            self.sample_loc = convert(self.sample_loc)
            
        # Convert child class (chain_crf) specific arrays
        if hasattr(self, 'crf_data_weight'):
            self.crf_data_weight = convert(self.crf_data_weight)
            
        if hasattr(self, 'initial_bed'):
            self.initial_bed = convert(self.initial_bed)
        print(f"\n data converted to tensors on {self.device}")

        # Tensor Loss helper
    def _loss_tensor(self, mc_res, mc_region_bool, sigma_denom):
        """
            Compute scalar loss values staying on GPU.

            Returns three plain Python floats so they can be stored in the
            cache tensors and used in the Markov Chain acceptance test without
            any device synchronisation penalty.
        """
        region_vals = mc_res[mc_region_bool]
        loss_mc = torch.nansum(region_vals ** 2) / sigma_denom

        loss_data = torch.tensor(0.0, dtype=mc_res.dtype, device=mc_res.device)
        total     = loss_mc + loss_data
        return total, loss_mc, loss_data

    def _set_torch_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f'Device to {self.device}')
        #self.device = torch.device("cpu")

    def run(self, n_iter, RF, only_save_last_bed=False, info_per_iter = 1000, plot=True, progress_bar=True):
        """Runs the MCMC sampling chain to generate topography realizations using PyTorch on GPU

        Args:
            n_iter (int): The total number of MCMC iterations to perform.
            RF (RandField): An initialized `RandField` object used to generate the topography perturbations.
            rng_seed (int, optional): A seed for the random number generator to ensure reproducibility. Defaults to None.
            only_save_last_bed (bool): If True, only the final topography is returned. If False, the topography from every iteration is saved, which requires more memory.
            info_per_iter (int): The iteration interval for printing progress updates, such as loss and acceptance rate, to the console.

        Returns:
            bed_cache (Tensor): A 4D array of saved topographies if `only_save_last_bed` is False, or a 2D array of the final topography if True.
            loss_mc_cache (Tensor): A 1D array of the mass conservation loss at each iteration. If no mass conservation loss is set, will return zeros.
            loss_data_cache (np.ndarray): A 1D array of the data misfit loss at each iteration. If no data misfit loss is set, will return zeros.
            loss_cache (np.ndarray): A 1D array of the total loss (mass conservation loss + data misfit loss) at each iteration.
            step_cache (np.ndarray): A 1D boolean array indicating whether the proposal was accepted at each iteration.
            resampled_times (np.ndarray): A 2D array counting how many times each grid cell was part of an accepted proposal.
            blocks_cache (np.ndarray): A 2D array logging the location and size `[row, col, height, width]` of the proposed update block at each iteration.
        """
        # synchronize the random generator with RF object
        rng = self.rng
        
        if not isinstance(RF, RandField):
            raise TypeError('The argument "RF" has to be an object of the class RandField')
        
        self._set_torch_device()

        dev   = self.device          # torch.device – cuda / mps / cpu
        dtype = torch.float32        # MPS only supports float32

        # Convert all parameters to tensors if not already
        self._to_tensor(device=dev)
        H, W = self.xx.shape

        # cache tensors
        loss_mc_cache_t   = torch.zeros(n_iter, dtype=dtype, device=dev)
        loss_data_cache_t = torch.zeros(n_iter, dtype=dtype, device=dev)
        loss_cache_t      = torch.zeros(n_iter, dtype=dtype, device=dev)
        step_cache_t      = torch.zeros(n_iter, dtype=torch.bool, device=dev)
        resampled_times_t = torch.zeros((H, W), dtype=dtype, device=dev)
        blocks_cache_t    = torch.full((n_iter, 4), float('nan'), dtype=dtype, device=dev)

        # Pre-computes - cache once
        mc_region_bool      = (self.mc_region_mask == 1)    
        region_bool         = self.region_mask.bool()        
        grounded_bool       = self.grounded_ice_mask.bool()  
        sigma_denom         = 2.0 * self.sigma_mc ** 2      
        block_record        = torch.empty(4, dtype=dtype, device=dev)


        if not only_save_last_bed:
            bed_cache_t = torch.zeros((n_iter, H, W), dtype=dtype, device=dev)
        
        # if the user request to return bed elevation in some sampling locations
        # Sample point trackng
        track_samples = self.sample_loc is not None
        if track_samples:
            sample_loc_t  = self.sample_loc                       # (K, 2) float tensor
            K             = sample_loc_t.shape[0]
            sample_vals_t = torch.zeros((K, n_iter), dtype=dtype, device=dev)

            # Convert (x, y) sample locations → (i, j) grid indices (on GPU)
            xx_flat = self.xx.flatten()          # (H*W,)
            yy_flat = self.yy.flatten()
            sample_loc_ij_t = torch.zeros((K, 2), dtype=torch.long, device=dev)
            for k in range(K):
                match = (xx_flat == sample_loc_t[k, 0]) & (yy_flat == sample_loc_t[k, 1])
                flat_idx = match.nonzero(as_tuple=False)[0, 0]
                sample_loc_ij_t[k, 0] = flat_idx // W
                sample_loc_ij_t[k, 1] = flat_idx %  W

            sample_vals_t[:, 0] = self.initial_bed[
                sample_loc_ij_t[:, 0], sample_loc_ij_t[:, 1]
            ]

        # Working Tensors
        bed_c = self.initial_bed.clone() # (H, W) tensor on dev
        resolution: float = self.resolution    # Float
        

        # Initialize loss
        mc_res = Topography.get_mass_conservation_residual_tensor( 
            bed_c, self.surf, self.velx, self.vely, self.dhdt, self.smb, self.resolution
            ) # -> (H, W) tensor
        
        loss_prev, loss_prev_mc, loss_prev_data = self._loss_tensor(mc_res, mc_region_bool, sigma_denom)

        #loss_prev_float = float(loss_prev)
        loss_cache_t[0]      = loss_prev
        loss_mc_cache_t[0]   = loss_prev_mc
        loss_data_cache_t[0] = loss_prev_data
        step_cache_t[0]      = False

        if not only_save_last_bed:
            bed_cache_t[0] = bed_c
        
        # Live plot -- using matplotlib on CPU
        if plot:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
            (line_loss,) = ax_loss.plot([], [], color='tab:blue',  label='Loss')
            (line_acc,)  = ax_acc.plot([], [],  color='tab:green', label='Acceptance Rate')
            ax_loss.set_xlabel("Iteration"); ax_loss.set_ylabel("Loss")
            ax_loss.set_title("MCMC Loss");  ax_loss.legend()
            ax_acc.set_xlabel("Iteration");  ax_acc.set_ylabel("Acceptance Rate (%)")
            ax_acc.set_ylim(0, 100);         ax_acc.set_title("MCMC Acceptance Rate")
            ax_acc.legend()
            display_handle = display.display(fig, display_id=True)
            plt.tight_layout()


        # PRogress setup
        accepted_count = 0
        acceptance_rates = []

        chain_id      = getattr(self, 'chain_id', 0)
        seed          = getattr(self, 'seed', 'Unknown')
        tqdm_position = getattr(self, 'tqdm_position', 0)

        if progress_bar:
            iterator = tqdm(range(1, n_iter),
                            desc=f'Chain {chain_id} | Seed {seed}',
                            position=tqdm_position,
                            leave=True)
        else:
            iterator   = range(1, n_iter)
            output_line = tqdm_position + 2

        iter_start_time = time.time()

        # Ensure that this is ndarray to ensure no sync from GPU to CPU is happening
        region_mask_np = self.region_mask.cpu().numpy()  

        for i in iterator:

            ## Potentially conerting to tensor native rfblock   
            f_np     = RF.get_rfblock()                          # (bx, by) ndarray
            f_t      = torch.tensor(f_np, dtype=dtype, device=dev)
            block_size = f_t.shape                               # (bx, by)
            
            # Sample block location
            if self.update_in_region:
                while True:
                    indexx = int(rng.integers(0, H))
                    indexy = int(rng.integers(0, W))
                    if region_mask_np[indexx, indexy] == 1: # Must be a numpy based mask, not GPU
                        break
            else:
                indexx = int(rng.integers(0, H))
                indexy = int(rng.integers(0, W))
                
            #record block

            block_record[0] = indexx
            block_record[1] = indexy
            block_record[2] = block_size[0]
            block_record[3] = block_size[1]
            blocks_cache_t[i] = block_record

            #find the index of the block side, make sure the block is within the edge of the map
            bxmin = max(0,  indexx - block_size[0] // 2)
            bxmax = min(H,  indexx + block_size[0] // 2)
            bymin = max(0,  indexy - block_size[1] // 2)
            bymax = min(W,  indexy + block_size[1] // 2)

            # Corresponding slice into the random-field block itself
            mxmin = max(block_size[0] - bxmax, 0)
            mxmax = min(H - bxmin, block_size[0])
            mymin = max(block_size[1] - bymax, 0)
            mymax = min(W - bymin, block_size[1])
            
            # Find block perturbation -- GPU operations
            if self.block_type == 'CRF_weight':
                perturb = (f_t[mxmin:mxmax, mymin:mymax]
                           * self.crf_data_weight[bxmin:bxmax, bymin:bymax])
            else:
                perturb = f_t[mxmin:mxmax, mymin:mymax]

            bed_next = bed_c.clone()
            bed_next[bxmin:bxmax,bymin:bymax]=bed_next[bxmin:bxmax,bymin:bymax] + perturb

            # Restrict update to region / grounded ice mask
            if self.update_in_region:
                bed_next = torch.where(self.region_mask.bool(), bed_next, bed_c)
            else:
                bed_next = torch.where(self.grounded_ice_mask.bool(), bed_next, bed_c)



            # Define Padded Block to solve gradient & mass conservation residual (MSR)
            pad   = 1
            c_xmin = max(0, bxmin - pad)
            c_xmax = min(H, bxmax + pad)
            c_ymin = max(0, bymin - pad)
            c_ymax = min(W, bymax + pad)

            local_mc_res = Topography.get_mass_conservation_residual_tensor(
                    bed_next[c_xmin:c_xmax, c_ymin:c_ymax],
                    self.surf [c_xmin:c_xmax, c_ymin:c_ymax], 
                    self.velx [c_xmin:c_xmax, c_ymin:c_ymax],
                    self.vely [c_xmin:c_xmax, c_ymin:c_ymax],
                    self.dhdt [c_xmin:c_xmax, c_ymin:c_ymax],
                    self.smb  [c_xmin:c_xmax, c_ymin:c_ymax],
                    self.resolution
            )

            # Patch only the block region of the global residual tensor
            mc_res_candidate = mc_res.clone()
            valid_x_start = bxmin - c_xmin
            valid_x_end = valid_x_start + (bxmax - bxmin)
            valid_y_start = bymin - c_ymin
            valid_y_end = valid_y_start + (bymax - bymin)
            mc_res_candidate[bxmin:bxmax, bymin:bymax] = local_mc_res[valid_x_start:valid_x_end, valid_y_start:valid_y_end]


            # Compute loss
            loss_next_val, loss_next_mc, loss_next_data = self._loss_tensor(mc_res_candidate, mc_region_bool, sigma_denom)
            loss_next = float(loss_next_val)
            block_thickness   = self.surf[bxmin:bxmax, bymin:bymax] - bed_next[bxmin:bxmax, bymin:bymax]
            
            # Thickness guard -- penalize where bed elevation > surface elevation
            if self.update_in_region:
                block_region_mask = self.region_mask[bxmin:bxmax, bymin:bymax]
            else:
                block_region_mask = self.grounded_ice_mask[bxmin:bxmax, bymin:bymax]

            if torch.any((block_thickness <= 0) & (block_region_mask == 1)):
                loss_next = float('inf')

            # Acceptance steps
            if loss_prev > loss_next:
                acceptance_rate = 1.0
            else:
                #delta = loss_prev - loss_next        
                acceptance_rate = min(1.0, math.exp(loss_prev - loss_next))

            u = rng.random()                          # Python float from numpy RNG
            if u <= acceptance_rate:
                # Accept
                bed_c[bxmin:bxmax, bymin:bymax]  = bed_next[bxmin:bxmax, bymin:bymax]
                mc_res = mc_res_candidate

                loss_prev      = loss_next
                loss_prev_mc   = loss_next_mc
                loss_prev_data = loss_next_data

                loss_cache_t[i]      = loss_next
                loss_mc_cache_t[i]   = loss_next_mc
                loss_data_cache_t[i] = loss_next_data
                step_cache_t[i]      = True

                if self.update_in_region:
                    resampled_times_t[bxmin:bxmax,bymin:bymax] += self.region_mask[bxmin:bxmax,bymin:bymax]
                else:
                    resampled_times_t[bxmin:bxmax,bymin:bymax] += self.grounded_ice_mask[bxmin:bxmax,bymin:bymax]

                accepted_count += 1
            else:
                # Reject -- carry forward previous loss values
                loss_cache_t[i]      = loss_prev
                loss_mc_cache_t[i]   = loss_prev_mc
                loss_data_cache_t[i] = loss_prev_data
                step_cache_t[i]      = False

            
            if not only_save_last_bed:
                bed_cache_t[i] = bed_c

            if track_samples:
                sample_vals_t[:, i] = bed_c[
                    sample_loc_ij_t[:, 0], sample_loc_ij_t[:, 1]
                ]

            # Cmd progress bar
            if progress_bar:
                if i % 1000 == 0:
                    iterator.set_postfix({
                        'chain_id'        : chain_id,
                        'seed'            : seed,
                        'mc loss'         : f'{loss_mc_cache_t[i].item():.3e}',
                        'data loss'       : f'{loss_data_cache_t[i].item():.3e}',
                        'loss'            : f'{loss_cache_t[i].item():.3e}',
                        'acceptance rate' : f'{accepted_count / (i + 1):.6f}',
                    })
            else:
                if i % info_per_iter == 0 or i == 1 or i == n_iter - 1:
                    move_cursor_to_line(output_line)
                    clear_line()
                    progress      = i / (n_iter - 1) * 100
                    elapsed       = time.time() - iter_start_time
                    iter_per_sec  = i / elapsed if elapsed > 0 else 0
                    print(
                        f'Chain {chain_id}: {progress:.1f}% | i: {i} | '
                        f'mc loss: {loss_mc_cache_t[i].item():.3e} | '
                        f'loss: {loss_cache_t[i].item():.3e} | '
                        f'acc: {accepted_count / (i + 1):.4f} | '
                        f'it/s: {iter_per_sec:.2f} | '
                        f'seed: {str(seed)[:6]}',
                        end=''
                    )
                    sys.stdout.flush()

            # Calculate acceptance rate for plot
            total_acceptance = (accepted_count / (i + 1)) * 100
            acceptance_rates.append(total_acceptance)

            if plot:
                update_interval = 100 if i < 5000 else info_per_iter
                if i % update_interval == 0:
                    # Minimal CPU transfer only for plotting – rare
                    line_loss.set_data(
                        range(i + 1),
                        loss_cache_t[:i + 1].cpu().numpy()
                    )
                    ax_loss.relim(); ax_loss.autoscale_view()
                    line_acc.set_data(range(len(acceptance_rates)), acceptance_rates)
                    ax_acc.set_ylim(0, 100); ax_acc.relim(); ax_acc.autoscale_view()
                    display_handle.update(fig)
        
        # Transfer to Numpy Arrays -- single operation at the end
        loss_mc_cache_np   = loss_mc_cache_t.cpu().numpy()
        loss_data_cache_np = loss_data_cache_t.cpu().numpy()
        loss_cache_np      = loss_cache_t.cpu().numpy()
        step_cache_np      = step_cache_t.cpu().numpy()
        resampled_times_np = resampled_times_t.cpu().numpy()
        blocks_cache_np    = blocks_cache_t.cpu().numpy()


        if not only_save_last_bed:
            bed_cache_np = bed_cache_t.cpu().numpy()
            if track_samples:
                sample_values_np = sample_vals_t.cpu().numpy()

                return (bed_cache_np, loss_mc_cache_np, loss_data_cache_np,
                        loss_cache_np, step_cache_np, resampled_times_np,
                        blocks_cache_np, sample_values_np)
            else:
                return (bed_cache_np, loss_mc_cache_np, loss_data_cache_np,
                        loss_cache_np, step_cache_np, resampled_times_np,
                        blocks_cache_np)
        else:
            last_bed_np = bed_c.cpu().numpy()
            if track_samples:
                sample_values_np = sample_vals_t.cpu().numpy()
                return (last_bed_np, loss_mc_cache_np, loss_data_cache_np,
                        loss_cache_np, step_cache_np, resampled_times_np,
                        blocks_cache_np, sample_values_np)
            else:
                return (last_bed_np, loss_mc_cache_np, loss_data_cache_np,
                        loss_cache_np, step_cache_np, resampled_times_np,
                        blocks_cache_np)
