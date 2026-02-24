# Adding GPU support for gstatsMCMC
- This implementation adds GPU support for MCMC.py and Topography.py using PyTorch, allowing devices that has CUDA GPUs or Mac Sillicon Metal Performance Shaders (MPS) to perform a collection of highly optized compute. 
- Previously, gstatsMCMC only performed numpy calculations, which were all on CPUs. Calculations like spectral synthesis, gradient and mass conservation residual were highly repetitive for n_iterations. 
- Adding GPU support aims to handle calculations simultaneously, in parralel, in contrast to cPU's sequential processing. 
- Instead of numpy arrays, Pytorch uses tensors: a similar data structure to ndarray but allows access to GPU cores and VRAM directly.

# Changes

1. MCMC_gpu.py
- Introduce ` class chain_crf_gpu` subclass that inherits `chain_crf` class from MCMC.py . All original methods are encapsulated in chain_crf_gpu as well
- Adds Tensor + GPU support on top of existing CPU-based numpy class.
- Call `chain_crf_gpu` when using a GPU

2. GPU/Pytorch Integration
- All numerical arrays are converted from NumPy to PyTorch tensors via a new `_to_tensor()` method, targeting CUDA, MPS (Apple Silicon), or CPU automatically via _set_torch_device()
- `_set_torch_device()` auto detects your hardware's device
- For example, if you have a Macbook w/ Mac Sillicon architecture (e.g. M2, M3,..), then `mps` is prioritized over `cpu`
- Cache arrays (loss_cache, bed_cache, resampled_times, etc.) are now pre-allocated as GPU tensors
and only transferred to NumPy at the very end of the run, minimizing CPU-GPU synchronization.
- A new `_loss_tensor()` method computes the loss entirely on-device without leaving the GPU.
- We avoid CPU-GPU synchronization because it requires memory allocation overhead that takes time -- milliseconds * n_iterations * `shape`

3. Spectral Field Generation
- A new `spectral_synthesis_field_torch()` function is added that performs FFT-based random field generation using PyTorch on GPU, as a replacement for the original NumPy-based spectral_synthesis_field().
- The NumPy RNG is deliberately kept for scalar parameter sampling to preserve reproducibility


# Testing

Hardware: 
- Mac M2 | supports MPS
- 8GB Memory
 
1. Mass Conservation Residual Calculation `tests\test_mcr_dtypes.py`
- Computing MCR on Pytorch Tensors on MPS outperforms Numpy Ndarrays exponentially once the NxN grid sizes grows (N = [50, 4000])
- Tested resolution at a range of [50, 500], demonstrating the same results.
![MCR_viz](images/screenshot.png)

2. LargeScale Chain
- Using Bindshadler & MacAyeal data at roughly shape (2000, 2000) and 500 resolution
- 50000 iterations each
![mcmc_benmark](images/screenshot.png)
