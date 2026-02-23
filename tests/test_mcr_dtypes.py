import numpy as np
import torch
import timeit 
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def to_tensor(arr, device=device):
    # Using float64 to match NumPy's default precision for accurate comparison
    return torch.tensor(arr, dtype=torch.float32, device=device)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_mass_conservation_residual_torch(bed, surf, velx, vely, dhdt, smb, res):    
    thick_t = surf - bed

    # torch.gradient returns a list; axis 1 = x (columns), axis 0 = y (rows)
    (dx_t,) = torch.gradient(velx * thick_t, spacing=(float(res),), dim=1)
    (dy_t,) = torch.gradient(vely * thick_t, spacing=(float(res),), dim=0)

    res_t = dx_t + dy_t + dhdt - smb
    return res_t

def get_mass_conservation_residual_numpy(bed, surf, velx, vely, dhdt, smb, res):
    thick = surf - bed
    
    dx = np.gradient(velx*thick, res, axis=1)
    dy = np.gradient(vely*thick, res, axis=0)
    
    res_out = dx + dy + dhdt - smb
    return res_out

# Create Mock Data (Synthetic Glacier Dome) 
def generate_mock_data(resolution: int, ny: int, nx: int):
    # Let x be in a Resolution set: x meters per pixel

    # Create coordinate grid, explicitly using float32
    x = np.linspace(-nx//2 * resolution, nx//2 * resolution, nx, dtype=np.float32)
    y = np.linspace(-ny//2 * resolution, ny//2 * resolution, ny, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    # Flat subglacial topography at sea level
    bed = np.zeros((ny, nx), dtype=np.float32)

    # simple parabolic ice dome
    max_height = 2000.0
    surf = max_height - 0.0001 * (xv**2 + yv**2)
    surf = np.maximum(surf, 0).astype(np.float32)

    # Flowing outward from the center
    velx = (0.05 * xv).astype(np.float32)
    vely = (0.05 * yv).astype(np.float32)

    # dh/dt: Assume the glacier is thinning slightly everywhere
    dhdt = np.full((ny, nx), -0.5, dtype=np.float32)

    # SMB: Assume uniform accumulation
    smb = np.full((ny, nx), 0.3, dtype=np.float32)

    return bed, surf, velx, vely, dhdt, smb

def test_np(bed, surf, velx, vely, dhdt, smb, res):
    return get_mass_conservation_residual_numpy(bed, surf, velx, vely, dhdt, smb, res)

def test_torch(bed_t, surf_t, velx_t, vely_t, dhdt_t, smb_t, res):
    result = get_mass_conservation_residual_torch(bed_t, surf_t, velx_t, vely_t, dhdt_t, smb_t, res)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return result

def within_tolerance(a: np.ndarray, b: torch.Tensor, tolerance = 1e-6):
    a_t = torch.as_tensor(a, dtype=torch.float32, device=b.device)
    return torch.allclose(a_t, b, atol=tolerance)

if __name__ == '__main__':
    print(f"Running on device: {device}\n")
    
    # test diff sizes
    #grid_sizes = [50, 100, 150, 200, 250, 500, 1000, 2000, 2500, 3000, 3500, 4000]
    #resolutions = [50, 100, 150,200, 250, 300, 350, 400, 450, 500]
    num_iterations = 10

    """
    {grid_size:
    resolution: 
    np_runtime:
    torch_runtime:
    }
    """
    benchmark_results = []

    print(f"{'Grid Size':<12} | {'Resolution' :<12} | {'NumPy (s)':<12} | {'Torch (s)':<12} | {'Winner'}")
    print("-" * 62)

    for N in range(500, 4000, 500):
        for current_res in range(100, 500, 100):
            # Generate new arrays for the current grid size
            bed, surf, velx, vely, dhdt, smb = generate_mock_data(current_res, N, N)
            
            # Load tensors onto the target device
            bed_t, surf_t, velx_t, vely_t, dhdt_t, smb_t = [
                to_tensor(x) for x in (bed, surf, velx, vely, dhdt, smb)
            ]

            # Warm-up passes 
            np_res      = test_np(bed, surf, velx, vely, dhdt, smb, current_res)
            torch_res   = test_torch(bed_t, surf_t, velx_t, vely_t, dhdt_t, smb_t, current_res)
            # Assert that MCR is actually similar 
            assert within_tolerance(np_res, torch_res, tolerance=1e-10)
            
            # Execute benchmark using lambda to pass arguments cleanly
            np_time = timeit.timeit(
                lambda: test_np(bed, surf, velx, vely, dhdt, smb, current_res), 
                number=num_iterations
            ) / num_iterations
            
            torch_time = timeit.timeit(
                lambda: test_torch(bed_t, surf_t, velx_t, vely_t, dhdt_t, smb_t, current_res), 
                number=num_iterations
            ) / num_iterations

            # storing into key value. format
            result_entry = {
                "grid_size": N, 
                "resolution": current_res,
                "np_runtime": np_time,
                "torch_runtime": torch_time
            }
            benchmark_results.append(result_entry)

            
            # Calculate speedup
            if torch_time < np_time:
                speedup = np_time / torch_time
                winner = f"Torch ({speedup:.2f}x)"
            else:
                speedup = torch_time / np_time
                winner = f"NumPy ({speedup:.2f}x)"

            # Output formatted results
            grid_label = f"{N}x{N}"
            print(f"{grid_label:<12} | {current_res: <6} | {np_time:<12.6f} | {torch_time:<12.6f} | {winner}")
'''
    output_filename = "tests/benchmark_results.json"
    with open(output_filename, "w") as f:
        json.dump(benchmark_results, f, indent=4)
        
    print(f"saved to '{output_filename}'")
'''