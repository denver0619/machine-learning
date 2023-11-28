import torch
import time
import torch_directml

dml = torch_directml.device()

def bench():
    print(f'running on {torch_directml.device_name(dml.index)}:')
    a = torch.randn(size=(2000,2000)).to(dml)
    b = torch.randn_like(a).to(dml)
    
    start = time.time()
    c = a+b
    end = time.time()
    
    # print(f'available devices: {torch.dml.device_count()}')
    # print(f'current device: { torch.dml.current_device()}')
    print(f'--took {end-start:.2f} seconds')

bench()