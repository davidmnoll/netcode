
import os
CUDA_VERSION = '12.3'  
CUDA_PATH = os.path.join('C:\\', 'Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v' + CUDA_VERSION)
os.add_dll_directory(CUDA_PATH)
