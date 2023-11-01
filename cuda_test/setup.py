from setuptools import setup, Extension
import os
import subprocess

# Adjust this to the CUDA Toolkit version you have installed
CUDA_VERSION = '12.3'  
CUDA_PATH = os.path.join('C:\\', 'Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v' + CUDA_VERSION)



# Compile the CUDA code to produce an object file
subprocess.check_call(['nvcc', '-c', 'hello_kernel.cu', '-o', 'hello_kernel.obj', '-Xcompiler', '/MD'], shell=True)


include_dirs = [os.path.join(CUDA_PATH, 'include')]
library_dirs = [os.path.join(CUDA_PATH, 'lib', 'x64')]
libraries = ['cudart']
extra_objects = ['hello_kernel.obj']

module = Extension('hello',
    sources = ['hello.c'],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=["/MD"],
    extra_link_args=[os.path.join(CUDA_PATH, 'lib', 'x64', 'cudart.lib'), 'hello_kernel.obj'],
  )

setup(name='HelloModule',
      version='1.0',
      description='Simple hello module',
      ext_modules=[module])
