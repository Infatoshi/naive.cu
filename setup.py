from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Get the directory of this setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='naive-cu-extensions',
    ext_modules=[
        # Training extension
        CUDAExtension(
            name='custom_training_extension',
            sources=[
                os.path.join(setup_dir, 'src', 'training', 'binding.cpp'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'matmul.cu'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'elementwise.cu'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'activation.cu'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'softmax.cu'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'layernorm.cu'),
                os.path.join(setup_dir, 'src', 'training', 'kernels', 'embedding.cu'),
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            }
        ),
        # Inference extension
        CUDAExtension(
            name='custom_inference_extension',
            sources=[
                os.path.join(setup_dir, 'src', 'inference', 'binding.cpp'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'matmul_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'gemv_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'elementwise_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'activation_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'softmax_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'layernorm_fwd.cu'),
                os.path.join(setup_dir, 'src', 'inference', 'kernels', 'topk_fwd.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--expt-extended-lambda']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
