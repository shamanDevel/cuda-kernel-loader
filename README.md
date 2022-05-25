# CUDA Kernel Loader

Tired of large switch statements to compile from runtime arguments to static arguments for template instancing?
Tired of long compilation times?
Tired of being limited to the options included during compilation?

Fret no more, with this library, you can configure, assemble and compile CUDA kernels during runtime.



### Example:
Assume you have a kernel with a lot of compile-time switches via template in a file `my_kernel.cu`

```c++
//my_kernel.cu
template<bool Switch1, int Variable2, MyEnum Enum3>
__global__ MyKernel(int arg1, float* arg2) {...}
```

Traditionally, you would need to instantiate all possible combinations during compile time and then use nested switch-statements to translate the arguments.

```c++
//traditional_launch.cpp
void launchKernel(bool Switch1, int Variable2, MyEnum Enum3, int arg1, float* arg2)
{
    unsigned gridDim = 1;
    unsigned blockDim = 1;
    switch (Enum3)
    {
    case MyEnum::Value1:
        switch (Variable2):
        {
        case 1:
            if (Swtich1)
                //finally the kernel
                MyKernel<true, 1, MyEnum::Value1><<<gridDim,blockDim>>>(arg1, arg2);
            else
                MyKernel<false, 1, MyEnum::Value1><<<gridDim,blockDim>>>(arg1, arg2);
            break;
        case 2:
            //more lines for all possible values of 'Variable2'
        default:
            throw exception("This value is outside of what was specified during compilation :( ");
        }
        break;
    case MyEnum::Value2:
        //more lines for all possible values of 'Enum3'
        //need to duplicate all of above, but change MyEnum.
    }
}
```

This is a lot of code duplication and code bloat. It also restricts the possible values of something like `Variable2` to values specified during compilation.
With this library, you can simplify this to:

```c++
//runtime_compilation.cpp
#include <ckl/kernel_loader.h>
void launchKernel(bool Switch1, int Variable2, MyEnum Enum3, int arg1, float* arg2)
{
    ckl::KernelLoader kl = ... //see below
    
    //generate kernel name, e.g. MyKernel<true, 1, MyEnum::Value1>
    std::string kernelName = "MyKernel<" + to_string(Switch1) + ", " + to_string(Variable2) + ", " + to_string(Enum3) + ">";
    //compile the kernel
    std::string kernelCode = ckl::KernelLoader::MainFile("my_kernel.cu");
    auto fun = kl->getKernel(kernelName, kernelCode);
    
    //launch kernel
    unsigned gridDim = 1;
    unsigned blockDim = 1;
    fun->call(gridDim, blockDim, 0, nullptr, 
       /* now follows the arguments to the kernel*/
       arg1, arg2);
}
```

## Requirements

 - CMake 3.4 or higher

 - CUDA SDK 10.0 or higher
   Note: the SDK must be installed, not just the runtime. Otherwise, NVRTC can't be found

 - A C++17 compatible compiler






Note: the string conversions have to be implemented by the user.
For enums, see e.g. enum_name by https://github.com/Neargye/magic_enum



## Installation / Project structure

The project consists of a C++/CUDA part that has to be compiled first:

- `renderer`: the renderer static library, see below for [noteworthy files](#noteworthy-files). Files ending in `.cuh` and `.cu` are CUDA kernel files.
- `bindings`: entry point to the Python bindings, after compilation leads to a python extension module `pyrenderer`, placed in `bin`
- `gui`: the interactive GUI to design the config files, explore the reference datasets and the trained networks. Requires OpenGL

For compilation, we recommend CMake. For running on a headless server, specify `-DRENDERER_BUILD_OPENGL_SUPPORT=Off -DRENDERER_BUILD_GUI=Off`.
Alternatively, `compile-library-server.sh` is provided for compilation with the built-in extension compiler of PyTorch. We use this for compilation on our headless GPU server, as it simplifies potential wrong dependencies to different CUDA, Python or PyTorch versions with different virtualenvs or conda environments.

After compiling the C++ library, the network training and evaluation is performed in Python. The python files are all found in `applications`:

- `applications/volumes` the volumes used in the ablation studies
- `applications/config-files` the config files
- `applications/common`: common utilities, especially `utils.py` for loading the `pyrenderer` library and other helpers
- `applications/losses`: the loss functions, including SSIM and LPIPS
- `applications/volnet`: the main network code for training in inference, see below.

To generate a stubfile with all the Python functions exposed by the renderer, launch `python applications/common/create_sub.py`

## Noteworthy Files

Here we list and explain noteworthy files that contain important aspects of the presented method

On the side of the C++/CUDA library in `renderer/` are the following files important. Note that for the various modules, multiple implementations exists, e.g. for the TF. Therefore, the CUDA-kernels are assembled on-demand using NVRTC runtime compilation.

- 
  Image evaluators (`iimage_evaluator.h`), the entry point to the renderer. Only one implementation:
  
  - `image_evaluator_simple.h`,  `renderer_image_evaluator_simple.cuh`: Contains the loop over the pixels and generates the rays -- possibly multisampled for Monte Carlo -- from the camera
  
- Ray evaluators (`iray_evaluation.h`), called per ray and returns the colors. They call the volume implementation to fetch the density

  - `ray_evaluation_stepping.h`, `renderer_ray_evaluation_stepping_iso.cuh`, `renderer_ray_evaluation_stepping_dvr.cuh`: constant stepping for isosurfaces and DVR.
  - `ray_evaluation_monte_carlo.h` Monte Carlo path tracing with multiple bounces, delta tracking and various phase functions

- Volume interpolations (`volume_interpolation.h`). On the CUDA-side, implementations provide a functor that evaluates a position and returns the density or color at that point

  - Grid interpolation (`volume_interpolation_grid.h`), trilinear interpolation into a voxel grid stored in `volume.h`.
  - Scene Reconstruction Networks (`volume_interpolation_network.h`). The SRNs as presented in the paper. See the header for the binary format of the `.volnet` file.
    The proposed **tensor core implementation** (Sec. 4.1) can be found in `renderer_volume_tensorcores.cuh`

On the python side in `applications/volnet/`, the following files are important:

- `train_volnet`: the entry point for training
- `inference.py`: the entry point for inference, used in the scripts for evaluation. Also converts trained models into the binary format for the GUI
- `network.py`: The SRN network specification
- `input_data.py`: The loader of the input grids, possibly time-dependent
- `training_data.py`: world- and screen-space data loaders, contains routines for importance sampling / adaptive resampling. The rejection sampling is implemented in CUDA for performance and called from here
- `raytracing.py`: Differentiable raytracing in PyTorch, including the memory optimization from Weiss&Westermann 2021, [DiffDVR](https://github.com/shamanDevel/DiffDVR)

## How to train

The training is launched via `applications/volnet/train_volnet.py`. Have a look at `python train_volnet.py --help` for the available command line parameters.

A typical invocation looks like this (this is how fV-SRN with Ejecta from Fig. 1 was trained)

    python train_volnet.py
       config-files/ejecta70-v6-dvr.json
       --train:mode world  # instead of 'screen', Sec. 5.4
       --train:samples 256**3
       --train:sampler_importance 0.01   # importance sampling based on the density, optional, see Section 5.3
       --train:batchsize 64*64*128
       --rebuild_dataset 51   # adaptive resampling after 51 epochs, see Section 5.3
       --val:copy_and_split  # for validation, use 20% of training samples
       --outputmode density:direct  # instead of e.g. 'color', Sec. 5.3
       --lossmode density
       --layers 32:32:32  # number of hidden feature layers -> that number + 1 for the number of linear layers / weight matrices.
       --activation SnakeAlt:2
       --fouriercount 14
       --fourierstd -1  # -1 indicates NeRF-construction, positive value indicate sigma for random Fourier Features, see Sec. 5.5
       --volumetric_features_resolution 32  # the grid specification, see Sec. 5.2
       --volumetric_features_channels 16
       -l1 1  #use L1-loss with weight 1
       --lr 0.01
       --lr_step 100  #lr reduction after 100 epochs, default lr is used 
       -i 200  # number of epochs
       --save_frequency 20  # checkpoints + test visualization

After training, the resulting `.hdf5` file contains the network weights + latent grid and can be compiled to our binary format via `inference.py`. The resulting `.volnet` file can the be loaded in the GUI.

## How to reproduce the figures

Each figure is associated with a respective script in `applications/volnet`. Those scripts include the training of the networks, evaluation, and plot generation. They have to be launched with the current path pointing to `applications/`. Note that some of those scripts take multiple hours due to the network training.

- Figure 1, teaser: `applications/volnet/eval_CompressionTeaser.py`
- Table 1, possible architectures: `applications/volnet/collect_possible_layers.py`
- Section 4.2, change to performance due to grid compression: `applications/volnet/eval_VolumetricFeatures_GridEncoding`
- Figure 4, performance of the networks:  `applications/volnet/eval_NetworkConfigsGrid.py`
- Figure 5+6, latent grid, also includes other datasets:  `applications/volnet/eval_VolumetricFeatures.py`
- Figure 7, Fourier features:  `applications/volnet/eval_Fourier_Grid.py` , includes the datasets not shown in the paper for space reasons
- Figure 8, density-vs-color:
   `applications/volnet/eval_world_DensityVsColorGrid_NoImportance.py` without initial importance sampling and adaptive resampling (Fig. 6)
  `applications/volnet/eval_world_DensityVsColorGrid.py` , includes initial importance sampling, not shown
  `applications/volnet/eval_world_DensityVsColorGrid_WithResampling.py` , with initial importance sampling and adaptive resampling, improvement reported in Section 5.4
- Figure 9, gradient prediction: `applications/volnet/eval_GradientNetworks1_v2.py`
- Figure 10, curvature prediction: `applications/volnet/eval_CurvatureNetworks2.py`
- Figure 11, 12, comparison with baseline methods: `applications/volnet/eval_CompressionExtended.py`
- Figure 13,14, time-dependent fields:
   `applications/volnet/eval_TimeVolumetricFeatures.py`: train on every fifth timestep
   `applications/volnet/eval_TimeVolumetricFeatures2.py`: train on every second timestep
   `applications/volnet/eval_TimeVolumetricFeatures_plotPaper.py`: assembles the plot for Figure 14

Supplementary Paper:

- Section 1, study on the activation functions:  `applications/volnet/eval_ActivationFunctions.py`
- Table 2, Figure 1, screen-vs-world:  `applications/volnet/eval_ScreenVsWorld_GridNeRF.py`
- Figure 2-6, ablation study for gradients:  `applications/volnet/eval_GradientNetworks1_v2.py`, `applications/volnet/eval_GradientNetworks2.py`
- Figure 8, comparison with baseline methods: `applications/volnet/eval_CompressionExtended.py`

The other `eval_*.py` scripts were cut from the paper due to space limitations. They equal the tests above, except that no grid was used and instead the largest possible networks fitting into the TC-architecture

