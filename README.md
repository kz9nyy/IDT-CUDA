# IDT-CUDA
Improved Dense Trajectories using OpenCV4 + CUDA

This is a fork of the Improved Dense Trajectories implementation developed by Heng Weng

*Copyright (C) 2011 Heng Wang*
Please cite the original work when using this code:
@inproceedings{wang2011action,
  title={Action recognition by dense trajectories},
  author={Wang, Heng and Kl{\"a}ser, Alexander and Schmid, Cordelia and Liu, Cheng-Lin},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on},
  pages={3169--3176},
  year={2011},
  organization={IEEE}
}

I modified Weng's original work merely as an exercise to learn more about optical flow as well as how to use the OpenCV GPU module.
Therefore, my hope is that this repo may serve as a useful reference to anyone interested in optical flow and the CUDA library.

Many of the functions used in the original original work can now be calculated on the GPU.
I tested using a NVIDIA RTX 2080 and it is blazing fast compared to the original implementation!

## Compiling
In order to compile the improved trajectories code, you need to have the following libraries installed in your system:
* **OpenCV-4.0.1** (needs to be compiled with non_free and gpu modules)
* **NVIDIA_CUDA-10.0**
