# Reinforcement Learning of Beam Codebooks in Millimeter Wave and Terahertz MIMO Systems
This is the Python codes related to the following article: Yu Zhang, Muhammad Alrabeiah, and Ahmed Alkhateeb, “[Reinforcement Learning of Beam Codebooks in Millimeter Wave and Terahertz MIMO Systems](https://ieeexplore.ieee.org/document/9610084),” in IEEE Transactions on Communications, 2021.
# Abstract of the Article
Millimeter wave (mmWave) and terahertz MIMO systems rely on pre-defined beamforming codebooks for both initial access and data transmission. These pre-defined codebooks, however, are commonly not optimized for specific environments, user distributions, and/or possible hardware impairments. This leads to large codebook sizes with high beam training overhead which makes it hard for these systems to support highly mobile applications. To overcome these limitations, this paper develops a deep reinforcement learning framework that learns how to optimize the codebook beam patterns relying only on the receive power measurements. The developed model learns how to adapt the beam patterns based on the surrounding environment, user distribution, hardware impairments, and array geometry. Further, this approach does not require any knowledge about the channel, RF hardware, or user positions. To reduce the learning time, the proposed model designs a novel Wolpertinger-variant architecture that is capable of efficiently searching the large discrete action space. The proposed learning framework respects the RF hardware constraints such as the constant-modulus and quantized phase shifter constraints. Simulation results confirm the ability of the developed framework to learn near-optimal beam patterns for line-of-sight (LOS), non-LOS (NLOS), mixed LOS/NLOS scenarios and for arrays with hardware impairments without requiring any channel knowledge.

# How to regenerate Fig. 7(c) in [this](https://ieeexplore.ieee.org/document/9610084) paper?
**Disclaimer:** The following dependencies have been verified to have no compatibility issues with the developed codes. However, lower or higher versions might also work fine. Furthermore, a successful execution might also require extra support from the hardware, e.g., the CPU/GPU capabilities, available RAM, etc.
## Dependencies
1. Python 3.8.8
2. Pytorch 1.9.1
3. Numpy 1.20.1
4. Scipy 1.6.2
5. Sklearn 0.24.1
6. NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit))

## Following these steps
1. Download all the files of this repository.
2. Run `main.py`.
3. When `main.py` finishes, run `read_beams.py`.
4. Load `beam_codebook.mat` in Matlab.
5. Run `plot_pattern(beams.')` in Matlab Command Window, which will give Fig. 7(c) shown below as result.

![Figure](https://github.com/YuZhang-GitHub/CBL_RL/blob/main/LOS_4beams.png)

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).  
To find more information about the paper and other deep-learning based wireless communication work, please visit [DeepMIMO dataset applications](https://deepmimo.net/applications/).  
To generate your own dataset, please visit [DeepMIMO.net](https://deepmimo.net/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Yu Zhang, Muhammad Alrabeiah, and Ahmed Alkhateeb, “[Reinforcement Learning of Beam Codebooks in Millimeter Wave and Terahertz MIMO Systems](https://ieeexplore.ieee.org/document/9610084),” in IEEE Transactions on Communications, 2021.
