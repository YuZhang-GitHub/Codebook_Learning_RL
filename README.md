# Neural Networks Based Beam Codebooks: Learning mmWave Massive MIMO Beams that Adapt to Deployment and Hardware
This is the Python codes related to the following article: Muhammad Alrabeiah, Yu Zhang, and Ahmed Alkhateeb, “[Neural Networks Based Beam Codebooks: Learning mmWave Massive MIMO Beams that Adapt to Deployment and Hardware](https://arxiv.org/pdf/2006.14501),” arXiv e-prints, p. arXiv:2006.14501, Jun 2020.
# Abstract of the Article
Millimeter wave (mmWave) and massive MIMO systems are intrinsic components of 5G and beyond. These systems rely on using beamforming codebooks for both initial access and data transmission. Current beam codebooks, however, generally consist of a large number of narrow beams that scan all possible directions, even if these directions are never used. This leads to very large training overhead. Further, these codebooks do not normally account for the hardware impairments or the possible non-uniform array geometries, and their calibration is an expensive process. To overcome these limitations, this paper develops an efficient online machine learning framework that learns how to adapt the codebook beam patterns to the specific deployment, surrounding environment, user distribution, and hardware characteristics. This is done by designing a novel complex-valued neural network architecture in which the neuron weights directly model the beamforming weights of the analog phase shifters, accounting for the key hardware constraints such as the constant-modulus and quantized-angles. This model learns the codebook beams through online and self-supervised training avoiding the need for explicit channel state information. This respects the practical situations where the channel is either unavailable, imperfect, or hard to obtain, especially in the presence of hardware impairments. Simulation results highlight the capability of the proposed solution in learning environment and hardware aware beam codebooks, which can significantly reduce the training overhead, enhance the achievable data rates, and improve the robustness against possible hardware impairments.

# How to generate this codebook beam patterns figure?
1. Download all the files of this repository.
2. Run `main.py`.
3. When `main.py` finishes, load `theta_self_sup_64beams.mat` in Matlab.
4. Run `plot_pattern((1/sqrt(size(codebook,1)))*exp(1j*codebook))` in Matlab Command Window, which will give the codebook beam patterns figure as shown below.

![Figure](https://github.com/YuZhang-GitHub/CBL_Self_Supervised/blob/master/codebook_64.png)

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Muhammad Alrabeiah, Yu Zhang, and Ahmed Alkhateeb, “[Neural Networks Based Beam Codebooks: Learning mmWave Massive MIMO Beams that Adapt to Deployment and Hardware](https://arxiv.org/pdf/2006.14501),” arXiv e-prints, p. arXiv:2006.14501, Jun 2020.
