🙊 damped
=========

[![pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange.svg?style=flat-square)](https://img.shields.io/badge/made%20with-pytorch-orange.svg?style=flat-square)

**D**omain **A**daptation **M**odule for **P**rivacy **E**nable and **D**istributed learning


`damped` is a library to experiment with hybrid deep learning architecture.
It features a framework to incorporate off-tasks domain classifiers onto existing and well-defined toolkit easily.
The goal is to have a minimal footprint on the main task network architecture, training loop, data preparation, and resource consumption.

## 🔗 References
List of the original papers that are implemented here:
1. Ganin, Y. et al. **Domain-Adversarial Training of Neural Networks**. arXiv:1505.07818 [cs, stat] (2015).
2. Anonymous. **Multi-Step Decentralized Domain Adaptation**. Paper under double-blind review for ICLR 2020 (2019).
3. Osia, S. A. et al. **A Hybrid Deep Learning Architecture for Privacy-Preserving Mobile Analytics**. arXiv:1703.02952 [cs] (2017).
4. Srivastava, B., Bellet, A., Tommasi, M. & Vincent, E. **Privacy-Preserving Adversarial Representation Learning in ASR: Reality or Illusion?** Interspeech (2019) doi:10.21437/Interspeech.2019-2415.

Part of the code in this repository is inspired or borrowed from other implementations, especially:
- https://github.com/fungtion/DANN
- https://github.com/domainadaptation/salad
- https://github.com/espnet/espnet
