# JSHFL: 
# Defending Against Poisoning Attacks in Hierarchical Federated Learning by Utilizing Jensen-Shannon Divergence

## Overview
This repository contains the official implementation of JSHFL as described in the paper "Defending Against Poisoning Attacks in Hierarchical Federated Learning by Utilizing Jensen-Shannon Divergence". Our method leverages the Jensen-Shannon Divergence to enhance the robustness of hierarchical federated learning systems against various untargeted malcious poisoning attacks. Furthermore, JSHFL enhances the robustness of the aggregation process by applying a survival-of-the-fittest rule to regulate the aggregation weights at the edge server.

## Datasets

This implementation has been tested on the following datasets:
- MNIST
- Fashion-MNIST
- CIFAR-10

## Model Architectures

We have integrated JSHFL with the following model architectures:
- Convolutional Neural Networks (CNN)
- ResNet-20

## Installation
- Python 3.6+
- PyTorch 1.7+
- Other Python libraries as listed in `requirements.txt`
- To set up the environment and install the required dependencies, follow these steps:
  ```bash
  git clone https://github.com/yourusername/JSHFL.git
  cd JSHFL
  pip install -r requirements.txt
  
## Training JSHFL

There are two ways to run the training process for JSHFL. First, navigate to the appropriate directory:

```bash
cd JSHFL/fedml_experiments/standalone/fedavg/

Method 1:
Run the training using the run_for_methods_attacks.sh script:
bash run_for_methods_attacks.sh

Method 2:
Alternatively, you can use the run_fedavg_standalone_pytorch.sh script with specific parameters:

sh run_fedavg_standalone_pytorch.sh 0 100 10 8 mnist "mnist" cnn hetero 100 10 0.001 sgd 0 30 JSHFL gaussian

