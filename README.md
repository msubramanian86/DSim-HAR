
# DSimGT-HAR: Dual Similitude Graph Transformer for Human Activity Recognition

This repository contains the code, datasets, and model for the **Dual Similitude Graph Transformer (DSimGT)** designed for **Human Activity Recognition (HAR)**. The proposed DSimGT model improves HAR performance by leveraging dual graph construction and edge-aware attention, capturing both direct and indirect inter-sensor dependencies.

## Overview

This repository provides:
- The **DSimGT architecture** for HAR
- Code for **data preprocessing**, **model training**, and **evaluation**
- Preprocessed datasets for **PAMAP2**, **OPPORTUNITY**, and **MHEALTH**
  
It includes the model architecture, evaluation benchmarks, and training pipelines to replicate the results of the research.

## Paper

This work is presented in the paper:

**Dual Similitude Graph Transformer Model for Human Activity Recognition**,  
by *[Your Name]* and *[Co-author Names]*,  
**Machine Learning: Science and Technology** (2025).  
DOI: [DOI if available]

You can read the full paper [here](insert-link-to-paper).

## Features

- **Dual Graph Construction**: Constructs two complementary graphs — one based on Pearson correlation and the other on convolutional feature correlations.
- **Dual Similitude Graph Transformer (DSimGT)**: A novel graph transformer that incorporates both direct and indirect edge information for more accurate and interpretable HAR.
- **State-of-the-art performance**: DSimGT outperforms traditional CNN, LSTM, and GNN-based methods on the **PAMAP2**, **OPPORTUNITY**, and **MHEALTH** datasets.

## Datasets

We provide preprocessed versions of the following datasets used for the experiments:

- **PAMAP2**: A large-scale activity recognition dataset with multiple sensor modalities.
- **OPPORTUNITY**: A multimodal sensor dataset including ambient, wearable, and object sensors.
- **MHEALTH**: A dataset for human activity recognition collected from wearable devices.

You can access the raw datasets from the following links:
- [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)

The preprocessed datasets are included in this repository as `.npy` files for easier use.

## Installation

To use the code and datasets, please follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/msubramanian86/DSim-HAR.git
    cd DSimGT-HAR-Data
    ```



