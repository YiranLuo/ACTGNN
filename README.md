# ACTGNN: Assessment of Clustering Tendency with Synthetically-Trained Graph Neural Networks
## Overview
This repository contains the code for the paper "ACTGNN: Assessment of Clustering Tendency with Synthetically-Trained Graph Neural Networks."

ACTGNN is a novel method for assessing the clustering tendency of a dataset using Graph Neural Networks (GNNs). Unlike traditional non-learning-based methods such as the Hopkins Statistic and VAT, ACTGNN leverages a trained GNN model to determine whether a dataset exhibits a k-means clustering structure. This approach enhances clustering assessment by utilizing graph representations and deep learning techniques.

## Features
* Transforms datasets into graph representations using k-nearest neighbors.

* Supports multiple node and edge feature extraction methods, including Euclidean distance, cosine similarity, and radial basis function (RBF) kernel.

* Utilizes a Graph Convolutional Network (GCN) with five convolutional layers and global mean pooling for classification.

* Provides a binary classification decision on whether a dataset has an inherent k-means clustering structure.


## Acknowledgement
This software was created by the University of California – Riverside under Army Research Office (ARO) Award Number W911NF-24-1-0397. ARO, as the Federal awarding agency, reserves a royalty-free, nonexclusive, and irrevocable right to reproduce, publish, or otherwise use this software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b).
