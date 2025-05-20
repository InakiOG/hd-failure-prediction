# hd-failure-prediction

This repository contains the code and resources for my master's thesis in Systems Engineering at ITESO University, developed in collaboration with Intel.

## Project Overview

The goal of this project is to predict hard drive failures using machine learning models trained on SMART (Self-Monitoring, Analysis, and Reporting Technology) data collected from hard drives. Early detection of potential failures can help prevent data loss and improve system reliability.

## Models

This project implements and compares two different machine learning models:

- **Classification Tree**: A decision tree-based classifier that predicts if the SMART data of a drive indicates that this has failed.

- **Transformer**: A deep learning model based on the transformer architecture, designed to proyect using time series data SMART information about drives in the future to be fed into the classification tree.

## Dataset

The dataset consists of multiple CSV files containing SMART attributes provided by backblaze, they have failure labels for a large number of hard drives. The data is stored in the `data/` directory.

## Project Structure

- `data/` - Raw CSV files with SMART data.
- `models/` - Python scripts for the classification tree and transformer models.
- `src/` - Data processing, feature engineering, and utility scripts.
- `notebooks/` - Jupyter notebooks for data exploration and experimentation.

## Getting Started

1. Place your dataset CSV files in the `data/` folder.
2. Install the required Python dependencies (see `requirements.txt`).
3. Run the scripts in `src/` for data preprocessing.
4. Train and evaluate the models in `models/` or use the provided notebooks.

## Acknowledgements

- ITESO University
- Intel Corporation

For more details, please refer to the documentation in each folder and the thesis report.
