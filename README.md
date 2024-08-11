# ML-UsefulnessPredictor

This repository contains the implementation of a machine learning model designed to predict the usefulness of a medical kit offered by an airline company. This project was developed as part of a coding challenge and involves building a predictive model based on passenger and transaction data.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Processing](#data-processing)
- [Model Development](#model-development)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

### Problem Statement:
An airline company has launched a new medical kit that passengers can purchase during check-in. The company collected data from passengers who bought the kit and wants to analyze its usefulness. Your task is to build a model that predicts whether the kit is useful (0) or not useful (1) based on the provided data.

### Task:
The goal is to develop a predictive model that determines the usefulness of the medical kit using the given dataset.

### Dataset Description:
The dataset includes the following files:
- **train.csv**: 6736 rows and 10 columns
- **test.csv**: 2164 rows and 9 columns

Key columns in the dataset:
- `ID`: Unique identification for each entry
- `Distributor`: Distributor's code
- `Product`: Product's code
- `Duration`: Time taken to reach the destination
- `Destination`: Destination's code
- `Sales`: Sale price
- `Commission`: Commission charged by the distributor
- `Gender`: Gender of the passenger
- `Age`: Age of the passenger
- `Target`: Target variable (0: Useful, 1: Not useful)

**Evaluation Metric:**  
The model's performance is evaluated using the weighted F1 score.

## Data Processing

The data processing pipeline includes:
- Handling missing values
- Feature selection and extraction
- Data normalization and scaling
- Splitting the dataset into training and testing sets

## Model Development

Multiple machine learning algorithms were explored to identify the best-performing model. Techniques such as cross-validation and hyperparameter tuning were employed to optimize model accuracy.

The models tested include:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machines (SVM)
- Gradient Boosting

## Evaluation

The models were evaluated based on the weighted F1 score. The final model was selected for its superior performance on the validation set.

## Usage

To replicate the results, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML-UsefulnessPredictor-Challenge.git

## Usage

To replicate the results, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ML-UsefulnessPredictor.git
   ```
2. **Install the required dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
3. **Open the Jupyter Notebook:**
  ```bash
  jupyter notebook predict-usefulness-kit.ipynb
  ```
4. **Execute the cells to see the entire process, from data preprocessing to model evaluation.**
 
## Requirements
To run the notebooks, you will need the following Python packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Contributing
I welcome contributions! If you have suggestions, improvements, or find any errors, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
