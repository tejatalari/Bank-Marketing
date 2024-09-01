

# Bank Marketing Campaign Prediction Using Deep Learning

This repository contains a project focused on predicting the success of a bank marketing campaign using deep learning techniques. The objective is to determine whether a client will subscribe to a term deposit based on various features related to the client's profile and past interactions with the bank.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, we use deep learning models to predict whether a customer will subscribe to a term deposit based on features such as age, job, marital status, and previous interactions with the bank. The project aims to provide valuable insights into customer behavior and help optimize marketing strategies.

## Dataset

The dataset used for this project is the "Bank Marketing" dataset, which is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The dataset includes the following features:

- `age`: The age of the client.
- `job`: Type of job.
- `marital`: Marital status.
- `education`: Level of education.
- `default`: Whether the client has credit in default.
- `balance`: The average yearly balance in euros.
- `housing`: Whether the client has a housing loan.
- `loan`: Whether the client has a personal loan.
- `contact`: Contact communication type.
- `day`: Last contact day of the month.
- `month`: Last contact month of the year.
- `duration`: Last contact duration, in seconds.
- `campaign`: Number of contacts performed during this campaign.
- `pdays`: Number of days since the client was last contacted from a previous campaign.
- `previous`: Number of contacts performed before this campaign.
- `poutcome`: Outcome of the previous marketing campaign.
- `y`: The target variable indicating whether the client subscribed to a term deposit (yes/no).

## Installation

To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required dependencies using pip:

```bash
git clone https://github.com/your-username/bank-marketing-deep-learning.git
cd bank-marketing-deep-learning
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

## Data Preprocessing

Before feeding the data into the deep learning model, several preprocessing steps are performed:

1. **Data Cleaning**: Handle missing values and inconsistencies in the dataset.
2. **Encoding Categorical Variables**: Convert categorical variables into numerical form using techniques such as one-hot encoding or label encoding.
3. **Feature Scaling**: Normalize or standardize features to improve model convergence.
4. **Train-Test Split**: Split the dataset into training and testing sets to evaluate model performance.

## Modeling

A deep learning model is built using TensorFlow/Keras to predict the target variable. The model architecture typically includes:

- **Input Layer**: Accepts the feature vector.
- **Hidden Layers**: Multiple dense layers with activation functions such as ReLU (Rectified Linear Unit) to learn complex patterns.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

The model is compiled with the following settings:

- **Loss Function**: Binary Crossentropy, suitable for binary classification tasks.
- **Optimizer**: Adam, an efficient optimization algorithm for training deep learning models.
- **Metrics**: Accuracy, used to evaluate the performance of the model.

## Evaluation

The model's performance is evaluated using various metrics:

- **Accuracy**: The percentage of correctly predicted outcomes out of the total predictions.
- **Precision, Recall, and F1-Score**: Metrics to evaluate the model's ability to handle class imbalances.
- **Confusion Matrix**: A matrix that summarizes the performance of the classification model.
- **AUC-ROC Curve**: A plot that illustrates the model's capability of distinguishing between classes.

## Results

The results section presents the performance of the deep learning model on the test set, highlighting key metrics like accuracy, precision, recall, and F1-score. The AUC-ROC curve is also analyzed to assess the model's predictive power.

## Visualization

The project includes visualizations to aid in understanding the data and model performance:

- **Correlation Heatmap**: Shows the correlation between different features.
- **Distribution Plots**: Visualize the distribution of features and target variables.
- **ROC Curve**: Graphically represents the performance of the classification model.
- **Confusion Matrix Heatmap**: Displays the confusion matrix as a heatmap for better understanding.

## Usage

To run the prediction model on the Bank Marketing dataset, execute the Python script provided in this repository. The script will preprocess the data, train the deep learning model, and display the results, including visualizations.

```bash
python bank_marketing_deep_learning.py
```

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or enhancements, feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


