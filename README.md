
# Binary Classification Streamlit App 🍄

This Streamlit app is designed to classify mushrooms as either edible or poisonous using binary classification models. It provides an interactive interface for training, evaluating, and comparing different machine learning models.  
**Note**: This project is part of the **CPE312 Intro to Data Science** course for Computer Engineering at SWU.

## Features
- **Model Selection**: Choose between SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Adjust model hyperparameters for customized training.
- **Performance Metrics**: View metrics including Accuracy, Precision, Recall, Confusion Matrix, ROC Curve, and Precision-Recall Curve.
- **Data Display**: Option to view raw mushroom dataset.

## Getting Started

### Prerequisites
Ensure you have the required libraries:

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

### Project Structure
- `app.py`: Main Streamlit app file that contains the logic for loading data, training models, and displaying results.
- `data/`: Directory containing the `mushrooms.csv` dataset.
- `requirements.txt`: Lists required dependencies.

### Code Explanation
- **Loading Data**: The `load_data` function reads the mushroom dataset from a CSV file and encodes categorical features using `LabelEncoder`.
- **Data Splitting**: The `split_data` function divides the dataset into training and testing sets using `train_test_split` from `scikit-learn`.
- **Model Training**: Depending on the user’s selection, the app trains one of three classifiers (SVM, Logistic Regression, Random Forest) with user-defined hyperparameters.
- **Performance Evaluation**: The app calculates and displays performance metrics (Accuracy, Precision, Recall) and visualizes them using plots like Confusion Matrix, ROC Curve, and Precision-Recall Curve via `matplotlib` and `scikit-learn`.

### Usage
Use the sidebar to select a classifier, adjust parameters, and view model performance metrics. The app dynamically displays evaluation metrics to help analyze each model's effectiveness.

### Citations
1. Mushroom [Dataset]. (1981). UCI Machine Learning Repository. https://doi.org/10.24432/C5959T.
2. Build a Machine Learning Web App with Streamlit and Python. Coursera Project Network.
