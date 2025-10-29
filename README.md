Car Price Prediction Project

https://img.shields.io/badge/Python-3.7%252B-blue
https://img.shields.io/badge/Machine-Learning-orange
https://img.shields.io/badge/Scikit--learn-1.0%252B-green
https://img.shields.io/badge/Status-Completed-brightgreen

📋 Project Overview
This project implements a comprehensive machine learning solution for predicting car prices based on various vehicle features. Using a dataset of 10,000 car listings from Craigslist, we develop and compare multiple regression models to accurately estimate vehicle prices.

🎯 Objectives
Perform comprehensive exploratory data analysis (EDA) on car listing data

Preprocess and clean the dataset for machine learning

Implement and compare multiple regression algorithms

Optimize model performance through hyperparameter tuning

Create a robust price prediction system

📊 Dataset
The dataset contains 10,000 car listings with 20 features including:

Key Features:
price: Target variable (car price)

year: Manufacturing year (1915-2022)

manufacturer: Car manufacturer (40 unique brands)

model: Car model (3,466 unique models)

odometer: Mileage reading

fuel: Fuel type (gas, diesel, hybrid, electric, other)

transmission: Transmission type

title_status: Vehicle title status

state: Geographical location

lat/long: Coordinates

price_category: Pre-classified price range

🛠️ Installation
Prerequisites
Python 3.7+

pip package manager

Project Structure
car-price-prediction/
│
├── data/
│   └── df_out.csv                 # Main dataset
│
├── notebooks/
│   └── car_price_prediction_project_20251024_094500.ipynb  # Main analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py     # Feature creation and selection
│   ├── model_training.py          # Model training functions
│   └── utils.py                   # Utility functions
│
├── models/                        # Saved models
├── results/                       # Output and evaluation results
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── config.yaml                    # Configuration file


🔧 Usage
Data Preprocessing
The notebook includes comprehensive data preprocessing:

Missing value imputation

Data type conversion

Anomaly detection and treatment

Feature encoding

Model Training
The project implements and compares multiple algorithms:

Linear Models

Linear Regression

Ridge Regression

Lasso Regression

Ensemble Methods

Random Forest Regressor

Gradient Boosting Regressor

Other Algorithms

Support Vector Regressor (SVR)

Model Evaluation
Models are evaluated using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R-squared (R²) Score

Cross-validation

📈 Results
Key Findings from EDA:
Price range: $500 - $12,345,680

Most common manufacturers: Ford, Chevrolet, Toyota

Fuel type distribution: 84% gasoline vehicles

Geographical coverage across all US states

Model Performance:
(Results will be populated after model training)

🧪 Technical Details
Libraries Used
Data Manipulation: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

Model Optimization: GridSearchCV, RandomizedSearchCV

🤝 Contributing
We welcome contributions! Please follow these steps:

Fork the project

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset sourced from Craigslist vehicle listings

Scikit-learn community for excellent documentation

Google Colab for computational resources

📞 Contact
For questions or suggestions, please open an issue or contact:

Your Name - Andrey Kuleshov andrejkuleshov1987@gmail.com

Project Link: https://github.com/CrazyNordic19871987/car-price-prediction


