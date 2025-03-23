# Dynamic Room Pricing Prediction

This repository contains code for predicting dynamic room pricing using machine learning models. The models implemented include:

* **XGBoost Regressor**
* **Linear Regression**
* **Random Forest Regressor**

## Dataset

The dataset used for this project was custom-prepared.

## Implementation

The implementation focuses on the machine learning , covering the following stages:

1.  **Data Preprocessing:**
    * Data cleaning and handling missing values using `SimpleImputer`.
    * Feature engineering and selection.
    * Encoding categorical variables using `OneHotEncoder`.
    * Scaling numerical features using `StandardScaler`.
    * Handling date and time features using `datetime` and `timedelta`.
    * Splitting the dataset into training and testing sets using `train_test_split`.
2.  **Model Training:**
    * Training XGBoost Regressor using `XGBRegressor`.
    * Training Linear Regression using `LinearRegression`.
    * Training Random Forest Regressor using `RandomForestRegressor`.
    * Hyperparameter tuning (if applicable).
3.  **Model Evaluation:**
    * Evaluating the performance of each model using:
        * **R-squared (Coefficient of Determination) using `r2_score`**
        * **Root Mean Squared Error (RMSE) using `mean_squared_error`**
    * Data Visualization using `matplotlib.pyplot` and `seaborn`.
    * Data manipulation using `pandas` and `numpy`.

## Files

* `dynamic_room_pricing.ipynb`: Jupyter Notebook containing the complete machine learning pipeline.

## Dependencies

To run the code, you will need the following Python libraries:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

1.  Clone the repository:

    ```bash
    git clone https://github.com/ac265640/Dynamic_room_pricing
    cd Dynamic_room_pricing
    ```

2.  Install the dependencies:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

3.  Open and run the Jupyter Notebook `dynamic-room-pricing.ipynb`.

## Future Work

* Web integration: Develop a web application to deploy the trained models and provide dynamic room pricing predictions.
* Further feature engineering: Explore additional features and feature engineering techniques to improve model performance.
* More model evaluation metrics.
* Implement cross validation.
* Save the model.

## Author

Amit Singh Chauhan/ac265640
