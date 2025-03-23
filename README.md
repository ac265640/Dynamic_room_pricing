```markdown
# Dynamic Room Pricing Prediction

This repository contains code for predicting dynamic room pricing using machine learning models. The models implemented include:

* **XGBoost Regressor**
* **Linear Regression**
* **Random Forest Regressor**

## Dataset

The dataset used for this project was custom-prepared. It contains features related to room characteristics, demand, time-based factors, and other relevant variables that influence room pricing.

**Note:** Due to privacy or specific project requirements, the dataset itself is not included in this repository.

## Implementation

The implementation focuses on the machine learning pipeline, covering the following stages:

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
* `requirements.txt`: List of Python dependencies.

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
pip install -r requirements.txt
```

or by manually installing them:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd dynamic_room_pricing
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Open and run the Jupyter Notebook `dynamic_room_pricing.ipynb`.

## Future Work

* Web integration: Develop a web application to deploy the trained models and provide dynamic room pricing predictions.
* Further feature engineering: Explore additional features and feature engineering techniques to improve model performance.
* More model evaluation metrics.
* Implement cross validation.
* Save the model.

## Author

Amit Singh Chauhan/ac265640
