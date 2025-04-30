# Backpack Price Prediction

## Description
In this project, we tackle the backpack price prediction problem presented in the Kaggle Playground Series (Season 5, Episode 2). Accurate price prediction is crucial for inventory management, consumer targeting, and market analysis. We perform comprehensive exploratory data analysis (EDA) to understand the underlying patterns and characteristics of the provided datasets. Our approach combines multiple robust regression models—LightGBM, XGBoost, and CatBoost—leveraging Optuna for extensive hyperparameter optimization. The predictions from these base models are then stacked, using linear regression as the meta-model. Our stacking method achieves a Root Mean Square Error (RMSE) of approximately 38.9 on the validation set, demonstrating improved predictive performance over individual base models. We discuss feature importance, the effectiveness of our stacking strategy, and potential areas for future enhancements. The project was a joint effort. Both authors contributed equally to the conception, implementation, analysis, and writing of this study. All sections of the work were discussed, developed, and refined collaboratively.

## Final Deliverables

* 📄 [Final Paper](./INFSCI2160_FinalPaper_DataPacker.pdf)  
* 📊 [Presentation Slides](./PRE%20(1).pdf)


## Team Member
Data Packer:
Ransheng Lin ral183@pitt.edu
Gaoyang Qiao gaq7@pitt.edu

##   Overview

    The notebook `backpack_price_prediction.ipynb` contains the code for data loading, preprocessing, model training, and evaluation. It utilizes LightGBM, XGBoost, and CatBoost as base models, with Optuna for hyperparameter optimization, and a Linear Regression model as the meta-learner for stacking.
##   Data

**The datasets for this project are too large to be stored directly in the repository. Please download them from the following Kaggle competition page:**

[https://www.kaggle.com/competitions/playground-series-s5e2](https://www.kaggle.com/competitions/playground-series-s5e2)

The following datasets are used:

* `train.csv`: Training data.
* `test.csv`: Testing data.
* `training_extra.csv`: Extra training data (not fully utilized in the current implementation due to computational constraints).

##   Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* lightgbm
* xgboost
* catboost
* optuna
* matplotlib
* seaborn

##   Usage

1.  Ensure all required libraries are installed. You can install them using pip:

    ```bash
    pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn
    ```

2.  **Download the datasets from the Kaggle competition page (see "Data" section above) and place them in the same directory as the notebook.**

3.  Run the notebook `backpack_price_prediction.ipynb` to execute the data processing, model training, and prediction pipeline.

##   Exploratory Data Analysis (EDA)

The exploratory data analysis was conducted and is documented in the `eda-notebook.html` file. Key findings and steps included:

* Identification of missing values in the 'Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color', and 'Weight Capacity (kg)' columns.
* Handling of missing values: Numerical columns were filled with the median, and categorical columns with 'unknown' to maintain consistency between training and test sets.
* Inspection of unique values in categorical columns to inform one-hot encoding, ensuring no dimensionality explosion.
* Visualizations (e.g., histograms, bar plots) to understand feature distributions and relationships with price.
* Analysis of feature correlations to identify potential multicollinearity.

For detailed EDA, please refer to the `eda-notebook.html` file.

##   Methodology

1.  **Data Loading**: The datasets are loaded using pandas.
2.  **Data Preprocessing**:
    * Missing values in numerical columns ('Compartments', 'Weight Capacity (kg)') are filled with the median.
    * Missing values in categorical columns ('Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style','Color') are filled with 'unknown'.
    * Categorical features are one-hot encoded using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.
3.  **Model Training**:
    * Hyperparameter tuning is performed for LightGBM, XGBoost, and CatBoost using Optuna to optimize the Root Mean Squared Error (RMSE).
    * A stacking ensemble is created using these three models as base learners.
    * Out-of-Fold (OOF) predictions are generated for each base model using K-Fold cross-validation (with 5 splits) to create meta-features.
    * Linear Regression is used as the meta-learner to combine the predictions of the base models.
4.  **Evaluation**: The stacked model's performance is evaluated using Root Mean Squared Error (RMSE) on a held-out validation set.
5.  **Visualization**:
    * A scatter plot is used to visualize predicted vs. actual prices.
    * A histogram is used to visualize the distribution of residuals (prediction errors).

##   Results

The stacked model achieves a certain RMSE on the validation set (see the notebook output for the exact value). The notebook also provides the best hyperparameters found for each base model:

    ```
    Best parameters for LightGBM: {'max_depth': 6, 'num_leaves': 102, 'min_child_samples': 29, 'learning_rate': 0.04811541008025368, 'subsample': 0.6373168094008929, 'colsample_bytree': 0.6287351039931749, 'reg_alpha': 24.94216574124474, 'reg_lambda': 13.3381875824349, 'random_state': 42}
    Best parameters for XGBoost: {'max_depth': 4, 'learning_rate': 0.0242976335510153, 'subsample': 0.7392383751300301, 'colsample_bytree': 0.7081780053291772, 'alpha': 1.3548669745281217, 'lambda': 0.1300423102945354, 'n_estimators': 445, 'random_state': 42}
    Best parameters for CatBoost: {'depth': 4, 'learning_rate': 0.06257815809830002, 'l2_leaf_reg': 23.148679122356665, 'subsample': 0.6957264471620155, 'iterations': 410, 'verbose': False, 'random_state': 42}
    RMSE of the stacked model: 38.90021
    ```

##   Note

    * The `training_extra.csv` dataset is not fully utilized in the current implementation to manage computational load. Future work could explore incorporating this data.
    * Further improvements might involve exploring additional feature engineering techniques or trying different meta-learners in the stacking ensemble.