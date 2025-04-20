# Titanic Survival Prediction

## 1. Introduction
This project investigates passenger survival on the RMS Titanic using supervised machine learning. The objective is to develop a predictive model that estimates survival probability based on passenger demographics, ticket information, and voyage details. The workflow comprises data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## 2. Data Description
The Kaggle dataset includes:
- **train.csv**: 891 records with `Survived` target.
- **test.csv**: 418 records without `Survived`.

Attributes:
- `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

## 3. Data Preprocessing
1. **Separate Processing**  
   - Training and test sets were processed independently.  
2. **Ground Truth Merge**  
   - After prediction, the test set’s `PassengerId` and predicted `Survived` values were merged with the true `Survived` labels from `gender_submission.csv` for evaluation.  
3. **Missing Value Imputation**  
   - **Age**: imputed with the mean age of the training set.  
   - **Fare**: the single missing value in the test set was imputed with the median fare.  
   - **Embarked**: missing entries filled with the most frequent port (‘S’) in the training set.  
4. **Feature Encoding**  
   - Categorical features (`Sex`, `Embarked`) were label-encoded using scikit-learn’s `LabelEncoder`.
5. **Feature Selection**  
   - Dropped sparse or non-predictive features: `Name`, `Ticket`, `Cabin`, `PassengerId`.

## 4. Exploratory Data Analysis
- **Age distribution**: right-skewed, with concentration in early adulthood.  
- **Fare distribution**: long-tailed, indicating high-variance ticket prices.  
- **Survival by Sex**: females had markedly higher survival than males.  
- **Survival by Class**: first-class passengers were more likely to survive than those in lower classes.  
- **Correlation**: `Pclass`, `Sex`, and `Fare` emerged as strong predictors.

## 5. Model Training and Evaluation
- **Algorithm**: XGBoost exclusively.  
- **Evaluation**: The model was trained on the full training set. Performance was assessed by comparing predicted and true labels on the test set.
