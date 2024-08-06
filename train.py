import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

# load and preprocess the dataset
# dataset url: https://www.kaggle.com/datasets/mirichoi0218/insurance

data = pd.read_csv("insurance.csv")
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['sex'])
data['Smoker'] = le.fit_transform(data['smoker'])
data['Region'] = le.fit_transform(data['region'])

# independent and dependent variables
x = data[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
y = data['charges']

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# normalization of data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# objective function for Optuna hyperparameter framework
def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso'])
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)

    if model_type == 'ridge':
        model = Ridge(fit_intercept=fit_intercept, alpha=alpha)
    else:
        model = Lasso(fit_intercept=fit_intercept, alpha=alpha)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# study object to optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
print("Best MSE: ", study.best_value)

# model training using the best hyperparameters
best_params = study.best_params

if best_params['model_type'] == 'ridge':
    final_model = Ridge(fit_intercept=best_params['fit_intercept'], alpha=best_params['alpha'])
else:
    final_model = Lasso(fit_intercept=best_params['fit_intercept'], alpha=best_params['alpha'])

final_model.fit(x_train, y_train)

# saving the final model
joblib.dump(final_model, "expense_model.joblib")
print("Training with the best hyperparameters completed and model saved")
