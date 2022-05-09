import argparse

import importlib
import json
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score     
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


TEST_SIZE = 0.2
RANDOM_STATE = 13
SCORING = 'r2'
CROSS_VALIDATION_NUM_FOLD=10
VERBOSE = 2
NUM_JOBS= -1


def get_categorical_numerical_features(df):
  numerical_features = df.select_dtypes(include='number').columns.tolist()
  categorical_features = df.select_dtypes(exclude='number').columns.tolist()
  return categorical_features, numerical_features


def get_preprocessor(categorical_features, numerical_features):
  numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', MinMaxScaler())
  ])

  categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
  ])
  
  preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
  ])

  return preprocessor


def predict(inputs_file, models_file, output_file):
  print(f'inputs_file: {inputs_file}')
  df_in = pd.read_csv(inputs_file)
  print(df_in.head())

  print(f'models_file: {models_file}')
  with open(models_file, 'r') as fp:
    models = json.load(fp)
  print(models)

  o_file = open(output_file, 'w')
  o_file.write('input_file,regressor,label_name,best_score(' + SCORING + '),best_parameters,test_score(' + SCORING + ')\n')

  # Iterate over each file in input_files
  for index, row in df_in.iterrows():
    input_file = row['input_file']
    label_name = row['label_name']
    print(f'input_file: {input_file}, label_name: {label_name}')
    df = pd.read_csv(input_file)

    X = df.drop(columns=[label_name], axis=1)
    y = df[label_name] 

    categorical_features, numerical_features = get_categorical_numerical_features(X)

    num_numerical_features = len(numerical_features)
    print(f'numerical features of the {input_file}')
    print(numerical_features)
    print(f'Number of numerical featuress: {num_numerical_features}')

    num_categorical_features = len(categorical_features)
    print(f'categorical features of the {input_file}')
    print(categorical_features)
    print(f'Number of categorical features: {num_categorical_features}')

    num_features = X.shape[1]
    print(f'Total number of features of the {input_file}: {num_features}')
    if num_numerical_features + num_categorical_features != X.shape[1]:
      raise Exception(f'Number of numerical features ({num_numerical_features}) plus \
              number of categorical features ({num_categorical_features}) \
              does not match the total number of features ({num_features})')
    
    preprocessor = get_preprocessor(categorical_features, numerical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Iterate over each model in models file
    for model_name in models:
      print(f'model_name: {model_name}')
      module_name = models[model_name]["module_name"]
      class_name = models[model_name]["class_name"]
      parameters = models[model_name]["parameters"]
      print(f'module_name: {module_name}')
      print(f'class_name: {class_name}')
      print(f'parameters: {parameters}')

      module = importlib.import_module(module_name)
      class_ = getattr(module, class_name)
      regressor = class_()
    
      regressor_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor)
      ])

      grid_search_cv = GridSearchCV(regressor_pipeline,
                                    parameters,
                                    scoring=SCORING,
                                    cv=CROSS_VALIDATION_NUM_FOLD,
                                    verbose=VERBOSE,
                                    n_jobs=NUM_JOBS)

      _ = grid_search_cv.fit(X_train, y_train)
      print(f'scorer_: {grid_search_cv.scorer_}')
      print(f'\nRegressor name: {class_name}')
      print(f'input_file: {input_file}, label_name: {label_name}')
      print(f'Best score for {class_name}: {grid_search_cv.best_score_}')
      print(f'Best params for {class_name}: {grid_search_cv.best_params_}')

     
      best_params_list = [str(k) +'='+ str(v) for k,v in grid_search_cv.best_params_.items()]
      best_params_str = ";".join(best_params_list)
      print(f'best_params_str: {best_params_str}')

      y_predicted = grid_search_cv.predict(X_test)
      prediction_score = r2_score(y_test, y_predicted) 
      print(f'prediction_score: {prediction_score}')

      o_file.write(",".join([input_file, model_name, label_name, str(grid_search_cv.best_score_), best_params_str, str(prediction_score)])+'\n')

  o_file.close()

if __name__ == '__main__':
  argument_parser = argparse.ArgumentParser('Runtime prediction argument parser')
  argument_parser.add_argument('--input_files', '-i', type=str, required=True)
  argument_parser.add_argument('--models', '-m', type=str, required=True)
  argument_parser.add_argument('--output_file', '-o', type=str, required=True)
  args = argument_parser.parse_args()
  predict(args.input_files, args.models, args.output_file)
