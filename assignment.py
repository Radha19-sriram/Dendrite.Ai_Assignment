import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def parse_config(filepath):
    """
    Parse the configuration file in JSON format.
    
    Args:
        filepath (str): Path to the JSON configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

def create_pipeline(config, X):
    """
    Create a data preprocessing pipeline based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        X (pd.DataFrame): Feature data.
    
    Returns:
        Pipeline: Preprocessing pipeline.
    """
    # Identify numerical and categorical columns
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    # Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('scaler', StandardScaler())  # Standardize features by removing the mean and scaling to unit variance
    ])
    
    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent value
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features as a one-hot numeric array
    ])
    
    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Initialize steps with the preprocessor
    steps = [('preprocessor', preprocessor)]

    # Add feature reduction step if specified in config
    if 'feature_reduction_method' in config['design_state_data']['feature_reduction']:
        n_components = int(config['design_state_data']['feature_reduction']['num_of_features_to_keep'])
        steps.append(('pca', PCA(n_components=n_components)))  # Apply PCA for feature reduction

    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline

def main():
    """
    Main function to execute the data pipeline, model training, and evaluation.
    """
    config_path = 'C:/Users/Admin/PycharmProjects/Dendrite.Ai/Screening Test - DS/algoparams_from_ui.json'
    data_path = 'C:/Users/Admin/PycharmProjects/Dendrite.Ai/Screening Test - DS/iris.csv'
    
    # Parse the configuration file
    config = parse_config(config_path)
    print("Loaded config:", config)
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Separate features and target variable
    target = config['design_state_data']['target']['target']
    X = data.drop(columns=[target])
    y = data[target]

    # Get train-test split parameters from config
    train_ratio = config['design_state_data']['train']['train_ratio']
    random_seed = config['design_state_data']['train']['random_seed']
    print(f"train_ratio: {train_ratio}, random_seed: {random_seed}")
    
    # Validate train ratio
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_seed)
    
    # Create the preprocessing pipeline
    base_pipeline = create_pipeline(config, X)
    
    # Initialize models and their parameters
    models = []
    params = {}
    
    # Configure RandomForestRegressor if selected in config
    if config['design_state_data']['algorithms']['RandomForestRegressor']['is_selected']:
        rf = Pipeline(steps=[('base', base_pipeline), ('rf', RandomForestRegressor())])
        min_trees = config['design_state_data']['algorithms']['RandomForestRegressor']['min_trees']
        max_trees = config['design_state_data']['algorithms']['RandomForestRegressor']['max_trees']
        min_depth = config['design_state_data']['algorithms']['RandomForestRegressor']['min_depth']
        max_depth = config['design_state_data']['algorithms']['RandomForestRegressor']['max_depth']
        min_samples_per_leaf_min_value = config['design_state_data']['algorithms']['RandomForestRegressor']['min_samples_per_leaf_min_value']
        min_samples_per_leaf_max_value = config['design_state_data']['algorithms']['RandomForestRegressor']['min_samples_per_leaf_max_value']
        
        # Define hyperparameters grid for RandomForestRegressor
        params['rf__n_estimators'] = [min_trees, max_trees]
        params['rf__max_depth'] = [min_depth, max_depth]
        params['rf__min_samples_leaf'] = [min_samples_per_leaf_min_value, min_samples_per_leaf_max_value]
        
        models.append(('RandomForestRegressor', rf, params))
    
    # Configure GradientBoostingRegressor if selected in config
    if config['design_state_data']['algorithms']['GBTRegressor']['is_selected']:
        gbr = Pipeline(steps=[('base', base_pipeline), ('gbr', GradientBoostingRegressor())])
        num_boosting_stages = config['design_state_data']['algorithms']['GBTRegressor']['num_of_BoostingStages']
        min_depth = config['design_state_data']['algorithms']['GBTRegressor']['min_depth']
        max_depth = config['design_state_data']['algorithms']['GBTRegressor']['max_depth']
        min_stepsize = config['design_state_data']['algorithms']['GBTRegressor']['min_stepsize']
        max_stepsize = config['design_state_data']['algorithms']['GBTRegressor']['max_stepsize']
        min_iter = config['design_state_data']['algorithms']['GBTRegressor']['min_iter']
        max_iter = config['design_state_data']['algorithms']['GBTRegressor']['max_iter']
        
        # Define hyperparameters grid for GradientBoostingRegressor
        params['gbr__n_estimators'] = num_boosting_stages
        params['gbr__max_depth'] = [min_depth, max_depth]
        params['gbr__learning_rate'] = [min_stepsize, max_stepsize]
        params['gbr__min_samples_split'] = [min_iter, max_iter]
        
        models.append(('GBTRegressor', gbr, params))

    # Perform Grid Search for each model
    for name, model, params in models:
        grid_search = GridSearchCV(model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        
        # Print the best parameters and score for each model
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best score for {name}: {grid_search.best_score_}")

if __name__ == "__main__":
    main()
