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
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

def create_pipeline(config, X):
    # Feature Handling
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Initialize steps with the preprocessor
    steps = [('preprocessor', preprocessor)]

    # Feature Reduction
    if 'feature_reduction_method' in config['design_state_data']['feature_reduction']:
        n_components = int(config['design_state_data']['feature_reduction']['num_of_features_to_keep'])
        steps.append(('pca', PCA(n_components=n_components)))

    pipeline = Pipeline(steps)
    
    return pipeline

def main():
    config_path = 'C:/Users/Admin/PycharmProjects/Dendrite.Ai/Screening Test - DS/algoparams_from_ui.json'
    data_path = 'C:/Users/Admin/PycharmProjects/Dendrite.Ai/Screening Test - DS/iris.csv'
    config = parse_config(config_path)
    print("Loaded config:", config)
    
    data = pd.read_csv(data_path)
    
    target = config['design_state_data']['target']['target']
    X = data.drop(columns=[target])
    y = data[target]

    train_ratio = config['design_state_data']['train']['train_ratio']
    random_seed = config['design_state_data']['train']['random_seed']
    print(f"train_ratio: {train_ratio}, random_seed: {random_seed}")
    
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_seed)
    
    base_pipeline = create_pipeline(config, X)
    
    # Algorithms and their parameters
    models = []
    params = {}
    
    if config['design_state_data']['algorithms']['RandomForestRegressor']['is_selected']:
        rf = Pipeline(steps=[('base', base_pipeline), ('rf', RandomForestRegressor())])
        min_trees = config['design_state_data']['algorithms']['RandomForestRegressor']['min_trees']
        max_trees = config['design_state_data']['algorithms']['RandomForestRegressor']['max_trees']
        min_depth = config['design_state_data']['algorithms']['RandomForestRegressor']['min_depth']
        max_depth = config['design_state_data']['algorithms']['RandomForestRegressor']['max_depth']
        min_samples_per_leaf_min_value = config['design_state_data']['algorithms']['RandomForestRegressor']['min_samples_per_leaf_min_value']
        min_samples_per_leaf_max_value = config['design_state_data']['algorithms']['RandomForestRegressor']['min_samples_per_leaf_max_value']
        
        params['rf__n_estimators'] = [min_trees, max_trees]
        params['rf__max_depth'] = [min_depth, max_depth]
        params['rf__min_samples_leaf'] = [min_samples_per_leaf_min_value, min_samples_per_leaf_max_value]
        
        models.append(('RandomForestRegressor', rf, params))
    
    if config['design_state_data']['algorithms']['GBTRegressor']['is_selected']:
        gbr = Pipeline(steps=[('base', base_pipeline), ('gbr', GradientBoostingRegressor())])
        num_boosting_stages = config['design_state_data']['algorithms']['GBTRegressor']['num_of_BoostingStages']
        min_depth = config['design_state_data']['algorithms']['GBTRegressor']['min_depth']
        max_depth = config['design_state_data']['algorithms']['GBTRegressor']['max_depth']
        min_stepsize = config['design_state_data']['algorithms']['GBTRegressor']['min_stepsize']
        max_stepsize = config['design_state_data']['algorithms']['GBTRegressor']['max_stepsize']
        min_iter = config['design_state_data']['algorithms']['GBTRegressor']['min_iter']
        max_iter = config['design_state_data']['algorithms']['GBTRegressor']['max_iter']
        
        params['gbr__n_estimators'] = num_boosting_stages
        params['gbr__max_depth'] = [min_depth, max_depth]
        params['gbr__learning_rate'] = [min_stepsize, max_stepsize]
        params['gbr__min_samples_split'] = [min_iter, max_iter]
        
        models.append(('GBTRegressor', gbr, params))

    for name, model, params in models:
        grid_search = GridSearchCV(model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best score for {name}: {grid_search.best_score_}")

if __name__ == "__main__":
    main()
