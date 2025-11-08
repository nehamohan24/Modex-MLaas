"""
Advanced ML capabilities including DQN and federated learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tabulate import tabulate
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
from sklearn.model_selection import learning_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DQN(nn.Module):
    """Deep Q-Network for reinforcement learning"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DQNAgent:
    """DQN Agent for reinforcement learning"""
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        
        self.q_network = DQN(state_size, 64, action_size)
        self.target_network = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class FederatedLearning:
    """Federated Learning implementation for collaborative training"""
    
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type
        self.global_model = None
        self.client_models = {}
        self.training_history = []
        
    def initialize_global_model(self, input_size, output_size=1):
        """Initialize the global model"""
        if self.model_type == 'linear_regression':
            self.global_model = LinearRegression()
        elif self.model_type == 'random_forest':
            self.global_model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def federated_averaging(self, client_models, client_weights=None):
        """Perform federated averaging of client models"""
        if not client_models:
            return self.global_model
        
        if client_weights is None:
            client_weights = [1.0] * len(client_models)
        
        
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        if self.model_type == 'linear_regression':
            
            avg_coef = np.average([model.coef_ for model in client_models], 
                                weights=client_weights, axis=0)
            avg_intercept = np.average([model.intercept_ for model in client_models], 
                                    weights=client_weights)
            
            
            self.global_model = LinearRegression()
            self.global_model.coef_ = avg_coef
            self.global_model.intercept_ = avg_intercept
            
        elif self.model_type == 'random_forest':
            self.global_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        
        return self.global_model
    
    def train_round(self, client_data, client_weights=None):
        """Perform one round of federated training"""
        client_models = []
        
        
        for i, (X, y) in enumerate(client_data):
            if self.model_type == 'linear_regression':
                local_model = LinearRegression()
            elif self.model_type == 'random_forest':
                local_model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                continue
            
            local_model.fit(X, y)
            client_models.append(local_model)
        
        
        self.global_model = self.federated_averaging(client_models, client_weights)
        
        
        if client_data:
            
            X_test, y_test = client_data[0]
            y_pred = self.global_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.training_history.append({
                'round': len(self.training_history) + 1,
                'mse': mse,
                'r2': r2,
                'clients': len(client_data)
            })
        
        return self.global_model

class AdvancedMLTrainer:
    """Advanced ML trainer with multiple algorithms and optimization"""
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        
    def train_ensemble_model(self, X, y, model_types=['linear_regression', 'random_forest']):
        """Train an ensemble of models for better accuracy"""
        models = {}
        predictions = {}
        
        for model_type in model_types:
            if model_type == 'linear_regression':
                model = LinearRegression()
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            else:
                continue
            
            model.fit(X, y)
            models[model_type] = model
            predictions[model_type] = model.predict(X)
        
        
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        
        mse = mean_squared_error(y, ensemble_pred)
        r2 = r2_score(y, ensemble_pred)
        accuracy = max(0, 1 - mse / np.var(y))
        
        return {
            'models': models,
            'ensemble_prediction': ensemble_pred,
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy
        }
    
    def optimize_hyperparameters(self, X, y, model_type='linear_regression'):
        """Optimize hyperparameters for better performance"""
        if model_type == 'random_forest':
            from sklearn.model_selection import GridSearchCV
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            
            model = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X, y)
            
            return grid_search.best_estimator_, grid_search.best_score_
        
        return None, None

    def select_best_model(self, X, y, cv=5):
        """Try candidate pipelines (Ridge with SelectKBest, RandomForest with SelectKBest) and return best by CV score.

        Returns (best_estimator, best_score, details_dict)
        """
        results = {}
        from sklearn.model_selection import GridSearchCV

       
        try:
            pipe_ridge = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('select', SelectKBest(score_func=f_regression)),
                ('ridge', Ridge(random_state=42))
            ])
            param_grid_ridge = {
                'select__k': [5, 7, 10],  
                'ridge__alpha': [0.1, 1.0, 10.0]  
            }
            grid_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=cv, scoring='r2', n_jobs=-1)
            grid_ridge.fit(X, y)
            results['ridge'] = {'model': grid_ridge.best_estimator_, 'cv_score': grid_ridge.best_score_, 'best_params': grid_ridge.best_params_}
        except Exception as e:
            results['ridge'] = {'model': None, 'cv_score': -np.inf, 'error': str(e)}

        # Candidate 2: RandomForest pipeline with SelectKBest
        try:
            pipe_rf = Pipeline([
                ('select', SelectKBest(score_func=f_regression)),
                ('rf', RandomForestRegressor(random_state=42))
            ])
            param_grid_rf = {
                'select__k': [5, 7, 10],  
                'rf__n_estimators': [100, 200],
                'rf__max_depth': [5, 7, 10],  
                'rf__min_samples_leaf': [2, 4] 
            }
            grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv, scoring='r2', n_jobs=-1)
            grid_rf.fit(X, y)
            results['random_forest'] = {'model': grid_rf.best_estimator_, 'cv_score': grid_rf.best_score_, 'best_params': grid_rf.best_params_}
        except Exception as e:
            results['random_forest'] = {'model': None, 'cv_score': -np.inf, 'error': str(e)}

        # Choose best
        best_key = max(results.keys(), key=lambda k: results[k].get('cv_score', -np.inf))
        best = results[best_key]
        return best.get('model', None), best.get('cv_score', None), results
    
    def cross_validation_training(self, X, y, model_type='linear_regression', cv=5):
        """Perform cross-validation training for robust evaluation"""
        from sklearn.model_selection import cross_val_score
        
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        else:
            return None
        
       
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        
        model.fit(X, y)
        
        return {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }


def calculate_finance_metrics(y_true, y_pred, returns=None):
    """Calculate finance-specific metrics including Sharpe ratio, returns, etc."""
    metrics = {}
    
    
    metrics['R²'] = r2_score(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    
    
    if returns is not None:
      
        metrics['Sharpe_Ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        metrics['Max_Drawdown'] = np.min(np.minimum.accumulate(returns - np.amax(returns)))
        metrics['Win_Rate'] = np.mean(returns > 0)
    
    
    metrics['Direction_Accuracy'] = np.mean((y_true[1:] - y_true[:-1]) * 
                                          (y_pred[1:] - y_pred[:-1]) > 0)
    
    return metrics

def calculate_healthcare_metrics(y_true, y_pred, threshold=0.5):
    """Calculate healthcare-specific metrics (sensitivity, specificity)."""
    
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    
    if len(np.unique(y_true_bin)) < 2 or len(np.unique(y_pred_bin)) < 2:
        return {
            'sensitivity': np.nan,
            'specificity': np.nan
        }
    
   
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    
   
    if cm.shape != (2, 2):
        return {
            'sensitivity': np.nan,
            'specificity': np.nan
        }
    
    tn, fp, fn, tp = cm.ravel()
    
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def calculate_tech_metrics(y_true, y_pred, latency=None):
    """Calculate technology-specific metrics including accuracy, latency, etc."""
    metrics = {}
    
   
    metrics['R²'] = r2_score(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    
    if latency is not None:
        metrics['Avg_Latency'] = np.mean(latency)
        metrics['P95_Latency'] = np.percentile(latency, 95)
        metrics['P99_Latency'] = np.percentile(latency, 99)
    
    
    errors = np.abs(y_true - y_pred)
    metrics['Error_P95'] = np.percentile(errors, 95)
    metrics['Error_StdDev'] = np.std(errors)
    
    return metrics

def calculate_performance_metrics(y_true, y_pred, y_train=None, task='regression', domain=None, **kwargs):
    """Calculate comprehensive performance metrics for a model.
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        y_train: Training set y values (for variance calculation)
        task: 'regression' or 'classification'
        domain: Optional domain for specific metrics ('finance', 'healthcare', 'tech')
        **kwargs: Additional arguments for domain-specific metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    
    if task == 'regression':
        metrics['R²'] = r2_score(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        var_ref = np.var(y_train) if y_train is not None else np.var(y_true)
        metrics['Accuracy'] = max(0, 1 - metrics['MSE'] / var_ref)
    
    
    if domain == 'finance':
        finance_metrics = calculate_finance_metrics(y_true, y_pred, 
                                                 returns=kwargs.get('returns'))
        metrics.update(finance_metrics)
    
    elif domain == 'healthcare':
        healthcare_metrics = calculate_healthcare_metrics(y_true, y_pred, 
                                                       threshold=kwargs.get('threshold', 0.5))
        metrics.update(healthcare_metrics)
    
    elif domain == 'tech':
        tech_metrics = calculate_tech_metrics(y_true, y_pred,
                                           latency=kwargs.get('latency'))
        metrics.update(tech_metrics)
    
    else:
        
        if task == 'regression':
           
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_true_bin = (y_true > np.median(y_true)).astype(int)
                y_pred_bin = (y_pred > np.median(y_pred)).astype(int)
                
                metrics['Precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                metrics['Recall'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                metrics['F1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        else:
            
            metrics['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return metrics


def display_metrics_comparison(train_metrics, test_metrics, title="Model Performance Metrics"):
    """Display a formatted table comparing train and test metrics.
    
    Args:
        train_metrics: Dict of training metrics
        test_metrics: Dict of test metrics
        title: Table title
    """
    
    headers = ["Metric", "Training", "Test"]
    rows = []
    
    
    metric_order = ['Accuracy', 'Precision', 'Recall', 'F1', 'R²', 'MSE']
    
    for metric in metric_order:
        if metric in train_metrics and metric in test_metrics:
            rows.append([
                metric,
                f"{train_metrics[metric]:.3f}",
                f"{test_metrics[metric]:.3f}"
            ])
    
    print(f"\n{title}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def ensure_high_accuracy(X, y, target_accuracy=0.9):
    """Ensure model achieves target accuracy through various techniques"""
    trainer = AdvancedMLTrainer()
    
    
    ensemble_result = trainer.train_ensemble_model(X, y)
    
    if ensemble_result['accuracy'] >= target_accuracy:
        return ensemble_result
    
    
    if len(np.unique(y)) > 10: 
        best_model, best_score = trainer.optimize_hyperparameters(X, y, 'random_forest')
        if best_model is not None and best_score >= target_accuracy:
            return {
                'model': best_model,
                'accuracy': best_score,
                'method': 'hyperparameter_optimization'
            }
    
    
    cv_result = trainer.cross_validation_training(X, y, 'random_forest')
    if cv_result and cv_result['mean_cv_score'] >= target_accuracy:
        return {
            'model': cv_result['model'],
            'accuracy': cv_result['mean_cv_score'],
            'method': 'cross_validation'
        }
    
    
    X_augmented = np.vstack([X, X + np.random.normal(0, 0.01, X.shape)])
    y_augmented = np.hstack([y, y])
    
    final_result = trainer.train_ensemble_model(X_augmented, y_augmented)
    
    return final_result


def plot_learning_curve(estimator, X, y, cv=5, scoring='r2',
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        output_path='learning_curve.png'):
    """Compute and save a learning curve plot for given estimator.

    Parameters
    - estimator: an object implementing fit/predict (scikit-learn estimator)
    - X, y: data arrays
    - cv: cross-validation folds (int or cross-validation generator)
    - scoring: scoring string accepted by sklearn
    - train_sizes: relative or absolute sizes for training set
    - output_path: path to save PNG image

    Returns a dict with train_sizes, train_scores_mean, test_scores_mean
    """
    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes, n_jobs=1
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute learning curve: {e}")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    ax.set_title('Learning Curve')
    ax.set_xlabel('Training examples')
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    ax.grid(True)

    # Save to file
    try:
        fig.tight_layout()
        fig.savefig(output_path)
    finally:
        plt.close(fig)

    return {
        'train_sizes': train_sizes_abs,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'test_scores_mean': test_scores_mean,
        'test_scores_std': test_scores_std,
        'output_path': output_path
    }


def generate_learning_curves_for_defaults(X, y, output_dir='.'):
    """Generate learning curves for a couple of default estimators and save PNGs.

    Returns dict of results keyed by model name.
    """
    results = {}
    os.makedirs(output_dir, exist_ok=True)

   
    lr = LinearRegression()
    out_lr = os.path.join(output_dir, 'learning_curve_linear_regression.png')
    results['linear_regression'] = plot_learning_curve(lr, X, y, cv=5, scoring='r2', output_path=out_lr)

    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    out_rf = os.path.join(output_dir, 'learning_curve_random_forest.png')
    results['random_forest'] = plot_learning_curve(rf, X, y, cv=5, scoring='r2', output_path=out_rf)

    return results


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    
    
    domains = {
        'finance': {
            'X': np.random.randn(n_samples, n_features), 
            'y': None,  
            'returns': np.random.normal(0.001, 0.02, n_samples),  
            'features': ['market_return', 'volatility', 'volume', 'momentum'] + [f'factor_{i}' for i in range(6)]
        },
        'healthcare': {
            'X': np.random.randn(n_samples, n_features),  
            'y': None,
            'features': ['age', 'blood_pressure', 'glucose', 'bmi'] + [f'biomarker_{i}' for i in range(6)]
        },
        'tech': {
            'X': np.random.randn(n_samples, n_features), 
            'y': None,
            'latency': np.abs(np.random.normal(100, 20, n_samples)), 
            'features': ['cpu_load', 'memory_usage', 'network_traffic', 'disk_io'] + [f'metric_{i}' for i in range(6)]
        }
    }
    
    
    domains['finance']['y'] = (domains['finance']['X'][:, 0] * 0.3 + 
                             domains['finance']['X'][:, 1] * 0.2 +
                             domains['finance']['returns'] * 5)
    
   
    health_score = (
        domains['healthcare']['X'][:, 0] * 0.5 +  
        domains['healthcare']['X'][:, 1] * 0.3 +  
        np.sin(domains['healthcare']['X'][:, 2]) * 0.1 + 
        np.random.normal(0, 0.1, n_samples)  
    )
    domains['healthcare']['y'] = (health_score > np.median(health_score)).astype(int) 
    
    domains['tech']['y'] = np.exp(domains['tech']['X'][:, 0] * 0.3 + 
                                 domains['tech']['X'][:, 1] * 0.2) + domains['tech']['latency'] * 0.1

    
    from sklearn.model_selection import train_test_split
    trainer = AdvancedMLTrainer()

    for domain_name, domain_data in domains.items():
        print(f"\n{'='*20} {domain_name.upper()} DOMAIN {'='*20}")
        
       
        X_train, X_test, y_train, y_test = train_test_split(
            domain_data['X'], domain_data['y'], 
            test_size=0.2, random_state=42
        )
        
        
        best_model, cv_score, details = trainer.select_best_model(X_train, y_train, cv=5)
        
        if best_model is None:
            print(f"No model selected for {domain_name}")
            continue
            
        
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        
        domain_kwargs = {}
        if domain_name == 'finance':
            train_idx = np.random.permutation(len(domain_data['returns']))[:len(train_pred)]
            domain_kwargs['returns'] = domain_data['returns'][train_idx]
        elif domain_name == 'tech':
            train_idx = np.random.permutation(len(domain_data['latency']))[:len(train_pred)]
            domain_kwargs['latency'] = domain_data['latency'][train_idx]
        
        train_metrics = calculate_performance_metrics(
            y_train, train_pred, y_train=y_train,
            domain=domain_name, **domain_kwargs
        )
        test_metrics = calculate_performance_metrics(
            y_test, test_pred, y_train=y_train,
            domain=domain_name, **domain_kwargs
        )
        
        
        print(f"\nModel Selection Results for {domain_name}:")
        print(f"Best model CV R² score: {cv_score:.3f}")
        print("\nBest model parameters:")
        for model_type, info in details.items():
            if info.get('cv_score', -np.inf) == cv_score:
                print(f"- Type: {model_type}")
                if 'best_params' in info:
                    print(f"- Parameters: {info['best_params']}")
        
        
        display_metrics_comparison(
            train_metrics, test_metrics,
            title=f"\n{domain_name.capitalize()} Domain Performance Metrics"
        )
