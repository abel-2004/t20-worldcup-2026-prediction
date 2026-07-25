"""
Improved Model Training with Higher Accuracy
This version includes hyperparameter tuning, more models, and better preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not installed. Install with: pip install lightgbm")


class ImprovedT20ModelTrainer:
    """Improved trainer with hyperparameter tuning for better accuracy."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_accuracy = 0
    
    def prepare_training_data(self, features_df, target_col='team1_wins'):
        """Enhanced data preparation with feature selection."""
        if target_col not in features_df.columns:
            print(f"⚠️  Target column '{target_col}' not found.")
            return None, None, None, None, None
        
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
        
        X = X.fillna(X.median())
        
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        X = X.drop(columns=to_drop)
        
        print(f"📊 Feature Selection:")
        print(f"   Original features: {len(numeric_cols)}")
        print(f"   After removing low variance: {X.shape[1]}")
        print(f"   Removed {len(to_drop)} highly correlated features")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        print(f"\n✅ Data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Final features: {X_train.shape[1]}")
        print(f"   Class balance: {dict(y_train.value_counts())}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def train_tuned_random_forest(self, X_train, y_train, X_test, y_test, feature_names):
        """Random Forest with hyperparameter tuning."""
        print("\n" + "="*70)
        print("🎯 Training Tuned Random Forest (with GridSearch)")
        print("="*70)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("\n🔍 Searching for best hyperparameters...")
        print("   This may take a few minutes...")
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        print(f"\n✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Best CV Score: {grid_search.best_score_:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.models['tuned_random_forest'] = best_model
        self.feature_importance['tuned_random_forest'] = feature_importance
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = best_model
        
        return best_model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, feature_names):
        """XGBoost with tuned parameters."""
        if not XGBOOST_AVAILABLE:
            print("\n⚠️  XGBoost not available. Skipping...")
            return None, 0
        
        print("\n" + "="*70)
        print("🚀 Training XGBoost")
        print("="*70)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = feature_importance
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
        
        return model, accuracy
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, feature_names):
        """LightGBM with tuned parameters."""
        if not LIGHTGBM_AVAILABLE:
            print("\n⚠️  LightGBM not available. Skipping...")
            return None, 0
        
        print("\n" + "="*70)
        print("⚡ Training LightGBM")
        print("="*70)
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = feature_importance
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model
        
        return model, accuracy
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Create an ensemble of best models."""
        print("\n" + "="*70)
        print("🎭 Training Ensemble Model")
        print("="*70)
        
        estimators = []
        
        if 'tuned_random_forest' in self.models:
            estimators.append(('rf', self.models['tuned_random_forest']))
        
        if 'xgboost' in self.models:
            estimators.append(('xgb', self.models['xgboost']))
        
        if 'lightgbm' in self.models:
            estimators.append(('lgb', self.models['lightgbm']))
        
        if len(estimators) < 2:
            print("⚠️  Not enough models for ensemble. Need at least 2.")
            return None, 0
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Ensemble Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Combined {len(estimators)} models: {[name for name, _ in estimators]}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.models['ensemble'] = ensemble
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = ensemble
        
        return ensemble, accuracy
    
    def train_all_improved_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Train all improved models and compare."""
        print("\n" + "🚀"*35)
        print("TRAINING ALL IMPROVED MODELS FOR MAXIMUM ACCURACY")
        print("🚀"*35)
        
        results = {}
        
        _, rf_acc = self.train_tuned_random_forest(X_train, y_train, X_test, y_test, feature_names)
        results['Tuned Random Forest'] = rf_acc
        
        _, xgb_acc = self.train_xgboost(X_train, y_train, X_test, y_test, feature_names)
        if xgb_acc > 0:
            results['XGBoost'] = xgb_acc
        
        _, lgb_acc = self.train_lightgbm(X_train, y_train, X_test, y_test, feature_names)
        if lgb_acc > 0:
            results['LightGBM'] = lgb_acc
        
        _, ens_acc = self.train_ensemble(X_train, y_train, X_test, y_test)
        if ens_acc > 0:
            results['Ensemble'] = ens_acc
        
        print("\n" + "="*70)
        print("🏆 MODEL ACCURACY COMPARISON")
        print("="*70)
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        results_df['Accuracy %'] = results_df['Accuracy'] * 100
        print(results_df.to_string(index=False))
        
        best_model_name = results_df.iloc[0]['Model']
        best_acc = results_df.iloc[0]['Accuracy']
        
        print(f"\n🥇 BEST MODEL: {best_model_name}")
        print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
        return results
    
    def save_models(self):
        """Save all trained models."""
        for name, model in self.models.items():
            model_path = self.models_dir / f'{name}.pkl'
            joblib.dump(model, model_path)
            print(f"✅ Saved {name}")
        
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Saved scaler")
        
        if self.best_model:
            best_path = self.models_dir / 'best_model.pkl'
            joblib.dump(self.best_model, best_path)
            print(f"✅ Saved best model (accuracy: {self.best_accuracy:.4f})")
    
    def predict_match(self, model, scaler, features_dict):
        """Predict match outcome."""
        features_df = pd.DataFrame([features_dict])
        
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols]
        
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability


if __name__ == "__main__":
    print("Improved Model Trainer - Ready to boost your accuracy!")
    print("\nThis version includes:")
    print("  ✅ Hyperparameter tuning with GridSearch")
    print("  ✅ XGBoost (if installed)")
    print("  ✅ LightGBM (if installed)")
    print("  ✅ Ensemble methods")
    print("  ✅ Better feature selection")
    print("  ✅ Robust scaling")
