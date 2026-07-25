"""
Model Training Module for T20 World Cup Prediction
Train and evaluate multiple machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path


class T20ModelTrainer:
    """Train and evaluate models for T20 match prediction."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
    
    def prepare_training_data(self, features_df, target_col='team1_wins'):
        """
        Prepare features and target for training.
        """
        if target_col not in features_df.columns:
            print(f"⚠️  Target column '{target_col}' not found. Available columns:")
            print(features_df.columns.tolist())
            return None, None, None, None, None
        
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        print(f"✅ Data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Number of features: {X_train.shape[1]}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model (baseline)."""
        print("\n" + "="*60)
        print("Training Logistic Regression (Baseline Model)")
        print("="*60)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\n🔄 Cross-Validation Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models['logistic_regression'] = model
        return model, accuracy
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, feature_names):
        """Train Random Forest model."""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\n🔄 Cross-Validation Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.models['random_forest'] = model
        self.feature_importance['random_forest'] = feature_importance
        return model, accuracy
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, feature_names):
        """Train Gradient Boosting model."""
        print("\n" + "="*60)
        print("Training Gradient Boosting")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"\n🔄 Cross-Validation Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.models['gradient_boosting'] = model
        self.feature_importance['gradient_boosting'] = feature_importance
        return model, accuracy
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Train all models and compare performance."""
        print("\n" + "🤖"*30)
        print("TRAINING ALL MODELS")
        print("🤖"*30)
        
        results = {}
        
        _, lr_acc = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        results['Logistic Regression'] = lr_acc
        
        _, rf_acc = self.train_random_forest(X_train, y_train, X_test, y_test, feature_names)
        results['Random Forest'] = rf_acc
        
        _, gb_acc = self.train_gradient_boosting(X_train, y_train, X_test, y_test, feature_names)
        results['Gradient Boosting'] = gb_acc
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        print(results_df.to_string(index=False))
        
        best_model = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        
        return results
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            model_path = self.models_dir / f'{name}.pkl'
            joblib.dump(model, model_path)
            print(f"✅ Saved {name} to {model_path}")
        
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Saved scaler to {scaler_path}")
        
        for name, importance_df in self.feature_importance.items():
            importance_path = self.models_dir / f'{name}_feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            print(f"✅ Saved feature importance for {name}")
    
    def load_model(self, model_name='random_forest'):
        """Load a trained model from disk."""
        model_path = self.models_dir / f'{model_name}.pkl'
        scaler_path = self.models_dir / 'scaler.pkl'
        
        if not model_path.exists():
            print(f"❌ Model {model_name} not found at {model_path}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        print(f"✅ Loaded {model_name} and scaler")
        return model, scaler
    
    def predict_match(self, model, scaler, features_dict):
        """
        Predict match outcome given features.
        """
        features_df = pd.DataFrame([features_dict])
        
        features_scaled = scaler.transform(features_df)
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability


if __name__ == "__main__":
    print("Model Training Module - Ready to use!")
    print("\nExample usage:")
    print("  trainer = T20ModelTrainer()")
    print("  X_train, X_test, y_train, y_test, features = trainer.prepare_training_data(df)")
    print("  results = trainer.train_all_models(X_train, y_train, X_test, y_test, features)")
