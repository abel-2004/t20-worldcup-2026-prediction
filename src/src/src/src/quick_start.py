#!/usr/bin/env python3
"""
Quick Start Script for T20 World Cup Prediction
Run this to see the entire pipeline in action!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_collection import T20DataCollector
from feature_engineering import T20FeatureEngineer
from model_training import T20ModelTrainer
import pandas as pd
import numpy as np


def main():
    print("="*70)
    print("🏆 T20 WORLD CUP 2026 WINNER PREDICTION")
    print("="*70)
    print("\nThis script will:")
    print("  1. Load cricket data")
    print("  2. Engineer features")
    print("  3. Train ML models")
    print("  4. Predict 2026 World Cup winner")
    print("\n" + "="*70 + "\n")
    
    # Step 1: Data Collection
    print("\n📥 STEP 1: DATA COLLECTION")
    print("-"*70)
    collector = T20DataCollector(data_dir='data')
    collector.download_sample_data()
    matches_df, players_df, rankings_df = collector.load_data()
    
    if matches_df is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Step 2: Feature Engineering
    print("\n\n🔧 STEP 2: FEATURE ENGINEERING")
    print("-"*70)
    print("Building features for historical matches...")
    print("(This creates the 'intelligence' for our model)\n")
    
    engineer = T20FeatureEngineer()
    all_features = []
    
    # Process a subset for demonstration (first 50 matches)
    sample_size = min(50, len(matches_df))
    
    for idx, match in matches_df.head(sample_size).iterrows():
        if idx % 10 == 0:
            print(f"  Processing match {idx+1}/{sample_size}...")
        
        try:
            features = engineer.build_match_features(
                matches_df[:idx] if idx > 0 else matches_df.head(1),
                players_df,
                rankings_df,
                team1=match['team1'],
                team2=match['team2'],
                venue=match['venue']
            )
            features['team1_wins'] = 1 if match['winner'] == match['team1'] else 0
            all_features.append(features)
        except:
            continue
    
    features_df = pd.DataFrame(all_features)
    print(f"\n✅ Created {len(features_df)} training examples with {features_df.shape[1]-1} features each")
    
    # Step 3: Model Training
    print("\n\n🤖 STEP 3: MODEL TRAINING")
    print("-"*70)
    
    trainer = T20ModelTrainer(models_dir='models')
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_training_data(
        features_df, target_col='team1_wins'
    )
    
    if X_train is None:
        print("❌ Failed to prepare training data. Exiting...")
        return
    
    # Train only Random Forest for quick demo
    print("\nTraining Random Forest model...")
    model, accuracy = trainer.train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    
    print(f"\n✅ Model trained with {accuracy*100:.2f}% accuracy")
    
    # Save model
    trainer.save_models()
    
    # Step 4: Predictions
    print("\n\n🔮 STEP 4: 2026 WORLD CUP PREDICTIONS")
    print("-"*70)
    
    top_teams = ['India', 'Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand']
    team_scores = {}
    
    print("\nSimulating matches between top contenders...\n")
    
    for team in top_teams:
        wins = 0
        total_matches = 0
        
        for opponent in top_teams:
            if team != opponent:
                try:
                    features = engineer.build_match_features(
                        matches_df, players_df, rankings_df,
                        team1=team, team2=opponent, venue='Mumbai'
                    )
                    prediction, probability = trainer.predict_match(model, trainer.scaler, features)
                    wins += probability[1]  # Probability of team1 winning
                    total_matches += 1
                except:
                    continue
        
        avg_win_prob = wins / total_matches if total_matches > 0 else 0.5
        team_scores[team] = avg_win_prob
    
    # Sort and display
    sorted_teams = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 PREDICTED STANDINGS:\n")
    for rank, (team, score) in enumerate(sorted_teams, 1):
        bar = '█' * int(score * 40)
        print(f"  {rank}. {team:15} {score*100:5.2f}%  {bar}")
    
    print("\n" + "="*70)
    print(f"\n🏆 PREDICTED WINNER: {sorted_teams[0][0]}")
    print(f"   Win Probability: {sorted_teams[0][1]*100:.2f}%")
    print("\n" + "="*70)
    
    print("\n\n✅ Complete! Check these files:")
    print("   📁 data/processed_features.csv - Your engineered features")
    print("   📁 models/random_forest.pkl - Trained model")
    print("   📁 notebooks/complete_workflow.ipynb - Full interactive version")
    
    print("\n💡 Next steps:")
    print("   1. Open the Jupyter notebook for detailed exploration")
    print("   2. Try adding more features in feature_engineering.py")
    print("   3. Experiment with different models")
    print("   4. Get real data from Kaggle or ESPNcricinfo")
    
    print("\n🎓 Happy learning!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry running: pip install -r requirements.txt")
