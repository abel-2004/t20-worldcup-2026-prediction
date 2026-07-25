"""
Data Collection Module for T20 World Cup Prediction
This module handles downloading and organizing cricket data from various sources.
"""

import pandas as pd
import requests
from pathlib import Path
import json


class T20DataCollector:
    """Collect T20 International cricket data from public sources."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir = self.data_dir / 'raw'
        self.raw_dir.mkdir(exist_ok=True)
    
    def download_sample_data(self):
        """
        Download sample T20 data from Kaggle or create synthetic data for learning.
        
        For a real project, you would:
        1. Use Kaggle API: kaggle datasets download -d ...
        2. Scrape from ESPNcricinfo (with permission)
        3. Use cricket APIs
        
        For this tutorial, we'll create instructions for manual download.
        """
        print("📥 Data Collection Guide")
        print("-" * 50)
        print("\n🔍 Recommended Data Sources:")
        print("\n1. Kaggle Datasets:")
        print("   - 'T20 International Cricket Matches' datasets")
        print("   - Search: https://www.kaggle.com/search?q=t20+cricket")
        print("   - Download and place CSV files in 'data/raw/' folder")
        
        print("\n2. Manual Data Collection:")
        print("   - ESPNcricinfo Statsguru: https://stats.espncricinfo.com/")
        print("   - Filter for T20 Internationals")
        print("   - Export as CSV")
        
        print("\n3. Expected Data Files:")
        print("   - t20_matches.csv (match results)")
        print("   - player_stats.csv (batting/bowling stats)")
        print("   - team_rankings.csv (ICC rankings over time)")
        
        print("\n✅ Once downloaded, place files in: data/raw/")
        print("-" * 50)
        
        # Create a sample data structure for demonstration
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration purposes."""
        
        # Sample match data
        sample_matches = pd.DataFrame({
            'match_id': range(1, 101),
            'date': pd.date_range('2020-01-01', periods=100, freq='W'),
            'team1': ['India', 'Australia', 'England', 'Pakistan', 'South Africa'] * 20,
            'team2': ['Australia', 'England', 'Pakistan', 'South Africa', 'New Zealand'] * 20,
            'winner': ['India', 'Australia', 'England', 'Pakistan', 'South Africa'] * 20,
            'venue': ['Mumbai', 'Sydney', 'London', 'Dubai', 'Cape Town'] * 20,
            'toss_winner': ['India', 'England', 'England', 'Pakistan', 'New Zealand'] * 20,
            'toss_decision': ['bat', 'field', 'bat', 'field', 'bat'] * 20,
            'team1_score': [180, 165, 190, 175, 155] * 20,
            'team2_score': [175, 160, 185, 170, 150] * 20,
        })
        
        # Sample player stats
        sample_players = pd.DataFrame({
            'player_name': ['Virat Kohli', 'Babar Azam', 'Jos Buttler', 'David Warner', 
                           'Rashid Khan', 'Jasprit Bumrah', 'Mitchell Starc', 'Kagiso Rabada'],
            'team': ['India', 'Pakistan', 'England', 'Australia', 
                    'Afghanistan', 'India', 'Australia', 'South Africa'],
            'role': ['Batsman', 'Batsman', 'Batsman', 'Batsman',
                    'Bowler', 'Bowler', 'Bowler', 'Bowler'],
            'matches': [100, 95, 90, 98, 85, 80, 88, 75],
            'batting_avg': [52.5, 48.2, 35.8, 45.6, 18.5, 12.3, 15.2, 10.5],
            'strike_rate': [138.5, 128.3, 145.2, 142.8, 145.0, 105.5, 110.2, 95.5],
            'bowling_avg': [None, None, None, None, 18.5, 19.2, 21.5, 20.8],
            'economy': [None, None, None, None, 6.8, 7.2, 7.8, 7.5],
        })
        
        # Sample team rankings
        sample_rankings = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50, freq='M'),
            'team': ['India'] * 10 + ['England'] * 10 + ['Pakistan'] * 10 + ['Australia'] * 10 + ['New Zealand'] * 10,
            'ranking': [1, 1, 1, 2, 2, 1, 1, 1, 2, 2] * 5,
            'rating': [270, 275, 272, 268, 265, 271, 273, 269, 267, 270] * 5,
        })
        
        # Save sample data
        sample_matches.to_csv(self.raw_dir / 't20_matches_sample.csv', index=False)
        sample_players.to_csv(self.raw_dir / 'player_stats_sample.csv', index=False)
        sample_rankings.to_csv(self.raw_dir / 'team_rankings_sample.csv', index=False)
        
        print("\n✅ Sample data created in data/raw/")
        print("   - t20_matches_sample.csv")
        print("   - player_stats_sample.csv")
        print("   - team_rankings_sample.csv")
    
    def load_data(self):
        """Load all available data files."""
        try:
            matches = pd.read_csv(self.raw_dir / 't20_matches_sample.csv')
            players = pd.read_csv(self.raw_dir / 'player_stats_sample.csv')
            rankings = pd.read_csv(self.raw_dir / 'team_rankings_sample.csv')
            
            print(f"\n📊 Data Loaded Successfully!")
            print(f"   Matches: {len(matches)} rows")
            print(f"   Players: {len(players)} rows")
            print(f"   Rankings: {len(rankings)} rows")
            
            return matches, players, rankings
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Run download_sample_data() first!")
            return None, None, None


if __name__ == "__main__":
    # Example usage
    collector = T20DataCollector()
    collector.download_sample_data()
    matches, players, rankings = collector.load_data()
    
    if matches is not None:
        print("\n" + "="*50)
        print("Sample Match Data:")
        print(matches.head())
