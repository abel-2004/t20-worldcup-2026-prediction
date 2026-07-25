"""
Enhanced Feature Engineering for Better Accuracy
Adds more sophisticated features based on cricket domain knowledge
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class EnhancedT20FeatureEngineer:
    """Enhanced feature engineering with more sophisticated features."""
    
    def __init__(self):
        self.team_features = None
        self.player_features = None
    
    def create_weighted_form_features(self, matches_df, team_name, n_matches=10):
        """
        Calculate weighted team form (recent matches weighted more).
        """
        team_matches = matches_df[
            (matches_df['team1'] == team_name) | 
            (matches_df['team2'] == team_name)
        ].sort_values('date', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return self._default_form_features()
        
        # Exponential weights (recent matches count more)
        weights = np.exp(-np.arange(len(team_matches)) * 0.1)
        weights = weights / weights.sum()
        
        # Calculate weighted win rate
        wins = (team_matches['winner'] == team_name).astype(int).values
        weighted_win_rate = np.sum(wins * weights)
        
        # Recent momentum (last 3 vs previous 7)
        if len(team_matches) >= 10:
            last_3_wins = wins[:3].sum() / 3
            prev_7_wins = wins[3:10].sum() / 7
            momentum = last_3_wins - prev_7_wins
        else:
            momentum = 0
        
        # Winning streak
        current_streak = 0
        for match in team_matches.iterrows():
            if match[1]['winner'] == team_name:
                current_streak += 1
            else:
                break
        
        # Calculate scores with exponential weighting
        team1_matches = team_matches[team_matches['team1'] == team_name]
        team2_matches = team_matches[team_matches['team2'] == team_name]
        
        scores = []
        conceded = []
        
        for idx, match in team1_matches.iterrows():
            scores.append(match['team1_score'])
            conceded.append(match['team2_score'])
        
        for idx, match in team2_matches.iterrows():
            scores.append(match['team2_score'])
            conceded.append(match['team1_score'])
        
        avg_score = np.mean(scores) if scores else 150
        avg_conceded = np.mean(conceded) if conceded else 150
        
        # Score consistency (lower std = more consistent)
        score_consistency = 1 / (np.std(scores) + 1) if len(scores) > 1 else 0.5
        
        features = {
            f'weighted_win_rate_{n_matches}': weighted_win_rate,
            f'win_rate_{n_matches}': len(team_matches[team_matches['winner'] == team_name]) / len(team_matches),
            f'momentum_{n_matches}': momentum,
            f'winning_streak': current_streak,
            f'avg_score_{n_matches}': avg_score,
            f'avg_conceded_{n_matches}': avg_conceded,
            f'score_consistency': score_consistency,
            f'run_diff_{n_matches}': avg_score - avg_conceded
        }
        
        return features
    
    def create_advanced_h2h_features(self, matches_df, team1, team2):
        """Enhanced head-to-head with recent emphasis."""
        h2h_matches = matches_df[
            ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
            ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
        ].sort_values('date', ascending=False)
        
        if len(h2h_matches) == 0:
            return {
                'h2h_total': 0,
                'h2h_win_rate': 0.5,
                'h2h_recent_win_rate': 0.5,
                'h2h_avg_margin': 0,
                'h2h_dominance': 0
            }
        
        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        overall_win_rate = team1_wins / len(h2h_matches)
        
        recent_h2h = h2h_matches.head(5)
        recent_wins = len(recent_h2h[recent_h2h['winner'] == team1])
        recent_win_rate = recent_wins / len(recent_h2h) if len(recent_h2h) > 0 else 0.5
        
        margins = []
        for _, match in h2h_matches.iterrows():
            if match['winner'] == team1:
                margin = abs(match['team1_score'] - match['team2_score'])
                margins.append(margin)
        avg_margin = np.mean(margins) if margins else 0
        
        dominance = overall_win_rate * (1 + avg_margin / 50)
        
        features = {
            'h2h_total': len(h2h_matches),
            'h2h_win_rate': overall_win_rate,
            'h2h_recent_win_rate': recent_win_rate,
            'h2h_avg_margin': avg_margin,
            'h2h_dominance': dominance
        }
        
        return features
    
    def create_player_depth_features(self, players_df, team_name):
        """Advanced player features considering depth."""
        team_players = players_df[players_df['team'] == team_name]
        
        if len(team_players) == 0:
            return self._default_player_features()
        
        batsmen = team_players[team_players['role'] == 'Batsman']
        bowlers = team_players[team_players['role'] == 'Bowler']
        
        if len(batsmen) > 0:
            top_3_bat_avg = batsmen.nlargest(3, 'batting_avg')['batting_avg'].mean()
            middle_order = batsmen.nlargest(6, 'batting_avg').tail(3)['batting_avg'].mean()
            bat_depth_score = (top_3_bat_avg + middle_order) / 2
            
            top_strikers = batsmen.nlargest(3, 'strike_rate')['strike_rate'].mean()
            
            bat_experience = batsmen['matches'].sum()
        else:
            top_3_bat_avg = 30
            bat_depth_score = 30
            top_strikers = 125
            bat_experience = 100
        
        if len(bowlers) > 0:
            top_3_bowl_avg = bowlers.nsmallest(3, 'bowling_avg')['bowling_avg'].mean()
            top_3_economy = bowlers.nsmallest(3, 'economy')['economy'].mean()
            
            bowling_depth = min(len(bowlers) / 5, 1)
            
            bowl_experience = bowlers['matches'].sum()
        else:
            top_3_bowl_avg = 25
            top_3_economy = 7.5
            bowling_depth = 0.5
            bowl_experience = 100
        
        balance_score = (bat_depth_score / 50 + (30 - top_3_bowl_avg) / 30) / 2
        
        features = {
            'top_3_bat_avg': top_3_bat_avg,
            'bat_depth_score': bat_depth_score,
            'top_strikers_sr': top_strikers,
            'bat_experience': bat_experience,
            'top_3_bowl_avg': top_3_bowl_avg,
            'top_3_economy': top_3_economy,
            'bowling_depth': bowling_depth,
            'bowl_experience': bowl_experience,
            'team_balance_score': balance_score,
            'total_team_matches': team_players['matches'].sum()
        }
        
        return features
    
    def create_pressure_performance_features(self, matches_df, team_name):
        """
        How team performs under pressure (close matches).
        """
        team_matches = matches_df[
            (matches_df['team1'] == team_name) | 
            (matches_df['team2'] == team_name)
        ].sort_values('date', ascending=False).head(20)
        
        if len(team_matches) == 0:
            return {
                'close_match_win_rate': 0.5,
                'big_match_composure': 0.5,
                'clutch_factor': 0.5
            }
        
        close_matches = []
        for _, match in team_matches.iterrows():
            margin = abs(match['team1_score'] - match['team2_score'])
            if margin < 20:
                close_matches.append(match)
        
        close_df = pd.DataFrame(close_matches) if close_matches else pd.DataFrame()
        
        if len(close_df) > 0:
            close_wins = len(close_df[close_df['winner'] == team_name])
            close_win_rate = close_wins / len(close_df)
        else:
            close_win_rate = 0.5
        
        chase_matches = team_matches[
            (team_matches['team2'] == team_name) & 
            (team_matches['team2_score'] >= 160)
        ]
        
        if len(chase_matches) > 0:
            chase_wins = len(chase_matches[chase_matches['winner'] == team_name])
            big_match_composure = chase_wins / len(chase_matches)
        else:
            big_match_composure = 0.5
        
        clutch_factor = (close_win_rate * 0.6 + big_match_composure * 0.4)
        
        features = {
            'close_match_win_rate': close_win_rate,
            'big_match_composure': big_match_composure,
            'clutch_factor': clutch_factor
        }
        
        return features
    
    def create_venue_advantage_features(self, matches_df, team_name, venue):
        """Enhanced venue features."""
        venue_matches = matches_df[
            ((matches_df['team1'] == team_name) | (matches_df['team2'] == team_name)) &
            (matches_df['venue'] == venue)
        ]
        
        if len(venue_matches) == 0:
            return {
                'venue_win_rate': 0.5,
                'venue_familiarity': 0,
                'venue_avg_score': 150
            }
        
        wins = len(venue_matches[venue_matches['winner'] == team_name])
        win_rate = wins / len(venue_matches)
        
        familiarity = min(len(venue_matches) / 10, 1.0)
        
        team1_venue = venue_matches[venue_matches['team1'] == team_name]
        team2_venue = venue_matches[venue_matches['team2'] == team_name]
        
        scores = list(team1_venue['team1_score']) + list(team2_venue['team2_score'])
        avg_score = np.mean(scores) if scores else 150
        
        features = {
            'venue_win_rate': win_rate,
            'venue_familiarity': familiarity,
            'venue_avg_score': avg_score,
            'venue_matches_played': len(venue_matches)
        }
        
        return features
    
    def build_enhanced_match_features(self, matches_df, players_df, rankings_df, team1, team2, venue=None):
        """
        Build comprehensive feature set with ALL enhancements.
        """
        features = {}
        
        print(f"Building enhanced features for {team1} vs {team2}...")
        
        team1_weighted_form = self.create_weighted_form_features(matches_df, team1, n_matches=10)
        team1_long_form = self.create_weighted_form_features(matches_df, team1, n_matches=20)
        team1_players = self.create_player_depth_features(players_df, team1)
        team1_pressure = self.create_pressure_performance_features(matches_df, team1)
        
        for key, value in {**team1_weighted_form, **team1_long_form, 
                           **team1_players, **team1_pressure}.items():
            features[f'team1_{key}'] = value
        
        team2_weighted_form = self.create_weighted_form_features(matches_df, team2, n_matches=10)
        team2_long_form = self.create_weighted_form_features(matches_df, team2, n_matches=20)
        team2_players = self.create_player_depth_features(players_df, team2)
        team2_pressure = self.create_pressure_performance_features(matches_df, team2)
        
        for key, value in {**team2_weighted_form, **team2_long_form,
                           **team2_players, **team2_pressure}.items():
            features[f'team2_{key}'] = value
        
        h2h = self.create_advanced_h2h_features(matches_df, team1, team2)
        features.update(h2h)
        
        if venue:
            team1_venue = self.create_venue_advantage_features(matches_df, team1, venue)
            team2_venue = self.create_venue_advantage_features(matches_df, team2, venue)
            for key, value in team1_venue.items():
                features[f'team1_{key}'] = value
            for key, value in team2_venue.items():
                features[f'team2_{key}'] = value
        
        features['form_gap'] = features['team1_weighted_win_rate_10'] - features['team2_weighted_win_rate_10']
        features['momentum_advantage'] = features['team1_momentum_10'] - features['team2_momentum_10']
        features['batting_advantage'] = features['team1_bat_depth_score'] - features['team2_bat_depth_score']
        features['bowling_advantage'] = features['team2_top_3_bowl_avg'] - features['team1_top_3_bowl_avg']
        features['pressure_advantage'] = features['team1_clutch_factor'] - features['team2_clutch_factor']
        features['experience_gap'] = features['team1_total_team_matches'] - features['team2_total_team_matches']
        
        features['form_x_pressure'] = features['team1_weighted_win_rate_10'] * features['team1_clutch_factor']
        features['batting_x_bowling'] = features['team1_bat_depth_score'] * (30 - features['team1_top_3_bowl_avg'])
        
        return features
    
    def _default_form_features(self):
        return {
            'weighted_win_rate_10': 0.5,
            'win_rate_10': 0.5,
            'momentum_10': 0,
            'winning_streak': 0,
            'avg_score_10': 150,
            'avg_conceded_10': 150,
            'score_consistency': 0.5,
            'run_diff_10': 0
        }
    
    def _default_player_features(self):
        return {
            'top_3_bat_avg': 30,
            'bat_depth_score': 30,
            'top_strikers_sr': 125,
            'bat_experience': 100,
            'top_3_bowl_avg': 25,
            'top_3_economy': 7.5,
            'bowling_depth': 0.5,
            'bowl_experience': 100,
            'team_balance_score': 0.5,
            'total_team_matches': 500
        }


if __name__ == "__main__":
    print("Enhanced Feature Engineering Module - For Maximum Accuracy!")
    print("\nNew features added:")
    print("  ✅ Weighted form (recent matches count more)")
    print("  ✅ Momentum tracking")
    print("  ✅ Winning streaks")
    print("  ✅ Pressure performance (clutch factor)")
    print("  ✅ Player depth analysis")
    print("  ✅ Advanced head-to-head")
    print("  ✅ Interaction features")
