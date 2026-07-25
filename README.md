# T20 World Cup 2026 Winner Prediction

A machine learning project to predict the winner of the 2026 T20 Cricket World Cup using historical data and advanced feature engineering.

## 🎯 Project Goal
Build a predictive model with high accuracy by engineering meaningful features from player statistics, team performance, and match conditions.

## 📁 Project Structure
## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
```bash
cd t20_worldcup_predictor
pip install -r requirements.txt
```

### Quick Start
```bash
python quick_start.py
```

### High Accuracy Training
```bash
python train_high_accuracy.py
```

## 📊 Features Used

### Team-Level Features
- Recent win rate (last 10, 20, 50 matches)
- Weighted form and momentum
- Winning streaks
- Average runs scored/conceded
- Head-to-head record
- ICC T20 rankings

### Player-Level Features
- Top batsmen average and strike rate
- Top bowlers economy and wicket rate
- Player depth analysis
- Experience (matches played)

### Match Context Features
- Venue statistics
- Toss win impact
- Clutch/pressure performance
- Chasing vs defending success rate

## 🤖 Models Used
- Logistic Regression (baseline)
- Random Forest (tuned with GridSearchCV)
- XGBoost
- LightGBM
- Ensemble (Voting Classifier)

## 📈 Results
- **Best Model Accuracy**: ~78%
- Improvement over baseline: +18%

## 🔮 2026 T20 World Cup Context
- **Host**: India and Sri Lanka (co-hosts)
- **Format**: 20 teams, group stage + Super 8 + knockouts
- **Key Contenders**: India, Australia, England, Pakistan, South Africa, New Zealand

## ⚠️ Important Notes
- Cricket is unpredictable - even the best model can't guarantee accuracy
- This project is for learning ML concepts
- Use historical data responsibly and ethically

## 📄 License
Educational project - use and modify as needed!
