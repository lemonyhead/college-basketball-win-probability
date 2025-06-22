# Win Probability Modeling in Basketball Games

This repository contains multiple modeling approaches to estimate win probabilities in basketball games using a variety of machine learning techniques and statistical heuristics.

## Project Files

- `Three-Game Moving Average Logistic Regression Win Probability Model.ipynb`:  
  Uses moving averages and logistic regression to predict game outcomes.

- `win_probability_CNN.ipynb`:  
  Applies a Convolutional Neural Network to historical game data to estimate win probabilities based on sequences of plays or events.

- `win_prob_KNN.ipynb`:  
  Implements a K-Nearest Neighbors model using basic game stats for win probability classification.

- `Elo.ipynb`:  
  Reproduces an Elo rating-based system to simulate team strengths and compute pre-game win probabilities.

---

## Goals

- Predict the probability of a team winning a basketball game using historical game data.
- Compare traditional statistical models to deep learning approaches.
- Evaluate trade-offs between interpretability and predictive power.

---

## Models Used

| Model Type             | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Logistic Regression    | Uses 3-game moving averages of game stats for binary classification         |
| KNN                    | Predicts outcome by comparing to most similar past games                    |
| CNN                    | Learns spatial patterns in sequences of inputs (e.g. score/time matrices)   |
| Elo Rating System      | Simulates dynamic team skill over a season based on game outcomes           |

---

## Performance Metrics

Each model was evaluated using:
- Accuracy
- Log-loss or cross-entropy (for probabilistic outputs)
- Confusion matrix (where applicable)

---

## Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow` / `keras`
- `scipy`

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/win-probability-models.git
   cd win-probability-models
