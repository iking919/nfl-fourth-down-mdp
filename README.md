# Expected Points vs. Win Probability: Sequential Optimization of NFL Fourth-Down Decisions Under Uncertainty

## Overview

This project investigates optimal fourth-down decision-making in the National Football League (NFL) using Markov decision processes (MDPs) and sequential planning methods.

NFL coaches must frequently choose between attempting a fourth-down conversion, attempting a field goal, or punting. The optimal decision depends on several factors, including field position, yards to go, score differential, and time remaining. Although an action may maximize immediate expected points, it may not maximize the team's probability of eventually winning the game.

The goal of this project is to examine how fourth-down recommendations change when decisions are evaluated using different optimization objectives and when future game consequences are explicitly considered.

Historical NFL play-by-play data will be used to estimate outcome probabilities and construct a simplified decision-making model.

## Research Questions

The project focuses on the following questions:

1. How do immediate expected-points decisions differ from sequential decision policies that account for future game states?
2. How do policies that maximize expected scoring outcomes differ from policies that maximize win probability?
3. Under what game conditions do these policies recommend different fourth-down actions?
4. How sensitive are recommended actions to uncertainty in estimated transition probabilities?
5. How do planning depth and simulation count affect decision quality, stability, and computational cost?

## Methodology

### 1. Data Collection and Preprocessing

Historical NFL play-by-play data will be obtained using [nflfastR](https://www.nflfastr.com/) and the associated nflverse data resources.

Relevant variables include:

- Field position
- Yards to go
- Score differential
- Time remaining
- Quarter
- Play type and outcome
- Expected points (EP)
- Win probability (WP)

The dataset will be filtered to identify fourth-down situations and relevant play outcomes. State variables will be discretized to create a computationally manageable state space.

### 2. MDP Formulation

Fourth-down decision-making will be modeled as a Markov decision process.

**State representation**

The state will describe the current game situation using:

- Field position
- Yards to go
- Score differential
- Time remaining
- Quarter and possession context

**Action space**

Available actions will include:

- Go for it: Attempt a fourth-down conversion.
- Field goal: Attempt a field goal when feasible.
- Punt: Punt the football to the opposing team.

**State transitions**

Transition probabilities will be estimated from historical NFL outcomes. Each action can produce different outcomes and subsequent game states, accounting for changes in possession, field position, score, and time.

**Reward functions**

Two optimization objectives will be considered:

- Expected scoring outcomes: Evaluate decisions based on expected points.
- Win probability: Evaluate decisions based on the probability of eventually winning the game.

### 3. Decision Policies

The project will compare three primary decision-making strategies:

**Immediate Expected-Points Policy**

Selects the action with the highest estimated immediate expected-points value without explicitly optimizing future decisions.

**Sequential Expected-Points Policy**

Uses an MDP-based value function to evaluate the expected scoring consequences of decisions over subsequent game states.

**Sequential Win-Probability Policy**

Uses a value function based on the probability of winning the game, accounting for the consequences of current decisions on future opportunities.

### 4. Sequential Planning

Online planning methods will be investigated to evaluate the long-term consequences of candidate actions.

The primary methods include:

- **Forward Search:** Explores possible future states to a limited depth and uses an approximate value function to estimate outcomes beyond the search horizon.
- **Rollout Planning:** Evaluates candidate actions by simulating future trajectories under a continuation policy.

Sampling-based approaches may be used to reduce computational requirements when evaluating large transition spaces.

### 5. Evaluation and Experiments

The resulting policies will be evaluated through simulations using the estimated transition model.

Planned experiments include:

1. **Policy Comparison:** Compare immediate expected points, sequential expected points, and sequential win probability.
2. **Game Situation Analysis:** Examine policy differences across score differentials, field positions, yards to go, and time remaining.
3. **Uncertainty Analysis:** Evaluate how uncertainty in estimated transition probabilities affects recommended actions.
4. **Planning Efficiency:** Investigate how planning depth and simulation count affect decision stability and computational cost.

Held-out historical data will be used to assess model calibration and characterize the situations represented in the evaluation.

## Data Sources

- [nflfastR](https://www.nflfastr.com/) — NFL play-by-play data and analytical tools.
- [nflverse](https://nflverse.com/) — Open-source NFL data, including historical play-by-play datasets.

## Technology Stack

The project is primarily developed in Python.

Planned libraries and tools:

- Python
- pandas
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Jupyter Notebook

R and nflfastR may also be used for data acquisition and preprocessing.

## Repository Structure

```text
nfl-fourth-down-mdp/
│
├── data/
│   ├── raw/                 # Original NFL play-by-play data
│   └── processed/           # Cleaned and transformed datasets
│
├── notebooks/               # Exploratory analysis and experiments
│
├── src/
│   ├── data_loader.py       # Data acquisition and loading
│   ├── preprocessing.py     # Data cleaning and state discretization
│   ├── state.py             # MDP state representation
│   ├── transitions.py       # Transition probability estimation
│   ├── baselines.py         # Baseline decision policies
│   ├── value_function.py    # Expected-points and win-probability values
│   ├── forward_search.py    # Forward search planning
│   ├── rollout.py           # Rollout-based planning
│   └── evaluation.py        # Policy evaluation and comparisons
│
├── experiments/             # Experiment configurations and scripts
├── results/                 # Experimental results and metrics
├── figures/                 # Generated visualizations
│
├── README.md
└── requirements.txt
```

## Project Timeline

| Milestone | Target Date |
|---|---|
| Initial data acquisition and exploratory analysis | November 13, 2026 |
| Progress report | November 17, 2026 |
| Transition model and reward formulation | November 21, 2026 |
| Sequential decision policies | November 27, 2026 |
| Forward search and rollout implementation | November 30, 2026 |
| Experiments and sensitivity analysis | December 1–5, 2026 |
| Final presentation | December 7, 2026 |

## Project Information

**Course:** ECSE 6494  
**Instructor:** Professor Yel  
**Author:** Izaak King  
**Project Start:** November 2026  
**Final Presentation:** December 7, 2026

---

*This project is being developed as part of ECSE 6494. The MDP formulation, transition model, and experimental methodology may be refined as implementation and empirical analysis progress.*
