# Machine Learning & Data Analysis - Homework 3

## Overview
This repository contains Homework 3, a Bayesian inference and classifier design assignment. The notebook guides the student through probability modeling, maximum likelihood estimation, and Bayesian classification using several datasets.

## Assignment Subject
The assignment focuses on:
- Bayesian learning principles
- Conditional independence and probability modeling
- Poisson maximum likelihood estimation (MLE)
- Comparing Normal Naive Bayes and Full Bayes classifiers
- Evaluating classifier performance on synthetic and real datasets

## Key Challenges
- Designing distributions that satisfy conditional independence constraints
- Implementing both iterative and analytic MLE solutions for a Poisson model
- Understanding the difference between Naive Bayes and Full Bayes assumptions
- Applying Bayesian decision rules correctly in a classification setting
- Interpreting results across multiple datasets, including Poisson samples, breast datasets, and synthetic animal classes

## Contents
- `hw3.py` — Python implementation of required functions and classifiers
- `hw3.ipynb` — notebook walkthrough with step-by-step instructions, analysis, and plots
- `data/` — datasets used in the assignment (`breast_*.csv`, `poisson_1000_samples.csv`, `randomammal_*.csv`)

## How to use
1. Open `hw3.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the notebook instructions and complete the required functions in `hw3.py`.
3. Run analysis cells, review plots, and compare classification results.
4. Save your final notebook and script for submission.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- SciPy or scikit-learn (if used by the notebook)

## Notes
- The notebook contains the full exercise workflow and statistical explanations.
- `hw3.py` is the primary graded file; the notebook supports testing and visualization.
- The assignment emphasizes both theoretical understanding and practical implementation of Bayesian models.
