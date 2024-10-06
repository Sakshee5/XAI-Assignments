# Global Explanations using PDP (+ local ICE) and ALE for Automobile Dataset

## Overview
This repository demonstrates use of Partial Dependence Plots (PDP) (+ Individual Conditional Expectation (ICE) Plots), and Accumulated Local Effects (ALE) Plots to provide global explanations for a Random Forest Regression model. The model predicts the miles per galon for an automobile dataset, which contains data on car specifications and performance indicators. The primary goal is to explore how various features impact miles per gallon (mpg), a measure of fuel efficiency.

## Dataset
The dataset chosen includes the following features:

- mpg: Miles per gallon (target variable)
- cylinders: Number of engine cylinders
- displacement: Engine displacement (size)
- horsepower: Engine power
- weight: Vehicle weight
- acceleration: Time to accelerate from 0 to 60 mph
- year: Year the car was manufactured
- origin: Country of origin (coded as a numerical value)
- name: Car make and model (not used in the model)


## Explanatory Techniques
1. Partial Dependence Plots (PDP)

show the marginal effect of one or two features on the predicted outcome. They allow us to observe how mpg changes with a particular feature while averaging out the effects of all other features.

2. Individual Conditional Expectation (ICE) Plots

display how predictions for individual data points change as a feature value varies.

3. Accumulated Local Effects (ALE) Plots

provide a more accurate way to visualize feature effects in the presence of correlated features. Unlike PDP, ALE does not assume independence between features and avoids the "average" problem of PDP when features are correlated.

## Correlation Exploration
exploratory analysis of feature correlation is perfomed to understand how input features might interact and affect our results, focusing on the extent to which PDPs can fail in the presence of correlated features.