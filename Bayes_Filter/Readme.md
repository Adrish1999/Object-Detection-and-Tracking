# The Bayes Filter
In probability theory, statistics, and machine learning, recursive Bayesian estimation, also known as a Bayes filter, is a general probabilistic approach for estimating an unknown Probability Density Function (PDF) recursively over time using incoming measurements and a mathematical process model.

## Filter Equations:
 x&‌#772; = x * f<sub>x</sub>( &‌#8729; )     Predict Step
 x = &‌#8739;&‌#8739; L &‌#8901; x&‌#772; &‌#8739;&‌#8739;       Update Step

 L is the likelihood function. The &‌#8739;&‌#8739; &‌#8739;&‌#8739; notation denotes taking the norm. We need to normalize the product of the likelihood with the prior to ensure x is a probability distribution that sums to one.

## Pseudocode:

### Initialization

1. Initialize our belief in the state.

### Predict

1. Based on the system behavior, predict state for the next time step.
2. Adjust belief to account for the uncertainty in prediction.

### Update

1. Get a measurement and associated belief about its accuracy.
2. Compute how likely it is the measurement matches each state.
3. Update state belief with this likelihood.

## Libraries used:
 - `OpenCV 4.1.1`
 - `Numpy 1.19.2`
 - `Matplotlib 3.1.1`
 - `Filterpy 1.4.5`