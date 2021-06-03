# **Object Detection and Tracking Unscented Kalman Filter**

## Kalman Filters variances

All Kalman filters have the same mains steps: 1. Initialization, 2. Prediction, 3. Update.
A **Standard Kalman Filter** (KF) can only handle linear equations. 
Both the **Extended Kalman Filter** (EKF) and the **Unscented Kalman Filter** (UKF) allow you to use non-linear equations; the difference between 
EKF and UKF is how they handle non-linear equations: Extended Kalman Filter uses the Jacobian matrix to 
linearize non-linear functions; Unscented Kalman Filter, on the other hand, does not need to linearize non-linear 
functions, insteadly, the unscented Kalman filter takes representative points from a Gaussian distribution.

## Unscented Kalman Filter roadmap
<img src="ukf_roadmap.jpg"/>


## Pseudocode for Update

- Create tracks if no tracks vector found.
- Calculate cost using sum of square distance between predicted vs detected centroids.
- Using Hungarian Algorithm assign the correct detected measurements to predicted tracks.
- Identify tracks with no assignment, if any.
- If tracks are not detected for long time, remove them.
- Now look for un_assigned detects.
- Start new tracks.
- Update KalmanFilter state, lastResults and tracks trace.


## Usage

1. Clone this repo.
2. Run it by the following commands: 
   * `cd Object-Detection-and-Tracking\Unscented_Kalman_Filter`
   * `python object_tracking.py`


## Libraries used:
 - `OpenCV 4.1.1`
 - `Numpy 1.19.2`
 - `SciPy 1.6.2`
