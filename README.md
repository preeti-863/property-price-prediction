# property-price-prediction
Machine Learning model for property worth prediction using scikit-learn and pandas.

## Results 
### Simple Linear Regression:
- **R² Score**: 0.4285
- **RMSE**: 89,288.33

### Multiple Linear Regression:
- **R² Score**: 0.5317
- **RMSE**: 80,824.93

### Top Feature Coefficients:
**Feature**                             **Coefficient**
longitude                               -88544.174059
latitude                                -73392.434408
ocean_proximity_NEAR BAY                -71177.709801
proximity_score                          57346.070474
location_cluster                        -39515.469287
median_income                            38475.332922
housing_median_age                       14058.761778
ocean_proximity_NEAR OCEAN               6915.819663
households                               93.877013
total_bedrooms                           86.758855

### Visualization
![Predicted vs Actual Prices]
<img width="991" height="743" alt="image" src="https://github.com/user-attachments/assets/25cc74ff-23b0-4a60-99e6-f1ef37266791" />
