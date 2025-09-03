# Day 10. Regression metrics
## Today's objective
Understand and apply the most common evaluation metrics for regression models to measure how well they predict continuous values.

## Explanation 
In regression tasks, we don’t measure accuracy (like in classification). Instead, we use metrics that capture errors between the predicted values and the actual ones:

- **Mean Absolute Error (MAE):** Average of absolute errors. Easy to interpret.
- **Mean Squared Error (MSE):** Penalizes larger errors more strongly (squares the error).
- **Root Mean Squared Error (RMSE):** Square root of MSE, interpretable in original units.
- **R² Score (Coefficient of Determination):** Measures how much variance is explained by the model (closer to 1 is better).

## Small note 
>Always compare models using multiple metrics.
For example:
>- A low MAE might look good, but if R² is close to 0, the model isn’t explaining much variance.
>- RMSE is often preferred in practice because it penalizes big mistakes more 🌱