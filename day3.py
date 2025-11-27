"""
PROBLEM:
We are given a joint probability table of two discrete random variables X and Y.
Each row contains: X value, Y value, and P(X, Y).
We must:
1. Check if X and Y are independent:
   X and Y are independent if for every pair (x,y):
        P(X=x, Y=y) = P(X=x) * P(Y=y)
2. Compute covariance Cov(X, Y).
3. Compute correlation corr(X, Y).

Return a dictionary:
{
    "are_independent": True/False,
    "cov": numeric_covariance,
    "corr": numeric_correlation
}

---------------------------------------------------------------------

- First compute marginal probabilities P(X) and P(Y)
- Independence holds only if every joint probability equals P(X)*P(Y)
- Using the probability distribution (not samples):
      μx = Σ P(X=x)*x
      μy = Σ P(Y=y)*y
- Covariance:
      Cov(X,Y) = Σ P(x,y) * (x - μx) * (y - μy)
- Standard deviation:
      σx = sqrt( Σ P(X=x)*(x - μx)^2 )
      σy = sqrt( Σ P(Y=y)*(y - μy)^2 )
- Correlation:
      corr = Cov(X,Y) / (σx * σy)
"""

import pandas as pd
import numpy as np


class CheckIndependence:

    def __init__(self):
        self.version = 1

    def check_independence(self, distr_table: pd.DataFrame):

        # Compute marginal distributions P(X) and P(Y)
        marginal_prob_x = distr_table.groupby("X")["pr"].sum().to_dict()
        marginal_prob_y = distr_table.groupby("Y")["pr"].sum().to_dict()

        # Test independence: joint must equal product of marginals
        variables_are_independent = True

        for _, row in distr_table.iterrows():
            x_value = row["X"]
            y_value = row["Y"]
            joint_probability = row["pr"]

            expected_joint_if_independent = (
                marginal_prob_x[x_value] * marginal_prob_y[y_value]
            )

            # If mismatch → X and Y are NOT independent
            if not np.isclose(joint_probability, expected_joint_if_independent):
                variables_are_independent = False
                break

        # Compute means (μx and μy)
        mean_x = sum(x_val * prob for x_val, prob in marginal_prob_x.items())
        mean_y = sum(y_val * prob for y_val, prob in marginal_prob_y.items())

        # Compute covariance using probability distribution
        covariance_xy = 0
        for _, row in distr_table.iterrows():
            covariance_xy += (
                row["pr"]
                * (row["X"] - mean_x)
                * (row["Y"] - mean_y)
            )

        # Compute variances for X and Y
        variance_x = sum(
            prob * (x_val - mean_x) ** 2
            for x_val, prob in marginal_prob_x.items()
        )
        variance_y = sum(
            prob * (y_val - mean_y) ** 2
            for y_val, prob in marginal_prob_y.items()
        )

        std_x = np.sqrt(variance_x)
        std_y = np.sqrt(variance_y)

        # Compute correlation
        if std_x == 0 or std_y == 0:
            correlation_xy = 0
        else:
            correlation_xy = covariance_xy / (std_x * std_y)

        # Return results
        return {
            "are_independent": variables_are_independent,
            "cov": covariance_xy,
            "corr": correlation_xy
        }


# --------------------- TEST CASES ---------------------

checker = CheckIndependence()

# Test Case 1 (NOT independent)
table1 = pd.DataFrame({
    "X": [0, 0, 1, 1],
    "Y": [1, 2, 1, 2],
    "pr": [0.3, 0.25, 0.15, 0.3]
})
print("Test Case 1:", checker.check_independence(table1))

# Test Case 2 (Independent)
table2 = pd.DataFrame({
    "X": [0, 0, 1, 1],
    "Y": [0, 1, 0, 1],
    "pr": [0.2, 0.3, 0.2, 0.3]   # = P(X)*P(Y)
})
print("Test Case 2:", checker.check_independence(table2))

# Test Case 3 (Perfect correlation X=Y)
table3 = pd.DataFrame({
    "X": [0, 1, 2],
    "Y": [0, 1, 2],
    "pr": [0.2, 0.5, 0.3]
})
print("Test Case 3:", checker.check_independence(table3))
