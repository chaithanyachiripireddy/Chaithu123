import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Descriptive Statistics
# Generate dataset with random heights and weights
np.random.seed(0)
heights = np.random.normal(170, 10, 50)
weights = np.random.normal(70, 15, 50)

# Calculate mean, median, standard deviation, and range for heights
heights_mean = np.mean(heights)
heights_median = np.median(heights)
heights_std = np.std(heights)
heights_range = np.max(heights) - np.min(heights)

# Calculate mean, median, standard deviation, and range for weights
weights_mean = np.mean(weights)
weights_median = np.median(weights)
weights_std = np.std(weights)
weights_range = np.max(weights) - np.min(weights)

# Probability
# Define event: Probability of a person being taller than 180 cm
taller_than_180 = np.sum(heights > 180) / len(heights)

# Distributions
# Plot histograms for heights and weights
plt.hist(heights, bins=10, alpha=0.5, label='Heights')
plt.hist(weights, bins=10, alpha=0.5, label='Weights')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histograms of Heights and Weights')
plt.legend()
plt.show()

# Inferential Statistics - Central Limit Theorem (CLT)
# Randomly sample subsets of 30 observations from heights and calculate means
sample_means = [np.mean(np.random.choice(heights, 30)) for _ in range(1000)]

# Plot distribution of sample means
plt.hist(sample_means, bins=30, alpha=0.5)
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Distribution of Sample Means (n=30)')
plt.show()

# Confidence Intervals (CI)
# Calculate 95% confidence interval for mean heights
heights_ci = stats.norm.interval(0.95, loc=np.mean(heights), scale=np.std(heights) / np.sqrt(len(heights)))
print("95% Confidence Interval for Mean Heights:", heights_ci)

# Hypothesis Testing
# Formulate hypothesis: Testing if mean height is different from 170 cm
t_stat, p_value = stats.ttest_1samp(heights, 170)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Critical Regions, Level of Significance, and Error Types
# Define critical region: |T| > critical_value (for two-tailed test)
critical_value = stats.t.ppf(0.025, df=len(heights) - 1)  # For alpha = 0.05
print("Critical Value:", critical_value)

# Discuss types of errors: Type I error (false positive), Type II error (false negative)

# Feature Selection Using P-values
# Perform hypothesis test to determine if height is a significant predictor of weight
t_stat, p_value = stats.ttest_ind(heights, weights)
print("P-value for hypothesis test:", p_value)

# Decision: Include or exclude variable based on P-value
if p_value < 0.05:
    print("Height is a significant predictor of weight.")
else:
    print("Height is not a significant predictor of weight.")
