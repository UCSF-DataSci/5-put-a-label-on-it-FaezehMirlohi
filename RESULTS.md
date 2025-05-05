# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

-   Which metric performed best and why?

-   Which metric performed worst and why?

-   How much did the class imbalance affect the results?

-   What does the confusion matrix tell you about the model's predictions?

-   *Accuracy (0.9195) was the highest because the model correctly predicted the majority class, which is likely the "No Disease" group. High accuracy often happens in imbalanced data where the model just learns to favor the majority.*

-   *Recall (0.3239) was the lowest. This means the model missed many actual "Disease" cases. In imbalanced data, the model struggles to detect the minority class, leading to poor recall.*

-   *Imbalance impact score: 0.481 (moderate to high). Accuracy looks good, but recall and F1 are much lower. This gap shows that the model is biased toward the majority class.*

-   *The confusion matrix shows that the model is very good at identifying "No Disease" cases (1302 TN), but misses many actual "Disease" cases (96 FN vs. 46 TP). This shows the model is biased toward the majority class, confirming that class imbalance is hurting recall and sensitivity to "Disease" cases.*

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

-   Which model performed better according to AUC score?

-   Why might one model outperform the other on this dataset?

-   How did the addition of time-series features (rolling mean and standard deviation) affect model performance?

-   *XGBoost performed better with an AUC of 0.9971 vs. 0.9762 for Random Forest.*

-   *XGBoost performed better because it handles complex patterns well, uses gradient boosting to reduce errors step-by-step, is more flexible with tuning and regularization. In contrast, Random Forest is simpler and may miss subtle patterns.*

-   *The rising, falling or fluctuating in a variable help the model detect patterns over time, not just isolated values. As a result, it can better predict outcomes like disease presence.*

## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

-   Which metrics showed the most significant improvement?

-   Which metrics showed the least improvement?

-   Why might some metrics improve more than others?

-   What does this tell you about the importance of addressing class imbalance?

-   *Recall improved the most (+156.56%) — the model is much better at catching actual "Disease" cases.*

-   *Precision dropped (–41.66%) and accuracy slightly decreased (–6.45%), meaning more false positives.*

-   *Addressing class imbalance (via SMOTE) helps the model see more examples of the minority class, improving recall and F1, but may hurt precision as the model makes more positive predictions — some of them wrong.*

-   It shows that imbalanced data hides true model performance, especially for rare outcomes. Fixing the imbalance gives a fairer and more useful model, especially when recall is critical (e.g., in health).\*

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

-   What were the most important factors affecting model performance?

-   Which techniques provided the most significant improvements?

-   What would you recommend for future modeling of this dataset?

-   *Class imbalance and lack of time-based context were key. The model initially favored the majority class and missed many disease cases.*

-   *SMOTE greatly improved recall, and XGBoost with time-series features gave the best overall performance (AUC = 0.9971).*

-   Combine balanced training data with time-series features, and use models like XGBoost. Also, consider using feature scaling and threshold tuning to further improve precision–recall balance.\*