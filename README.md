# Credit Risk Classification

## Executive Summary
The goal of this project is to strategically analyze the most accurate machine learning algorithm
for determining if a customer will receive a loan based on an external number of factors. The idea is to
predict the behavior mathematically with a series of popular machine learning techniques. Banks/loan providers
are able to use these techniques to decide whether or not a consumer is eligible to receive a loan.

## Outline
The solution included cleaning data by filling in the missing na values with the mean for any of the available features. 'loan_status'
was selected as the dependent variable for the analysis. One-hot-encoding was used to transform categorical variables to be processed numerically. Then,
descriptive statistical analysis was conducted on the dataset for a more specified understanding of the features.
Given that this is a classification problem (whether or not the customer will be provided a loan: 0 or 1 binary), only classification algorithms
were used including KNeighborsClassifier, SVC, RandomForestClassifier, AdaBoostClassifier, SGDClassifier, XGBClassifier, GaussianNB, and the
DecisionTreeClassifier. Feature importance was selected between ultimately running models inclusive of all features versus the most important/relevant features.


## Dataset Description
The dataset is a kaggle dataset with 12 features. Numerical features include 
'person_age', 'person_income', 'person_emp_length', 'loan_amnt','loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length.'
Categorical features include 'person_home_ownership', 'loan_intent','loan_grade','loan_status','cb_person_default_on_file.' The dataset has 32581 observations
and 12 columns.

## Methodology

### 1. Data Wrangling/Cleaning

Na values were filled with their respective means for features 'person_emp_length' and 'loan_int_rate.' Features were separated into two areas:
numerical and categorical to reproduce separate analyses. 

### 2. Exploratory Data Analysis

Exploratory data analysis was conducted on the numerical variables. A heatmap was used to identify if there was correlation between the features;
then histograms and boxplots were used to understand the distribution of the data. Loan_status was also identified based on the loan_intent in a barplot.
The skew and kurtosis were also retrieved for the numerical features to determine if there were any outliers/potential errors in the data.

### 3. Data Preprocessing

Data was transformed using the one-hot-encoder function and categorical/numerical variables were placed in the transformer pipeline for preprocessing.
The datatypes were factorized to return numerical representations of the columns for better processing. Data was split into a test/train dataset and scaled
with the StandardScaler() method for better accuracy results. Loan_status was selected as the dependent variable for analysis.

### 4. Feature Selection

The in-built class features_importances_ was used along with the XGBClassifier, RandomForestClassifier, ExtraTreesClassifier, and the DecisionTreesClassifier
to find the most optimal features to be used in the model. Chi-squared statistics and the ANOVA F-value was calculated as well to find the best features for the model.

### 5. Model Tuning/Selection

KNeighborsClassifier, SVC, RandomForestClassifier, AdaBoostClassifier, SGDClassifier, XGBClassifier, GaussianNB were the classification algorithms used for analysis.
Ideally, the classification metric used was accuracy to evaluate the algorithms. Each classifier was ran twice: one model that included all variables and another model
that simply included the best features after running the statistical tests.

## Results/Conclusion

the XGBClassifier with selected important variables was the most accurate model with an accuracy rate of 92%. If the models hyperparameters were also optimized,
it is possible that the accuracy could slightly increase.
