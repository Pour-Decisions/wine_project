# Pour_Decisions


# Project Description:

* Following the Data Science pipeline, build a machine learning model that accurately predicts wine quality rating.

# Goals:

* Acquire the data
* Prepare the data 
* Explore the data to find drivers of our target variable (quality rating)
* Use clustering to discover patterns 
* Use drivers to build models
* Validate, and then test our best model
* Deliver findings to a group of fellow data scientists


# Data Dictionary

| Feature | Description |
| ------ | ----|
| fixed acidity | most acids involved with wine or fixed or nonvolatile (do not evaporate readily) (tartaric acid - g / dm^3)   |
| volatile acidity | the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste (acetic acid - g / dm^3) |
| citric acid | found in small quantities, citric acid can add ‘freshness’ and flavor to wines (g / dm^3) |
 | residual sugar | the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet (g / dm^3) |
| chlorides | the amount of salt in the wine (sodium chloride - g / dm^3) |
| free sulfur dioxide | the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine (mg / dm^3) |
| total sulfur dioxide | amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine (mg / dm^3) |
| density | the density of water is close to that of water depending on the percent alcohol and sugar content (g / cm^3) |
| pH | describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale |
| sulphates | a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant (potassium sulphate - g / dm3) |
| alcohol | the percent alcohol content of the wine (% by volume) |
| color | color of the wine (red or white) |
| quality | rated by sensory observation of expert (score between 0 and 10) TARGET VALUE |

# Steps to Reproduce


* Clone my repo including the acquire.py, prepare.py, explore.py, and modeling.py (make sure to create a .gitignore to hide your csv files)

* Run notebook


# Takeaways and Conclusions

* PolynomialFeatures was the best performing model, reducing RMSE from the baseline of 0.85, to 0.68 on the validate set. 

* This resulted in a 20% reduction in root mean squared error on our validate set.

# Recommendations

* Aquire additional features such as type of grape, yeast type, fermentation temp, etc. 

# Next Steps

* With more time, we would like to explore additional feature engineering.

=======
This is a repo for: Quit Your Wine-ing: What makes a high quality wine?

# Wine Data Dictionary

This dataset contains the following columns:

| Column Name          | Data Type | Description                                                                                       |
|----------------------|-----------|---------------------------------------------------------------------------------------------------|
| fixed acidity        | str       | Measure of the concentration of fixed acids in the wine                                          |
| volatile acidity     | str       | Measure of the concentration of volatile acids in the wine                                       |
| citric acid          | str       | Measure of the concentration of citric acid in the wine                                           |
| residual sugar       | int       | Amount of sugar that remains in the wine after fermentation is complete                          |
| chlorides            | float     | Measure of the concentration of chloride ions in the wine                                         |
| free sulver dioxide  | str       | Measure of the concentration of sulfur dioxide that is not bound to other compounds in the wine  |
| total sulfer dioxide | str       | Measure of the total concentration of sulfur dioxide in the wine, including both free and bound forms |
| density              | str       | Measure of the mass per unit volume of the wine                                                   |
| pH                   | str       | Measure of the acidity of the wine                                                                 |
| sulphates            | str       | Measure of the concentration of sulfur dioxide and sulfate ions in the wine                      |
| alcohol              | str       | Percentage of alcohol by volume in the wine                                                       |
| quality              | str       | Subjective measure of the quality of the wine                                                      |
| wine                 | str       | Categorical variable indicating the type of wine, such as red or white                           |
