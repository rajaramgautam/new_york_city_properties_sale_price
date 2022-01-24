
# Sale Price Prediction on Properties of New York City based on 12 month period DataSet

Rajaram Gautam - January 24, 2022 - Submitted To: Codeup Data Science Team

# Project Summary

# Business Goals

- To construct a Machine Learning Regression Model that can predict the sale price of Properties of New York City based on 12 months period sales using various attributes of the properties.
- To find key drivers of the sale price for the Properties.
- Deliver a report that any data science team can read and understands about the procedures taken to come to conclusion.
- To make recommendation based on my works what would work and what not in the predictions of the properties sale price.

# Executive Summary

After modeling the New York City Property data using five features (Age, Total Units, Land Square Feet, and Gross Square Feet), the Polynomial Model with degree 2 produced the best results with RMSE values of 285585, 283355 and 284915 for train, validate and test dataset. 

It is improvement by 10.78% over the baseline model (when comparing training data set). It was improved by 11.49 % for validate datadet. Therefore, these models do show that they could be used to produce a prediction for home values; however, the error is still high at over $284915 RSME. I would recommed further cleaning data for a reliable predictor for sale price in order to use this model.
If I have more time, I would clean the dataset for best result possible to reduce RMSE value for my models.

# Deliverables

- A report in the form of presnetation and review  of two peers work presentation.
- Github Repository with a complete readme.md, a final report(.ipynb), wrangle, explore, and model modules made to make workflow in project pipeline easy.
- The report will summarize the findings about the drivers of sale price of the properties with suitable visualizations.

# Intial Questions
- Is there positive correlation between sale price of Properties and Total Units?
- Is there positive correlation between sale price of Properties and Land Square Feet?
- Is there positive correlation between sale price of Properties and Gross Square Feet?
- Is there positive correlation between sale price of Properties and Age?

###### Hypothesis:
- HO: Age and Sale Price has not linear relation.
- H1: Age and Sale Price has linear relation.

# Data dictionary
|Index | Column Name | Description 
|---|---|---|
|0 |  borough                          | The name of the borough in which the property is located                                
|1 |  neighborhood                     | Neighborhood the properties is located in                                
|2 |  tax_class_at_present             | Square footage of the properties                        
|3 |  zip_code                         | Zip Code of the Property                              
|4 |  residential_units                | The number of residential units at the listed property                            
|5 |  commercial_units                 | The number of commercial units at the listed property.                            
|6 |  total_units                      | The total number of units at the listed property. 
|7 |  land_square_feet                 | The land area of the property listed in square feet                                      |8 |  gross_square_feet                | The total area of all the floors of a building                    
|9 |  tax_class_at_time_of_sale        | Every property in the city is assigned to one of four tax classes                        |10|  building_class_at_time_of_sale   | Building Class at the time of sale / Building Zone                                       
|11|  sale_price                       | Sale Price of the Property                                     
|12|  sale_date                        | Transaction date for the property 
|13|  age                              | Life of the property = Current Year - Built_Year 

For details on data dictionary, https://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf
 

# Project Specifications

### Plan:

- Properties data to extracted from New York City Property (.csv file downloaded from kaggle) provided with suitable attritues that will help us in determing sale price of the properties.
- Prepare: Prepare data to ensure that data format of each attriutes selected will fit into our model, remove outliers, handle NaN in data properly using best judgement to so that our model will give give us less error.
- Explore: Explore the data for attribute that will have possible relationship with sale price and remove the attributes in that have colinearly dependent to each other.
- Model : We will use Linear Regression Model like Ordinar Least Squares, LASSO + LARS, Polynomial Regression Generalized Linear Model (TweedieRegressor)
- Iterate : We will iterate the process for various model with different values of model parameter to choose the best model for our test data, to make better predictions.

- Trello Planning Board
![Getting Started](trello.png)

![alternative text](trello.png "Trellp Planning Board")


### Acquire

I created a series of functions to acquire and clean the New York City Property data, which are located in the New York City Property_wrangle.py file. They take in all of the single unit properties (code 261 from the propertylandusetypeid column on the properties_2017 table in the New York City Property data set) that were sold between 1 May and 1 Sept of 2017 (based on the predictions_2017 table) into a Pandas dataFrame.
I have teken a column for zipcodes and modified it to average properties price per zipcode, as well as the county and sate name.
### Prep

Top and Bottom of outliers were removed so that when we take median as baseline it will give us less error.
Features and columns that seems to have no influence in sale price has been dropped and even those of extracted some are removed as they seemed irrelevant and two columns have same meaning.

### Explore

Scatterplots with trend lines are used to look for obvious correlations.
Boxplots are used to show the distributions and possible outliers even after preparing.
A heatmap was generated to show correlation strength between features.
Statistical tests were used to confirm correlation between the target and other features.


# Model & Evaluate

The following models were used:
Baseline (using mean)
Ordinary Least Squares
LASSO + LARS
Generalized Linear Model
Polynomial 2nd Degree


# Reproduce My Project

- Read this README.md
- Download the nyc-rolling-sales.csv, wrangle.py, model.py final.ipynb files into your working directory
- Run the final_report.ipynb notebook
