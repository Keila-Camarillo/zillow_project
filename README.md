# Zillow Property Value Prediction

# Project Description
This project aims to analyze the Zillow dataset, a real estate marketplace dataset, to discover the drivers of property value for single-family properties. 

# Project Goal
* Discover drivers of property value in the Zillow dataset. 
* Use drivers to develop a machine learning model to predict the property value for a single family property.
* Property value is defined as an estimated value for a property.
* This information could be used to further our understanding of which elements contribute to the value of a property.

# Initial Thoughts

The initial hypothesis for this project is that certain factors such as 'area', 'bathrooms', 'year', 'pool', 'bedroom', 'two_bed', 'one_bath','fips', and 'year' may be significant drivers of property value.

# The Plan

# Acquire
    * Obtain the dataset from the Codeup database.
    * The initial dataset contained 52,442 rows and 7 columns before cleaning.
    * Each row represents a property listed on Zillow.
    * Each column represents a specific feature of the properties.

# Prepare
    * Perform the following preparation actions on the dataset:
        * Filter out columns that do not provide useful information.
        * Rename columns to improve readability.
        * Handle null values:
            * Drop rows with null values in columns 'year', 'area', and 'property_value'.
            * Estimate that properties with null values in the 'pool' column do not have pools.
    * Check and adjust column data types as needed.
    * Add the 'full_bath' column to differentiate homes with only full bathrooms.
    * Encode categorical variables.
    * Split the data into training, validation, and test sets (approximately 60/20/20 split).
    * Remove outliers (2641 outliers removed based on falling outside 3 standard deviations).

- Create Engineered columns from existing data
    * property_value
    * area
    * bathrooms
    * bedroom
    * fips
    * year
    * full_bath
    * orange
    * ventura
* Explore data in search of drivers of upsets

- Answer the following initial questions
    * Does area sqft affect property value ?
    * Is the average area for properties in Orange county greater than the overall average area?
    * Is having 1 bathroom worse for property value than having 2 bedrooms?
    * Is a pool better for property value?
    * Does bedrooms affect property value?
    * Does bathrooms affect property value?
    * Does year affect property value?
    * Does pool affect property value?
    * Do homes in Orange county affect property value?
    * Do homes in Ventura county affect property value?

* Develop a Model to predict property

    - Use drivers identified in explore to build predictive models of different types
    - Evaluate models on train and validate data
    - Select the best model based on lowest RMSE and highest R2
    - Evaluate the best model on test data

* Draw conclusions



# Data Dictionary
Here is a data dictionary describing the features in the dataset:

| Feature | Definition |
|:--------|:-----------|
|Property Value (Target)| Value of a single family property|
|Area | The calculated finished square footage|
|Bathrooms| The number of bathrooms, (half baths: .5 & three-quarters bath: .75|
|Bedroom| 0 (No) or 1 (Yes), The number of bedrooms|
|Pool| 0 (No) or 1 (Yes), The home has a pool|
|Year| The year the home was built|
|Full Bath| 0 (No) or 1 (Yes), The home has only full bathrooms|
|Fips| 6037: "LA", 6059: "Orange", 6111: "Ventura"|
|Orange| 0 (No) or 1 (Yes), The home has is located in Orange county|
|Ventura| 0 (No) or 1 (Yes), The home has is located in Ventura county|

# Steps to Reproduce
    1. Clone this repo.
    2. Acquire the data from Codeup Database - create username and password
    3. Place the data in the file containing the cloned repo.
    4. Run notebook.

# Takeaways and Conclusions
* 'Area' was found to have a correlation with 'Property Value' 
* 'Full Bathrooms' was found to be a driver of 'Property Value', properties with more 'Full Bathrooms' tend to have lower property value, than homes with half bathrooms.
* 'Bedrooms' was found to have a correlation with 'Property Value'
* 'Pool' was found to be a driver of 'Property Value', properties with more 'Pools' tend to have higher property value, than homes with no 'pools'.
* 'Year' was found to have a correlation with 'Property Value'
* 'Orange County' were found to a moderate correlation with 'Property Value'
* 'Ventura County' having a weak significant correlation with 'Property Value'

# Recommendations
* To increase the model performance an additional feature
    - Properties with or without an HOA
    - Properties with or without a basement 
    - Property exterior features: siding, roof type, and floor type
    - Population density within the area
