# Job Eligibility Matching with LightGBM

This project aims to create a job eligibility function that matches job requirements with candidate eligibilities using a LightGBM classifier. The goal is to identify potential candidates who are a good fit for specific job openings based on their skills, location, type of employment, and experience.

## Project Overview

The project involves comparing two datasets: one containing job requirements and the other containing candidate eligibilities. By analyzing and matching relevant features from both datasets, a predictive model is built to determine the likelihood of a candidate being eligible for a job.

## Data

The project uses two datasets:

- `1.csv`: Contains job requirements, including `job_id`, `required_skill`, `location`, `currency`, `years` of experience required, and `type_of_employment`.
- `2.csv`: Contains candidate eligibilities, including `user_id`, `location`, `type_of_employment`, `currency`, `experience`, and `skills`.

## Methodology

The project utilizes the following steps:

1.  **Data Loading and Understanding**: Load and explore the structure and content of both datasets.
2.  **Feature Engineering**: Create relevant features for the model by:
    - Handling missing values in location and employment type.
    - Parsing and standardizing the 'years' and 'experience' columns.
    - Creating a function to match skills between job requirements and candidate skills.
    - Creating binary features for location and employment type matching.
    - Calculating the difference between candidate experience and required years.
3.  **Target Variable Creation**: Define a binary target variable 'eligible' based on specific criteria (e.g., matching skills and at least one other matching criterion like location, employment type, or sufficient experience).
4.  **Data Splitting**: Split the engineered data into training and testing sets for model development and evaluation. Stratification is used to handle class imbalance in the target variable.
5.  **LightGBM Model Training**: Train a LightGBM classifier on the training data. `scale_pos_weight` is used to address the class imbalance.
6.  **Model Evaluation**: Evaluate the trained model's performance using metrics such as accuracy, precision, recall, and F1-score on the testing data.
7.  **Eligibility Prediction**: Use the trained model to predict the eligibility of candidates for all job-candidate pairs.

## Dependencies

The project requires the following libraries:

- pandas
- numpy
- scikit-learn
- lightgbm

These can be installed using pip:
