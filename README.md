# Salary-Variance-Predictor
 comprehensive predictive model using various regression techniques

Chapter 1: Introduction


In the contemporary landscape of the tech industry, understanding the dynamics of software professionals' salaries is pivotal for both employers and employees. The compensation of software professionals is influenced by a myriad of factors, including but not limited to their role, experience, the company they work for, and the market demands. This project delves into the intricate relationship between these factors and the salaries of software professionals, employing advanced data analysis and machine learning techniques to uncover patterns and insights that can guide decision-making in the realm of human resources and career development.
1.1 Aim : 
The primary aim of this project is to develop a comprehensive predictive model using various regression techniques. This model will not only forecast the salaries of software professionals based on diverse inputs like company ratings, job roles, and reporting structures but also provide insights into how these different factors weigh in determining salary scales. The project aims to harness the power of data analytics and machine learning to bring a more data-driven approach to salary predictions in the tech industry.
1.2 Objectives:
Data Preprocessing: To clean and prepare a dataset containing information about software professionals' salaries and the various factors affecting them.
Model Development and Application: To apply multiple regression models using Python and relevant libraries such as Pandas, NumPy, and Scikit-Learn. The use of Google Colab is envisaged to facilitate computational efficiency.
Performance Evaluation: To compare the effectiveness of these models in predicting salaries, using metrics like Mean Squared Error (MSE) and R-squared (R²) values.
Feature Analysis: To employ feature selection techniques to identify and analyze the most significant factors impacting software professionals' salaries.
Insight Generation: To provide actionable insights that can help in making informed decisions regarding salary structures and employee compensation in the tech industry.

Chapter 2 : Dataset Selection and Pre-Processing

2.1 Dataset Selection
The dataset titled "Software_Professional_Salaries.csv" was selected for this project. It includes variables such as 'Rating', 'Jobs', 'Salary', 'Reported', 'Location', and 'Company Name'. The dataset is expected to be rich enough to train regression models effectively, as it contains both numerical and categorical data that can correlate with a software professional's salary.
Dataset  Link: https://www.kaggle.com/datasets/whenamancodes/software-professional-salary-dataset

2.2 Pre-Processing
Pre-processing involved several steps to prepare the dataset for modeling:
Mounting Drive and Loading Data: Data was loaded into a pandas DataFrame from a CSV file stored on Google Drive.
Data Cleaning: The 'Company Name' column was dropped since it was deemed irrelevant to the prediction of salaries.
Handling Missing Values: Missing values in the 'Salary' column were imputed using the mean strategy.
Encoding Categorical Data: Categorical variables like 'Jobs' and 'Location' were transformed into numerical values using Label Encoding, and 'Jobs' was also subjected to OneHotEncoding.
Feature Selection: SelectKBest was used with f_regression to select the top 'k' features that have the most significant impact on the salary.

Chapter 3 : Implementation 

Three regression models were implemented, and their performances were evaluated:
1. Linear Regression: This model was trained using 'Rating', 'Jobs', and 'Reported' as features, with 'Salary' as the target variable.
2. Random Forest Regressor: A more complex model which can capture non-linear relationships and interactions between features.
3. Ridge Regression: This model was used to analyze the impact of regularization on the prediction accuracy and was also employed post-feature selection.
The dataset was split into a training set (80%) and a test set (20%), and the models were evaluated based on their MSE and R² on the test set.

3.1 Code and Working : 
Google Colab Link: https://colab.research.google.com/drive/1POFpBr57dguqsMf1fTrtlZgTVes_12Bk#scrollTo=3tnaiycKga0p


Conclusion 

Through meticulous data preprocessing and the application of various regression models, we endeavored to predict salaries with precision. Our analysis, powered by Python libraries like Pandas, NumPy, Scikit-Learn, and the computational prowess of Google Colab, yielded valuable insights into the factors that influence compensation.
By comparing model performance using metrics like Mean Squared Error (MSE) and R-squared (R²), we discerned the strengths and weaknesses of different regression techniques. Additionally, through feature selection, we identified the pivotal elements that play a significant role in determining salary scales.
This project's primary objective was to create a predictive model capable of accurately forecasting software professionals' salaries while shedding light on the impact of factors like company ratings, job roles, and reporting structures. It is our hope that the outcomes of this project will empower employers, employees, and HR professionals with data-driven insights to make informed decisions regarding compensation and career development in the ever-evolving tech industry.


