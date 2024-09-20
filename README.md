###    UNIT-4

`Andrew Thomas`
`Overview of Projects for Unit 4`

###    UNIT #4 : Machine Learning

###    Block 23 Workshop - Real Estate Distance Analysis

Data:  

    The dataset contains the following columns: 
    
    Property_ID: A unique identifier for each property 
    Latitude: Latitude of the property 
    Longitude: Longitude of the property 

Goals: 

    1. Loading the dataset using Numpy.
    
    2. Calculating distances using three distance metrics.
    
    3. Determining the closest 5 properties to a given.
        

###    Block 24 Workshop - Hyperparameter Tuning

Goals: 

    1. Import the necessary libraries and generate a synthetic classification dataset using make_classification from sklearn.datasets. 
    
    2. Split the data into training and testing sets using the train_test_split function.
    
    3. Create a loop to train Decision Tree Classifiers.
    
    4. Plot the training and testing accuracies against the max_depth values using matplotlib.pyplot.
    
    5. Implement hyperparameter tuning using GridSearchCV from sklearn.model_selection.
    
    6. Fit the GridSearchCV on the training data and output the best hyperparameters using best_estimator_. 
    
    7. Output the accuracy of the best estimator on both the training and testing sets. 
    
    8. Matplotlib visualizing data
    
    9. Using accuracy_score from sklearn.metrics. 
    
    10. Determine Best hyperparameters


###    Block 25 Workshop - Regression and Its Applications

Goals:
    
    1. Import the required libraries for data preparation and visualization. 
    
    2. Read the input dataset and use visualizations to understand the data. 
    
    3. Create a function create_ingredient_df() to analyze the ingredients used in each type of cuisine.
    
    4. Prepare the data for regression. 
    
    5. Train a logistic regression model. 
    
    6. Make predictions with the model and evaluate the results. 


![common ingredients](https://github.com/user-attachments/assets/b13c093a-7e63-4de0-a752-ead6edace47c)


###    Block 26 Workshop - 

1. Exploratory Data Analysis (EDA):    
   Understand the data, check for missing values, and get a dataset summary.

        - Use pandas to load the CSV dataset into a DataFrame. 
        
        - Display the initial rows using the head() method to get a glimpse of the data. 
        
        - Use the describe() method to get a summary of the dataset. 
        
        - Check for missing values using the isnull() and sum() methods. 

2. Data Visualization:
Visualize the data to get insights into the distribution of key variables and their relationship with the likelihood of having AHD.

        - Utilize seaborn and matplotlib for visualizing the distribution of key variables and their relationships with the target variable 'AHD'.

![chestpain type vs AHD - boxplot](https://github.com/user-attachments/assets/9e7c9c40-7b97-4d71-80b6-f745391ea74b)
![chestpain type vs AHD - violin plot](https://github.com/user-attachments/assets/d63636c2-fdfc-4fdc-9015-579e80179922)
![chestpain type vs AHD](https://github.com/user-attachments/assets/56a891bb-b38c-4677-a017-716e2b95f6c9)
![Correlationonal Heatmap with dendrogram groupings of AHD](https://github.com/user-attachments/assets/8a7cbac2-7834-4cf1-bf5c-22ccbca3333a)
![Correlationonal Grid](https://github.com/user-attachments/assets/31939b55-82c0-44e5-8bfe-19e4dca26963)


4. Data Cleaning and Preprocessing: 

        - Handle any missing values or anomalies and process the data to make it suitable for machine learning models. 
        
        - Encode the categorical variables using LabelEncoder from sklearn.preprocessing to convert them into numeric form so they can be used in machine learning algorithms. 
        
        - Split the dataset into training and testing sets using train_test_split. 
        
        - Scale the features to have a zero mean and unit variance using StandardScaler to ensure optimal performance of the Gaussian Naive Bayes algorithm. 

5. Model Development:
   Choose an appropriate machine learning algorithm and train the model.
    
        - Initialize the Gaussian Naive Bayes classifier. 

        - Train the model using the scaled training data. 

6. Model Evaluation:

       - Check the accuracy and performance of the model on a test set to ensure reliability.
   
       - Use classification_report from sklearn.metrics.
   
                         precision    recall  f1-score   support

           0               0.79      0.90      0.84        29
           1               0.89      0.78      0.83        32
        accuracy                               0.84        61
        macro avg          0.84      0.84      0.84        61
        weighted avg       0.84      0.84      0.84        61
  
###    Block 27 Workshop - 
###    Block 28 Workshop - 
###    Block 29 Workshop - 
###    Block 30 Workshop - 
###    Block 31 Workshop - 
###    Block 32 Workshop - 
   
    
   
   
    
    
    
    
