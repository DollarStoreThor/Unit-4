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
  
###    Block 27 Workshop - Classification and Applications Part II

1. Data Preprocessing:

   Clean and process the dataset, handling missing values and encoding categorical variables.

        - There are some ‘?’ instead of values in all columns. Replace them with the mode values.
        - Use fillna().

2. Feature Engineering:
   
   Create new features if relevant and apply feature scaling.

   
4. Exploratory Data Analysis (EDA):

   Visualize the distribution of income levels and relationships between features and income.

        - Fix the imbalanced dataset using the oversampling method.
        - Use RandomOverSampler from imblearn.over_sampling. 
        - Split the predictor and target variables and fit them using fit().
    ![Class imbalance in inital dataset for our desired prediction](https://github.com/user-attachments/assets/007541ef-f692-462d-aa2f-6a7fbf61a74d)
   ![Dataset Age distribution](https://github.com/user-attachments/assets/257dcbe7-9895-434f-a879-fca30563d43a)
   ![Dataset Age and Working Class distribution - Violin Plot](https://github.com/user-attachments/assets/eae1c3cc-2632-49bb-96cc-ed7010de214a)
    ![Age count - Hue 50k or more](https://github.com/user-attachments/assets/8ec35a80-81b6-488a-97d2-84b5532dc8be)
    ![Education - Hue 50k or more](https://github.com/user-attachments/assets/285d0bf8-eb78-4509-bcf2-f48f64a85dc4)
    ![Eductaion type - hue of 50k or more](https://github.com/user-attachments/assets/2f329ca7-22b5-43ca-92b1-7d2d1e619a11)
    ![Eductaion vs count - hue of 50k or more](https://github.com/user-attachments/assets/2b3d39dd-e256-4aab-a931-f3e0e892dee4)

    ![Dataset Age and Eductaion distribution - Scatter Plot](https://github.com/user-attachments/assets/ee3372bf-1e59-480e-a469-174480438d46)
    
    ![Correlational Heatmap between numeric Datatypes](https://github.com/user-attachments/assets/e1ec9634-bcb7-4daf-8edb-a9cf9b92fec1)




4. Model Selection and Training:

   Choose KNN, Naive Bayes, and Random Forest as classification algorithms and train them on the data.

        - Import the models from sklearn.linear_model, sklearn.neighbor, sklearn.svm, sklearn.naive_bayes, sklearn.tree, and sklearn.ensemble library.
        
        - Use .fit() to fit the training data.

6. Model Evaluation:

   Evaluate model accuracy and F1 score.

        - Import the models from sklearn.metrics.    

7. Hyperparameter Tuning:

   Optimize model hyperparameters for improved performance.
   
8. Model Comparison and Insights:

   Compare algorithm performance and provide insights on strengths and weaknesses.

![Comparing Model Performance - Higher is better](https://github.com/user-attachments/assets/5b455625-b85a-42c6-8509-d99018d82ceb)

    - AUC Model Performace informs us that the Random Forest Classifier, using the Random Oversampling performed the best compared to all other models.
   
9. Visualization and Reporting:

     Create visualizations and a comprehensive report.

        - Use classification_report from sklearn.metrics.

Confusion Matrix Before Hyperparameter Tuning:

![Final Model Confusion Matrix Before Hyperparameter Tuning](https://github.com/user-attachments/assets/b5f81c66-8bc3-4bd5-94ca-39b98ed1393f)

![Confusion Matrix before and after Hyperparameter](https://github.com/user-attachments/assets/02abeb25-8f6f-484a-a059-9285eddef408)

Confusion Matrix After Hyperparameter Tuning:

![Final Model Confusion Matrix After Hyperparameter Tuning](https://github.com/user-attachments/assets/948aa258-11f3-4227-81c9-40679bcaaf4d)



###    Block 28 Workshop - 

Goals:  

        1. Import packages import pandas as pd, import pyplot as plt, from sklearn.cluster import KMeans, and from sklearn.metrics import silhouette_score.
        
        2. Load the dataset.
        
        3. Check for missing values using the isnull() method.
        
        4. Select the relevant features and standardize the data.
        
        5. Choose the optimal K value and determine it using the elbow method.
        
        6. Plot the elbow curve using the plot loop through K values and calculate silhouette scores.
        
        7. Initialize K-means, fit the model, and assign cluster labels.
        
        8. Calculate cluster centers using inverse() method, and save the result in a DataFrame.
        
        9. Create a scatter plot to visualize the clusters and centroids.
        
        10. Loop through clusters to extract insights for each cluster.
        
        11. Display the summary statistics for each cluster (e.g., number of customers, average annual income, average spending score, etc.).
        
        12. Add the cluster labels to the original data and adjust cluster labels to start from 1.
        
        13. Compute summary statistics for each cluster using the groupby() method.
    
    
###    Block 29 Workshop - 
###    Block 30 Workshop - 
###    Block 31 Workshop - 
###    Block 32 Workshop - 
   
    
   
   
    
    
    
    
