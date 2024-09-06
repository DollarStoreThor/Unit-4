###    UNIT-4

`Andrew Thomas`
`Overview of Projects for Unit 4`

###    UNIT #4 : Machine Learning

###    Block 23 Workshop - Real Estate Distance Analysis

Problem Statement  

    Imagine you are a real estate analyst. You have been provided with the coordinates (latitude and longitude) of various properties in a city. Your task is to calculate the distances between these properties to understand the spatial distribution and proximity of properties to each other. The distance metrics you should consider are: 
    
    Euclidean Distance 
    Manhattan (City block) Distance
    Minkowski Distance 
    Using these distance metrics, you can determine which properties are closest to each other and how they are spread across the city. 

Data:  

    The dataset contains the following columns: 
    
    Property_ID: A unique identifier for each property 
    
    Latitude: Latitude of the property 
    
    Longitude: Longitude of the property 

Goals: 

    1. Load the dataset into your preferred programming environment.
    
    2. For a chosen property (say Property_ID = 1), calculate its distance to all other properties using the three distance metrics mentioned.
    
    3. Determine the top five properties closest to the chosen property for each distance metric.

###    Block 24 Workshop - Enhancing Credit Risk Assessment With Hyperparameter Tuning

Scenario:

    You are a data scientist working for a financial institution seeking to improve its credit risk assessment process. The institution processes a substantial volume of loan applications and wants to optimize its decision-making by adopting machine learning techniques. Your task is to fine-tune a classification model to enhance its ability to predict whether a loan applicant will likely default on their loan payments. 
    
Problem Statement:
    
    The financial institution you are employed at is keen on leveraging advanced machine learning techniques to make more accurate credit risk assessments. Given a dataset containing historical loan application data along with various applicant attributes, your objective is to optimize the performance of a Decision Tree Classifier through hyperparameter tuning. 

Goals: 

    1. Import the necessary libraries and generate a synthetic classification dataset using make_classification from sklearn.datasets. Set the random seed to ensure reproducibility. 
    
    2. Split the dataset into training and testing sets using the train_test_split function from sklearn.model_selection. Use a test size of 30%.
    
    3. Create a loop to train Decision Tree Classifiers with  max_depth values ranging from 1 to 20. Calculate and store both the training and testing accuracies for each depth value.
    
    4. Plot the training and testing accuracies against the max_depth values using matplotlib.pyplot. Label the axes and provide a legend to distinguish between training and testing accuracies. 
    
    5. Implement hyperparameter tuning using GridSearchCV from sklearn.model_selection. Define a parameter grid with options for criterion, max_depth, and min_samples_split. Use 3-fold cross-validation and consider all available CPU cores for parallel processing (n_jobs=-1).
    
    6. Fit the GridSearchCV on the training data and output the best hyperparameters using best_estimator_. 
    
    7. Calculate and print the accuracy of the best estimator on both the training and testing sets. 
    
    8. Utilize matplotlib to create a bar plot that displays the accuracy of the best estimator on the training and testing sets. Label the bars as "Train" and "Test". Provide a title and a ylabel for the plot. 
    
    9. Print the accuracy score of the best estimator's predictions on both the training and testing sets using accuracy_score from sklearn.metrics. 
    
    10. Run the code to observe the accuracy scores, the impact of different max depths, the results after hyperparameter tuning, and the final classification accuracy on both training and testing sets. 
   
    
   
   
    
    
    
    
