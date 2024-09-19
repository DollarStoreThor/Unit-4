Block 23: Workshop
Real Estate Distance Analysis
Problem Statement:
Imagine you are a real estate analyst. You have been provided with the coordinates (latitude and longitude) of various properties in a city. Your task is to calculate the distances between these properties to understand the spatial distribution and proximity of properties to each other. The distance metrics you should consider are:

Euclidean Distance
Manhattan (City block) Distance
Minkowski Distance
Using these distance metrics, you can determine which properties are closest to each other and how they are spread across the city.

Tasks:
Load the dataset into your preferred programming environment.
For a chosen property (say Property_ID = 1), calculate its distance to all other properties using the three distance metrics mentioned.
Determine the top 5 properties closest to the chosen property for each distance metric.
Step 1: Loading the Data

import pandas as pd
​
# Load the data
file_path = "real_estate_coordinates.csv"
df_properties = pd.read_csv(file_path)
df_properties.head()
Property_ID	Latitude	Longitude
0	1	36.872701	-99.842854
1	2	39.753572	-96.817948
2	3	38.659970	-98.428220
3	4	37.993292	-97.457147
4	5	35.780093	-95.462168
Step 2: Distance Calculation Functions

###
### YOUR CODE HERE
###
import numpy as np
​
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))
​
def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))
​
def minkowski_distance(point1, point2, p=2):
    return np.power(np.sum(np.abs(point1 - point2)**p), 1/p)
​
p1 = np.array([36.87, -99.84])
p2 = np.array([36.87, -99.84])
x = euclidean_distance(p1, p2)
​
print(x)
0.0
​
###
### AUTOGRADER TEST - DO NOT REMOVE
###
​
Step 3: Calculating Distances for Sample Property (Property_ID = 1)

###
### YOUR CODE HERE
###
​
​
p0 = np.array([df_properties['Latitude'][0], df_properties['Longitude'][0]])
'''
px = {}
for item in df_properties.index:
    px[item] = np.array([df_properties['Latitude'][item], df_properties['Longitude'][item]])
​
​
​
print(px[1])
print(euclidean_distance(p0, px[3]))
'''
​
df_properties['Euclidean_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: euclidean_distance(p0, pN), axis=1)
df_properties['Manhattan_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: manhattan_distance(p0, pN), axis=1)
df_properties['Minkowski_to_ID1'] = df_properties[['Latitude', 'Longitude']].apply(lambda pN: minkowski_distance(p0, pN, p=3), axis=1)
            
        
df_properties
Property_ID	Latitude	Longitude	Euclidean_to_ID1	Manhattan_to_ID1	Minkowski_to_ID1
0	1	36.872701	-99.842854	0.000000	0.000000	0.000000
1	2	39.753572	-96.817948	4.177257	5.905777	3.722618
2	3	38.659970	-98.428220	2.279368	3.201903	2.044030
3	4	37.993292	-97.457147	2.635778	3.506299	2.465425
4	5	35.780093	-95.462168	4.514887	5.473294	4.403226
...	...	...	...	...	...	...
95	96	37.468978	-98.253952	1.697102	2.185179	1.616415
96	97	37.613664	-96.370222	3.550803	4.213596	3.483841
97	98	37.137705	-95.514449	4.336510	4.593410	4.328736
98	99	35.127096	-95.564568	4.620700	6.023891	4.373039
99	100	35.539457	-96.100622	3.972636	5.075475	3.797811
100 rows × 6 columns

###
### AUTOGRADER TEST - DO NOT REMOVE
###
​
Step 4: Top Five Closest Properties for Each Distance Metric

###
### YOUR CODE HERE
###
​
print('\n~~~ EUCLIDEAN ~~~~ \n',df_properties.sort_values(by='Euclidean_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')
print('\n~~~ MANHATTAN ~~~ \n',df_properties.sort_values(by='Manhattan_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')
print('\n~~~ MINKOWSKI ~~~ \n',df_properties.sort_values(by='Minkowski_to_ID1').head(6).tail(), '\n-----------------------------------------------------------------\n')

~~~ EUCLIDEAN ~~~~ 
     Property_ID   Latitude  Longitude  Euclidean_to_ID1  Manhattan_to_ID1  \
23           24  36.831809 -99.449740          0.395235          0.434005   
64           65  36.404673 -99.548551          0.552869          0.762331   
48           49  37.733551 -99.742606          0.866668          0.961099   
71           72  35.993578 -99.917061          0.882249          0.953329   
24           25  37.280350 -98.860324          1.063740          1.390179   

    Minkowski_to_ID1  
23          0.393261  
64          0.503985  
48          0.861304  
71          0.879298  
24          1.005385   
-----------------------------------------------------------------


~~~ MANHATTAN ~~~ 
     Property_ID   Latitude  Longitude  Euclidean_to_ID1  Manhattan_to_ID1  \
23           24  36.831809 -99.449740          0.395235          0.434005   
64           65  36.404673 -99.548551          0.552869          0.762331   
71           72  35.993578 -99.917061          0.882249          0.953329   
48           49  37.733551 -99.742606          0.866668          0.961099   
28           29  37.962073 -99.965239          1.096225          1.211758   

    Minkowski_to_ID1  
23          0.393261  
64          0.503985  
71          0.879298  
48          0.861304  
28          1.089887   
-----------------------------------------------------------------


~~~ MINKOWSKI ~~~ 
     Property_ID   Latitude  Longitude  Euclidean_to_ID1  Manhattan_to_ID1  \
23           24  36.831809 -99.449740          0.395235          0.434005   
64           65  36.404673 -99.548551          0.552869          0.762331   
48           49  37.733551 -99.742606          0.866668          0.961099   
71           72  35.993578 -99.917061          0.882249          0.953329   
17           18  37.623782 -99.067150          1.079741          1.526786   

    Minkowski_to_ID1  
23          0.393261  
64          0.503985  
48          0.861304  
71          0.879298  
17          0.962065   
-----------------------------------------------------------------
